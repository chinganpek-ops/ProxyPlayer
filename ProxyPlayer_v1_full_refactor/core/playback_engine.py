"""
playback_engine.py – движок воспроизведения для ProxyPlayer v1.
Объединяет SeekEngine, ChunkPipeline, SyncManager и AudioOutput.
Реализует: start_playback, pause, resume, stop, seek, set_speed.
"""

import time
import threading
import logging
from typing import Optional, List, Callable

import numpy as np

from buffer.frame_buffer import FrameRingBuffer
from buffer.audio_buffer import MultiTrackAudioBuffer
from core.sync_manager import SyncManager
from pipeline.chunk_pipeline import ChunkPipeline
from seek.seek_engine import SeekEngine
from output.audio_output import AudioOutput  # будет создан позже или заменён на мок
from config.timebase import (
    SAMPLES_PER_VIDEO_FRAME,
    video_frame_to_pts,
    pts_to_video_frame,
)

logger = logging.getLogger(__name__)

SEEK_SPEEDS = [2.0, 4.0, 8.0]          # множители для JKL
LIVE_SEEK_OFFSET_FRAMES = 1600


class PlaybackEngine:
    """
    Управляет состоянием воспроизведения.
    Не зависит от GUI – получает команды и отдаёт кадры.
    """

    def __init__(
        self,
        pipeline: ChunkPipeline,
        seek_engine: SeekEngine,
        sync_manager: SyncManager,
        audio_output: AudioOutput,
        video_buffer: FrameRingBuffer,
        audio_buffers: MultiTrackAudioBuffer,
        start_frame_offset: int = 0,
        total_frames: int = 0,
        fps: float = 25.0,
    ):
        self._pipeline = pipeline
        self._seek_engine = seek_engine
        self._sync = sync_manager
        self._audio_output = audio_output
        self._video_buffer = video_buffer
        self._audio_buffers = audio_buffers

        self.start_frame_offset = start_frame_offset
        self.total_frames = total_frames
        self.fps = fps

        # Состояние
        self.playing = False
        self._paused = False
        self._audio_clock = 0
        self._clock_lock = threading.Lock()
        self._current_frame_idx = 0

        # JKL-перемотка
        self._seek_speed = 1.0
        self._seek_speed_index = -1
        self._seek_direction = 0
        self._last_seek_time = 0.0
        self._seek_accumulator = 0.0
        self._normal_playing_state = False

        # Ожидание видео при старте аудио
        self._video_ready_timer: Optional[threading.Timer] = None

    # ------------------------------------------------------------------
    # Управление воспроизведением
    # ------------------------------------------------------------------
    def start_playback(self, start_frame_idx: int):
        """Инициализирует буферы и запускает конвейер. После вызова плеер на паузе."""
        with self._clock_lock:
            self._audio_clock = video_frame_to_pts(start_frame_idx)
        self._current_frame_idx = start_frame_idx

        self._video_buffer.clear()
        self._audio_buffers.clear_all()

        self._pipeline.start(start_chunk=start_frame_idx // 12)
        self.playing = False
        self._paused = True

    def resume(self):
        """Запускает воспроизведение после паузы."""
        if not self._paused:
            return
        self._paused = False
        self.playing = True
        self._audio_output.reset_clock(self._audio_clock)
        self._sync.reset_drift()
        self._start_audio_when_video_ready()

    def pause(self):
        """Ставит на паузу."""
        if not self.playing:
            return
        self.playing = False
        self._paused = True
        self._audio_output.stop()
        self._stop_video_ready_timer()

    def stop(self):
        """Останавливает воспроизведение и конвейер."""
        self.playing = False
        self._paused = False
        self._pipeline.stop()
        self._audio_output.stop()
        self._stop_video_ready_timer()

    def toggle_play_pause(self):
        if self.playing:
            self.pause()
        elif self._paused:
            self.resume()

    # ------------------------------------------------------------------
    # Перемотка
    # ------------------------------------------------------------------
    def seek(self, frame_idx: int):
        """Перемотка на указанный кадр."""
        self.pause()
        self._seek_engine.seek_async(
            frame_idx,
            on_complete=self._on_seek_complete,
            on_error=lambda msg: logger.error(f"Seek error: {msg}"),
        )

    def _on_seek_complete(self, buffer: FrameRingBuffer):
        """Колбэк после успешного seek – подменяет буфер и запускает конвейер."""
        # Замена буфера
        self._video_buffer, buffer = buffer, self._video_buffer
        buffer.clear()

        first = self._video_buffer.peek_first()
        if first:
            pts, _ = first
            self._video_buffer.update_keep_last(first[1], pts)
            self._audio_clock = pts
            self._current_frame_idx = pts_to_video_frame(pts)
            self._audio_buffers.clear_all()
            self._audio_output.reset_clock(pts)

        self._pipeline.stop()
        self._pipeline.start(start_chunk=self._current_frame_idx // 12)
        self._paused = True
        self.playing = False

    # ------------------------------------------------------------------
    # JKL-перемотка
    # ------------------------------------------------------------------
    def set_speed(self, direction: int):
        """J (назад) или L (вперёд)."""
        if self.total_frames == 0:
            return
        if self._seek_direction != direction:
            self._seek_speed_index = 0
            self._seek_direction = direction
        else:
            self._seek_speed_index = min(self._seek_speed_index + 1, 2)
        self._seek_speed = SEEK_SPEEDS[self._seek_speed_index]
        if self._seek_speed_index == 0:
            self._normal_playing_state = self.playing
            self._audio_output.set_volume(0.0)
        self._seek_accumulator = 0.0
        self._last_seek_time = time.monotonic()

    def reset_speed(self):
        """K – сброс скорости."""
        if self._seek_speed == 1.0 and self._seek_direction == 0:
            return
        self._seek_speed = 1.0
        self._seek_speed_index = -1
        self._seek_direction = 0
        self._seek_accumulator = 0.0
        self._audio_output.set_volume(0.8)
        if self._normal_playing_state:
            if not self.playing:
                self.resume()
        else:
            if self.playing:
                self.pause()

    def get_speed_display(self) -> str:
        if self._seek_direction == 0:
            return "▶ x1"
        direction = "<<" if self._seek_direction < 0 else ">>"
        return f"{direction} x{self._seek_speed:.0f}"

    # ------------------------------------------------------------------
    # Получение кадра (вызывается из GUI по таймеру)
    # ------------------------------------------------------------------
    def get_display_frame(self) -> Optional[np.ndarray]:
        # Режим JKL
        if self._seek_direction != 0 and self._seek_speed > 1.0:
            now = time.monotonic()
            dt = now - self._last_seek_time
            self._last_seek_time = now
            frames_to_skip = self._seek_speed * self.fps * dt
            self._seek_accumulator += frames_to_skip
            if self._seek_accumulator >= 1.0:
                skip = int(self._seek_accumulator)
                self._seek_accumulator -= skip
                target = self._current_frame_idx + skip * self._seek_direction
                target = max(0, min(target, self.total_frames - 1))
                self._fast_seek(target)
            return self._video_buffer.get_keep_last()

        return self._sync.get_display_frame(
            self._video_buffer,
            self._audio_buffers,
            audio_clock=self._audio_clock,
            playing=self.playing,
        )

    def _fast_seek(self, frame_idx: int):
        """Быстрое перемещение без полного перестроения буфера."""
        self._current_frame_idx = frame_idx
        pts = video_frame_to_pts(frame_idx)
        with self._clock_lock:
            self._audio_clock = pts
        self._video_buffer.drop_until(pts)
        first = self._video_buffer.peek_first()
        if first:
            self._video_buffer.update_keep_last(first[1], first[0])
        # Перезапускаем конвейер с нового места (без очистки аудио)
        self._pipeline.stop()
        self._pipeline.start(start_chunk=frame_idx // 12)

    @property
    def audio_clock(self) -> int:
        with self._clock_lock:
            return self._audio_clock

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------
    def _start_audio_when_video_ready(self):
        """Запускает аудиовыход, когда в буфере есть кадр с подходящим PTS."""
        self._stop_video_ready_timer()
        first = self._video_buffer.peek_first()
        if first and first[0] <= self._audio_clock + SAMPLES_PER_VIDEO_FRAME // 2:
            self._audio_output.start()
            return
        # Иначе запускаем таймер проверки
        self._video_ready_timer = threading.Timer(0.01, self._check_video_ready)
        self._video_ready_timer.start()

    def _check_video_ready(self):
        if not self.playing:
            return
        first = self._video_buffer.peek_first()
        if first and first[0] <= self._audio_clock + SAMPLES_PER_VIDEO_FRAME // 2:
            self._audio_output.start()
        else:
            # Повторная проверка через 10 мс
            self._video_ready_timer = threading.Timer(0.01, self._check_video_ready)
            self._video_ready_timer.start()

    def _stop_video_ready_timer(self):
        if self._video_ready_timer:
            self._video_ready_timer.cancel()
            self._video_ready_timer = None

    # ------------------------------------------------------------------
    # Таймкоды (для GUI)
    # ------------------------------------------------------------------
    def get_local_timecode_str(self) -> str:
        idx = pts_to_video_frame(self._audio_clock)
        total_seconds = idx / self.fps
        h = int(total_seconds // 3600)
        m = int((total_seconds % 3600) // 60)
        s = int(total_seconds % 60)
        f = int(round((total_seconds - int(total_seconds)) * self.fps))
        return f"{h:02d}:{m:02d}:{s:02d};{f:02d}"

    def get_real_timecode_str(self) -> str:
        idx = pts_to_video_frame(self._audio_clock) + self.start_frame_offset
        total_seconds = idx / self.fps
        h = int(total_seconds // 3600)
        m = int((total_seconds % 3600) // 60)
        s = int(total_seconds % 60)
        f = int(round((total_seconds - int(total_seconds)) * self.fps))
        return f"{h:02d}:{m:02d}:{s:02d};{f:02d}"

    def close(self):
        self.stop()
        self._seek_engine.cancel_current()