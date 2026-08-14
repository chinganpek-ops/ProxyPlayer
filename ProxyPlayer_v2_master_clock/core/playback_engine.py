"""
playback_engine.py – движок воспроизведения с тройным буфером и фильтрацией старых кадров.

Исправления:
- Тройной буфер: display_buffer, fill_buffer, free_buffer.
- При seek конвейер НЕ останавливается, только перенаправляется в новый буфер.
- Звук не прерывается, отклик быстрый.
- Эталонный PTS устанавливается ДО переключения буферов, чтобы старые пакеты
  отбрасывались ещё на этапе вставки.
"""

import time
import threading
import logging
from typing import Optional

import numpy as np

from buffer.frame_buffer import FrameRingBuffer
from core.sync_manager import SyncManager
from pipeline.chunk_pipeline import ChunkPipeline
from seek.seek_engine import SeekEngine
from core.master_clock import MasterClock
from config.timebase import (
    AUDIO_SAMPLE_RATE,
    SAMPLES_PER_VIDEO_FRAME,
    video_frame_to_pts,
    pts_to_video_frame,
)

logger = logging.getLogger(__name__)

SEEK_SPEEDS = [2.0, 4.0, 8.0]


class PlaybackEngine:
    def __init__(
        self,
        pipeline: ChunkPipeline,
        seek_engine: SeekEngine,
        sync_manager: SyncManager,
        master_clock: Optional[MasterClock],
        video_buffer: FrameRingBuffer,
        start_frame_offset: int = 0,
        total_frames: int = 0,
        fps: float = 25.0,
    ):
        self._pipeline = pipeline
        self._seek_engine = seek_engine
        self._sync = sync_manager
        self._master_clock = master_clock
        self._video_buffer = video_buffer

        # Тройной буфер
        self._display_buffer = video_buffer
        self._fill_buffer = video_buffer
        self._free_buffer = FrameRingBuffer(max_frames=video_buffer.max_frames)
        self._buffer_lock = threading.Lock()
        self._pending_clear: Optional[FrameRingBuffer] = None

        self.start_frame_offset = start_frame_offset
        self.total_frames = total_frames
        self.fps = fps

        self.playing = False
        self._paused = False
        self._audio_clock = 0
        self._clock_lock = threading.Lock()
        self._current_frame_idx = 0

        # JKL
        self._seek_speed = 1.0
        self._seek_speed_index = -1
        self._seek_direction = 0
        self._last_seek_time = 0.0
        self._seek_accumulator = 0.0
        self._normal_playing_state = False

        self._playback_started = False

        # Защита от повторного seek и поколение запросов
        self._seek_generation = 0
        self._seek_lock = threading.Lock()

        # Эталонный PTS для фильтрации старых кадров после seek
        self._seek_pts_reference = 0

    # ------------------------------------------------------------------
    def set_master_clock(self, master_clock: MasterClock):
        self._master_clock = master_clock

    # ------------------------------------------------------------------
    def start_playback(self, global_start_frame: int, window_start_frame: int):
        if self._playback_started:
            logger.warning("start_playback вызван повторно, игнорируем")
            return
        self._playback_started = True

        if self._pipeline:
            self._pipeline.stop()

        local_chunk = (global_start_frame - window_start_frame) // 12
        pts = video_frame_to_pts(global_start_frame)
        with self._clock_lock:
            self._audio_clock = pts
        self._current_frame_idx = global_start_frame

        # Сбрасываем эталонный PTS для обычного воспроизведения
        self._seek_pts_reference = 0
        self._pipeline.set_seek_reference(0)
        self._sync.set_seek_reference(0)

        self._display_buffer = self._fill_buffer
        self._fill_buffer.clear()
        self._free_buffer.clear()

        self._pipeline.start(start_local_chunk=local_chunk)

        waited = 0.0
        while self._fill_buffer.count == 0 and waited < 5.0:
            time.sleep(0.1)
            waited += 0.1

        first = self._fill_buffer.peek_first()
        if first is not None:
            self._display_buffer.update_keep_last(first[1], first[0])

        self.playing = False
        self._paused = True

    def resume(self):
        if not self._paused:
            return
        self._paused = False
        self.playing = True
        if self._master_clock:
            self._master_clock.start()
        self._sync.reset_drift()

    def pause(self):
        if not self.playing:
            return
        self.playing = False
        self._paused = True
        if self._master_clock:
            self._master_clock.stop()

    def stop(self):
        self.playing = False
        self._paused = False
        self._pipeline.stop()
        if self._master_clock:
            self._master_clock.stop()

    # ------------------------------------------------------------------
    def seek(self, global_frame_idx: int, window: 'IndexWindow',
             on_complete: Optional[callable] = None):
        with self._seek_lock:
            self._seek_generation += 1
            gen = self._seek_generation
            self.pause()

        def _on_seek_complete_with_callback(buf, win, g):
            self._on_seek_complete(buf, win, g)
            if on_complete:
                on_complete()

        self._seek_engine.seek_async(
            global_frame_idx,
            on_complete=lambda buf: _on_seek_complete_with_callback(buf, window, gen),
            on_error=lambda msg: logger.error(f"Seek error: {msg}"),
        )

    def _on_seek_complete(self, buffer: FrameRingBuffer, window: 'IndexWindow', gen: int):
        with self._seek_lock:
            if gen != self._seek_generation:
                return  # устаревший запрос

        # НОВОЕ: устанавливаем эталон ДО всех манипуляций с буферами
        first_entry = buffer.peek_first()
        if first_entry:
            new_reference = first_entry[0]
            self._seek_pts_reference = new_reference
            self._pipeline.set_seek_reference(new_reference)
            self._sync.set_seek_reference(new_reference)

        # 1. Очищаем free_buffer
        self._free_buffer.clear()

        # 2. Переносим кадры из временного буфера в free_buffer
        while True:
            entry = buffer.peek_first()
            if entry is None:
                break
            pts, frame = entry
            if not self._free_buffer.try_push(frame, pts):
                break
            buffer.advance()

        # 3. Атомарно переключаем буферы
        with self._buffer_lock:
            old_display = self._display_buffer

            self._display_buffer = self._free_buffer
            self._fill_buffer = self._free_buffer
            self._free_buffer = old_display

            self._pending_clear = old_display

        # 4. Запускаем фоновую очистку старого буфера
        def _clear_old_buffer():
            if self._pending_clear:
                buf = self._pending_clear
                self._pending_clear = None
                buf.clear()

        threading.Thread(target=_clear_old_buffer, daemon=True).start()

        # 5. Обновляем конвейер и планировщик
        self._pipeline.update_window(window)
        self._pipeline.set_video_buffer(self._fill_buffer)
        try:
            self._pipeline.flush()
        except AttributeError:
            logger.warning("Метод flush() отсутствует в ChunkPipeline, пропускаем очистку очередей")

        first = self._display_buffer.peek_first()
        if first:
            pts, frame = first
            self._display_buffer.update_keep_last(frame, pts)
            with self._clock_lock:
                self._audio_clock = pts
            self._current_frame_idx = pts_to_video_frame(pts)
            if self._master_clock:
                self._master_clock.set_clock(pts)

        local_chunk = (self._current_frame_idx - window.window_start_frame) // 12
        self._pipeline._scheduler.set_normal_mode(local_chunk, window.total_chunks)

        self._paused = True
        self.playing = False

    # ------------------------------------------------------------------
    # JKL
    # ------------------------------------------------------------------
    def set_speed(self, direction: int):
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
            if self._master_clock:
                self._master_clock.set_muted(True)
        self._seek_accumulator = 0.0
        self._last_seek_time = time.monotonic()

    def reset_speed(self):
        if self._seek_speed == 1.0 and self._seek_direction == 0:
            return
        self._seek_speed = 1.0
        self._seek_speed_index = -1
        self._seek_direction = 0
        self._seek_accumulator = 0.0
        if self._master_clock:
            self._master_clock.set_muted(False)
        if self._normal_playing_state and not self.playing:
            self.resume()
        elif not self._normal_playing_state and self.playing:
            self.pause()

    def get_speed_display(self) -> str:
        if self._seek_direction == 0:
            return "▶ x1"
        direction = "<<" if self._seek_direction < 0 else ">>"
        return f"{direction} x{self._seek_speed:.0f}"

    # ------------------------------------------------------------------
    def get_display_frame(self) -> Optional[np.ndarray]:
        if self._master_clock:
            with self._clock_lock:
                self._audio_clock = self._master_clock.get_audio_clock()

        with self._buffer_lock:
            display_buf = self._display_buffer

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
            return display_buf.get_keep_last()

        return self._sync.get_display_frame(
            display_buf,
            audio_clock=self._audio_clock,
            playing=self.playing,
        )

    def _fast_seek(self, frame_idx: int):
        with self._seek_lock:
            self._current_frame_idx = frame_idx
            pts = video_frame_to_pts(frame_idx)
            with self._clock_lock:
                self._audio_clock = pts
        self._display_buffer.drop_until(pts)
        first = self._display_buffer.peek_first()
        if first:
            self._display_buffer.update_keep_last(first[1], first[0])

    @property
    def audio_clock(self) -> int:
        with self._clock_lock:
            return self._audio_clock

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