"""
playback_engine.py – движок воспроизведения для ProxyPlayer v1.4.
audio_clock идёт по системному времени независимо от AudioOutput.
Устранена зависимость от Qt-потока для таймера.
Добавлен метод set_audio_output для ленивого подключения AudioOutput.
"""

import time
import threading
import logging
from typing import Optional

import numpy as np

from buffer.frame_buffer import FrameRingBuffer
from buffer.audio_buffer import MultiTrackAudioBuffer
from core.sync_manager import SyncManager
from pipeline.chunk_pipeline import ChunkPipeline
from seek.seek_engine import SeekEngine
from output.audio_output import AudioOutput
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
        audio_output: Optional[AudioOutput],
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

        self.playing = False
        self._paused = False
        self._audio_clock = 0
        self._clock_lock = threading.Lock()
        self._current_frame_idx = 0

        # Основа для хода audio_clock по системному времени
        self._clock_start_time: Optional[float] = None
        self._clock_start_pts: int = 0

        # JKL
        self._seek_speed = 1.0
        self._seek_speed_index = -1
        self._seek_direction = 0
        self._last_seek_time = 0.0
        self._seek_accumulator = 0.0
        self._normal_playing_state = False

        self._video_ready_timer: Optional[threading.Timer] = None
        self._playback_started = False

        # Защита от повторного seek
        self._seeking = False
        self._seek_lock = threading.Lock()

    # ------------------------------------------------------------------
    def set_audio_output(self, audio_output: AudioOutput):
        """Подключает AudioOutput, созданный в главном потоке."""
        self._audio_output = audio_output

    def _update_clock_from_system(self):
        """Вычисляет audio_clock, прошедший с момента старта/возобновления."""
        if self._clock_start_time is not None and self.playing:
            elapsed = time.monotonic() - self._clock_start_time
            with self._clock_lock:
                self._audio_clock = self._clock_start_pts + int(AUDIO_SAMPLE_RATE * elapsed)

    def _reset_clock_timer(self, pts: int):
        """Запоминает текущее системное время и соответствующий pts."""
        self._clock_start_time = time.monotonic()
        self._clock_start_pts = pts
        with self._clock_lock:
            self._audio_clock = pts

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
        self._reset_clock_timer(pts)
        self._current_frame_idx = global_start_frame

        self._video_buffer.clear()
        self._audio_buffers.clear_all()

        self._pipeline.start(start_local_chunk=local_chunk)

        waited = 0.0
        while self._video_buffer.count == 0 and waited < 5.0:
            time.sleep(0.1)
            waited += 0.1

        first = self._video_buffer.peek_first()
        if first is not None:
            self._video_buffer.update_keep_last(first[1], first[0])

        self.playing = False
        self._paused = True

    def resume(self):
        if not self._paused:
            return
        self._paused = False
        self.playing = True
        self._reset_clock_timer(self._audio_clock)   # начинаем отсчёт с текущего audio_clock
        if self._audio_output:
            self._audio_output.reset_clock(self._audio_clock)
        self._sync.reset_drift()
        if self._audio_output and not self._audio_output._active:
            self._audio_output.start()
        self._start_audio_when_video_ready()

    def pause(self):
        if not self.playing:
            return
        self.playing = False
        self._paused = True
        if self._audio_output:
            self._audio_output.stop()
        self._stop_video_ready_timer()

    def stop(self):
        self.playing = False
        self._paused = False
        self._pipeline.stop()
        if self._audio_output:
            self._audio_output.stop()
        self._stop_video_ready_timer()

    # ------------------------------------------------------------------
    def seek(self, global_frame_idx: int, window: 'IndexWindow'):
        with self._seek_lock:
            if self._seeking:
                logger.debug("Seek already in progress, ignoring")
                return
            self._seeking = True

        self.pause()
        self._seek_engine.seek_async(
            global_frame_idx,
            on_complete=lambda buf: self._on_seek_complete(buf, window),
            on_error=lambda msg: logger.error(f"Seek error: {msg}"),
        )

    def _on_seek_complete(self, buffer: FrameRingBuffer, window: 'IndexWindow'):
        self._video_buffer, buffer = buffer, self._video_buffer
        buffer.clear()

        first = self._video_buffer.peek_first()
        if first:
            pts, frame = first
            self._video_buffer.update_keep_last(frame, pts)
            with self._clock_lock:
                self._audio_clock = pts
            self._current_frame_idx = pts_to_video_frame(pts)
            self._audio_buffers.clear_all()
            if self._audio_output:
                self._audio_output.reset_clock(pts)

        local_chunk = (self._current_frame_idx - window.window_start_frame) // 12
        self._pipeline.stop()
        self._pipeline.update_window(window)
        self._pipeline.start(start_local_chunk=local_chunk)
        self._paused = True
        self.playing = False

        with self._seek_lock:
            self._seeking = False

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
            if self._audio_output:
                self._audio_output.set_volume(0.0)
        self._seek_accumulator = 0.0
        self._last_seek_time = time.monotonic()

    def reset_speed(self):
        if self._seek_speed == 1.0 and self._seek_direction == 0:
            return
        self._seek_speed = 1.0
        self._seek_speed_index = -1
        self._seek_direction = 0
        self._seek_accumulator = 0.0
        if self._audio_output:
            self._audio_output.set_volume(0.8)
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
        self._update_clock_from_system()  # обновляем audio_clock перед каждым кадром

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
        self._current_frame_idx = frame_idx
        pts = video_frame_to_pts(frame_idx)
        with self._clock_lock:
            self._audio_clock = pts
        self._video_buffer.drop_until(pts)
        first = self._video_buffer.peek_first()
        if first:
            self._video_buffer.update_keep_last(first[1], first[0])

    @property
    def audio_clock(self) -> int:
        with self._clock_lock:
            return self._audio_clock

    # ------------------------------------------------------------------
    def _start_audio_when_video_ready(self):
        self._stop_video_ready_timer()
        first = self._video_buffer.peek_first()
        if first and first[0] <= self._audio_clock + SAMPLES_PER_VIDEO_FRAME // 2:
            if self._audio_output:
                self._audio_output.start()
            return
        self._video_ready_timer = threading.Timer(0.01, self._check_video_ready)
        self._video_ready_timer.start()

    def _check_video_ready(self):
        if not self.playing:
            return
        first = self._video_buffer.peek_first()
        if first and first[0] <= self._audio_clock + SAMPLES_PER_VIDEO_FRAME // 2:
            if self._audio_output:
                self._audio_output.start()
        else:
            self._video_ready_timer = threading.Timer(0.01, self._check_video_ready)
            self._video_ready_timer.start()

    def _stop_video_ready_timer(self):
        if self._video_ready_timer:
            self._video_ready_timer.cancel()
            self._video_ready_timer = None

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
        with self._seek_lock:
            self._seeking = False
        self._seek_engine.cancel_current()