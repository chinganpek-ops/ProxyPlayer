"""
stream_controller.py – фасад для всех новых компонентов ProxyPlayer v1.
Заменяет PlayerController из v6. Предоставляет единый API для GUI.
"""

import time
import threading
import logging
from pathlib import Path
from typing import Optional, List

import numpy as np

from buffer.frame_buffer import FrameRingBuffer
from buffer.audio_buffer import MultiTrackAudioBuffer
from decode.decoder import Decoder
from decode.audio_decoder import AudioDecoder
from index.lazy_index import LazyIndex
from index.idx_cache import prepare_mirror
from file_io.win_sequential_reader import WinSequentialReader
from utils.utils import get_real_size
from pipeline.chunk_pipeline import ChunkPipeline
from seek.seek_engine import SeekEngine
from core.sync_manager import SyncManager
from core.playback_engine import PlaybackEngine
from output.audio_output import AudioOutput
from config.timebase import (
    SAMPLES_PER_VIDEO_FRAME, video_frame_to_pts, pts_to_video_frame,
)
from file_io.ref_parser import extract_ftyp_avcc
from utils.tcd_parser import parse_tcd

logger = logging.getLogger(__name__)

DEFAULT_AVCC = bytes.fromhex(
    "014d001fffe1002e674d401f9652816824dff80200016a50101014000003"
    "000400000300cb8180009600000301e848fc6383b428532c01000568e9093520"
)
DEFAULT_ASC = b'\x11\x88'

LIVE_SEEK_OFFSET_FRAMES = 1600


class StreamController:
    """Фасад, заменяющий PlayerController."""

    def __init__(self, ref_path: Path, idx_path: Path, mp4_path: Path,
                 fps: float = 25.0, buffer_size: int = 600,
                 free_slots_required: int = 25,
                 group_chunks: int = 30,
                 max_retries: int = 3,
                 thread_type: str = "AUTO",
                 thread_count: int = 0,
                 skip_frame: bool = False,
                 gpu_mode: str = "off",
                 avcc_override: Optional[bytes] = None,
                 audio_delay_ms: int = 0,
                 start_from_live: bool = True):
        self.mp4_path = mp4_path
        self.idx_path = idx_path
        self.ref_path = ref_path
        self.fps = fps
        self.buffer_size = buffer_size
        self._start_from_live = start_from_live
        self.audio_delay_samples = int(audio_delay_ms * 48000 / 1000)
        self.start_frame_offset = 0
        self.total_frames = 0
        self._finalized = False

        # Буферы
        self.buffer_main = FrameRingBuffer(max_frames=buffer_size)
        self.audio_buffers = MultiTrackAudioBuffer(capacity_samples=1440000)

        # Декодеры
        avcc = avcc_override or self._load_avcc()
        self.decoder = Decoder(avcc, mp4_path, thread_type=thread_type,
                               thread_count=thread_count, skip_frame=skip_frame,
                               gpu_mode=gpu_mode)
        self.audio_decoders = [AudioDecoder(DEFAULT_ASC), AudioDecoder(DEFAULT_ASC)]

        # Аудиовыход (создаётся позже)
        self.audio_output: Optional[AudioOutput] = None
        self.active_tracks = [2, 3]

        # Индекс и конвейер
        self._lazy_index: Optional[LazyIndex] = None
        self._pipeline: Optional[ChunkPipeline] = None
        self._seek_engine: Optional[SeekEngine] = None
        self._playback: Optional[PlaybackEngine] = None
        self._sync_mgr = SyncManager()

        self.playing = False
        self._paused = False
        self._ready = threading.Event()
        self._init_error: Optional[str] = None

        # Инициализация в фоне
        self._init_thread = threading.Thread(target=self._background_init, daemon=True)
        self._init_thread.start()

    # ------------------------------------------------------------------
    def _load_avcc(self) -> bytes:
        if self.ref_path and self.ref_path.exists():
            try:
                _, avcc = extract_ftyp_avcc(self.ref_path)
                return avcc
            except Exception:
                pass
        return DEFAULT_AVCC

    def _background_init(self):
        try:
            mirror = prepare_mirror(self.idx_path)
            mdat_end = get_real_size(str(self.mp4_path))

            self._lazy_index = LazyIndex(mirror, self.mp4_path, mdat_end)

            # Определяем стартовый кадр
            if self._start_from_live and not self._finalized:
                self.total_frames = len(self._lazy_index._video_records_full)
                start_frame = max(0, self.total_frames - LIVE_SEEK_OFFSET_FRAMES)
            else:
                self.total_frames = len(self._lazy_index._video_records_full)
                start_frame = 0

            window = self._lazy_index.open_window(start_frame)
            self.total_frames = len(self._lazy_index._video_records_full)

            # TCD
            try:
                tcd = self.mp4_path.with_suffix('.tcd')
                if not tcd.exists():
                    tcd = self.idx_path.parent / f"{self.mp4_path.stem}.tcd"
                h, m, s, f = parse_tcd(tcd)
                self.start_frame_offset = int((h * 3600 + m * 60 + s) * self.fps + f)
            except Exception:
                pass

            # Создаём конвейер и движок
            self._pipeline = ChunkPipeline(
                self.mp4_path, window, self.decoder, self.audio_decoders,
                self.buffer_main, self.audio_buffers,
            )

            reader = WinSequentialReader(self.mp4_path, 0, False)
            self._seek_engine = SeekEngine(self._lazy_index, self.decoder, reader)

            self._playback = PlaybackEngine(
                self._pipeline, self._seek_engine, self._sync_mgr,
                self.audio_output or AudioOutput(self.audio_buffers),  # временная заглушка
                self.buffer_main, self.audio_buffers,
                start_frame_offset=self.start_frame_offset,
                total_frames=self.total_frames,
                fps=self.fps,
            )

            self._ready.set()
            logger.info("StreamController готов")
        except Exception as e:
            self._init_error = str(e)
            logger.exception("Ошибка инициализации StreamController")

    # ------------------------------------------------------------------
    # API, совместимый с PlayerController
    # ------------------------------------------------------------------
    def start_playback(self):
        if not self._ready.is_set():
            return
        start_frame = 0
        if self._start_from_live and not self._finalized:
            start_frame = max(0, self.total_frames - LIVE_SEEK_OFFSET_FRAMES)
        self._playback.start_playback(start_frame)
        self.playing = False
        self._paused = True

    def resume(self):
        if self._playback:
            self._playback.resume()
            self.playing = True
            self._paused = False

    def pause(self):
        if self._playback:
            self._playback.pause()
            self.playing = False
            self._paused = True

    def stop(self):
        if self._playback:
            self._playback.stop()
            self.playing = False
            self._paused = False

    def toggle_pause(self):
        if self.playing:
            self.pause()
        elif self._paused:
            self.resume()

    def seek_absolute(self, frame_idx: int):
        if self._playback:
            self._playback.seek(frame_idx)

    def seek_relative(self, delta_sec: float):
        if self._playback:
            frame = pts_to_video_frame(self._playback.audio_clock)
            target = frame + int(round(delta_sec * self.fps))
            target = max(0, min(target, self.total_frames - 1))
            self.seek_absolute(target)

    def set_seek_speed(self, direction: int):
        if self._playback:
            self._playback.set_speed(direction)

    def reset_seek_speed(self):
        if self._playback:
            self._playback.reset_speed()

    def get_seek_speed_display(self) -> str:
        if self._playback:
            return self._playback.get_speed_display()
        return "▶ x1"

    def get_display_frame(self) -> Optional[np.ndarray]:
        if self._playback:
            return self._playback.get_display_frame()
        return None

    @property
    def audio_clock(self) -> int:
        if self._playback:
            return self._playback.audio_clock
        return 0

    def get_local_timecode_str(self) -> str:
        if self._playback:
            return self._playback.get_local_timecode_str()
        return "00:00:00;00"

    def get_real_timecode_str(self) -> str:
        if self._playback:
            return self._playback.get_real_timecode_str()
        return "00:00:00;00"

    def set_active_tracks(self, tracks: List[int]):
        self.active_tracks = [t for t in tracks if t in (2, 3)]
        if self.audio_output:
            self.audio_output.set_active_tracks(self.active_tracks)

    def close(self):
        if self._playback:
            self._playback.close()
        if self.decoder:
            self.decoder.close()