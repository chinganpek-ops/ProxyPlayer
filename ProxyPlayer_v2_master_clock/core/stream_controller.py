"""
stream_controller.py – фасад для всех компонентов ProxyPlayer v1.
Архитектурное изменение v1.1:
- плеер больше не вызывает prepare_mirror (зеркалом владеет IndexService/менеджер).
- использует переданный mirror_path или ожидает готовности зеркала.
- защита от повторного запуска start_playback.
- обновление окна в пайплайне при live-старте.
- AudioOutput создаётся в главном потоке (в start_playback).
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
from index.lazy_index import LazyIndex, IndexWindow
from index.idx_cache import get_mirror_path, open_idx_mmap, cleanup_cache
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
                 start_from_live: bool = True,
                 mirror_path: Optional[str] = None):
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
        self.mirror_path = mirror_path

        # Буферы
        self.buffer_main = FrameRingBuffer(max_frames=buffer_size)
        self.audio_buffers = MultiTrackAudioBuffer(capacity_samples=1440000)

        # Декодеры – создаются в _background_init
        self.decoder: Optional[Decoder] = None
        self.audio_decoders: List[AudioDecoder] = []
        self.audio_output: Optional[AudioOutput] = None
        self.active_tracks = [2, 3]

        # Компоненты новой архитектуры – инициализируются в _background_init
        self._lazy_index: Optional[LazyIndex] = None
        self._pipeline: Optional[ChunkPipeline] = None
        self._seek_engine: Optional[SeekEngine] = None
        self._playback: Optional[PlaybackEngine] = None
        self._sync_mgr = SyncManager()

        # Флаги состояния
        self.playing = False
        self._paused = False
        self._ready = threading.Event()
        self._init_error: Optional[str] = None
        self._closed = False
        self._playback_started = False  # защита от повторного запуска

        # Фоновая инициализация
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
        """Фоновая инициализация всех компонентов с защитой от ошибок."""
        try:
            # 1. Определяем путь к зеркалу
            if self.mirror_path:
                mirror = Path(self.mirror_path)
                if not mirror.exists():
                    raise RuntimeError(f"Переданное зеркало не существует: {mirror}")
            else:
                mirror = get_mirror_path(self.idx_path)
                remote_size = self.idx_path.stat().st_size if self.idx_path.exists() else 0
                if remote_size == 0:
                    raise RuntimeError(f"Индексный файл {self.idx_path} пуст или не существует")
                waited = 0.0
                while waited < 30.0:
                    if mirror.exists() and mirror.stat().st_size >= remote_size:
                        break
                    time.sleep(0.5)
                    waited += 0.5
                if not mirror.exists() or mirror.stat().st_size < remote_size:
                    raise RuntimeError("Зеркало индекса не готово")

            # 2. Получаем mdat_end
            mdat_end = get_real_size(str(self.mp4_path))
            if mdat_end == 0:
                raise RuntimeError(f"Не удалось определить размер MP4: {self.mp4_path}")

            # 3. Создаём LazyIndex
            self._lazy_index = LazyIndex(mirror, self.mp4_path, mdat_end)

            # 4. Определяем стартовый кадр
            if self._start_from_live and not self._finalized:
                self.total_frames = len(self._lazy_index._video_records_full)
                start_frame = max(0, self.total_frames - LIVE_SEEK_OFFSET_FRAMES)
            else:
                self.total_frames = len(self._lazy_index._video_records_full)
                start_frame = 0

            if self.total_frames == 0:
                raise RuntimeError("Индекс не содержит видео-записей")

            window = self._lazy_index.open_window(start_frame)
            if window is None or len(window.video_records) == 0:
                raise RuntimeError("Не удалось открыть окно индекса")

            # 5. Парсим TCD
            try:
                tcd = self.mp4_path.with_suffix('.tcd')
                if not tcd.exists():
                    tcd = self.idx_path.parent / f"{self.mp4_path.stem}.tcd"
                h, m, s, f = parse_tcd(tcd)
                self.start_frame_offset = int((h * 3600 + m * 60 + s) * self.fps + f)
            except Exception as e:
                logger.warning(f"Не удалось прочитать TCD: {e}")

            # 6. Создаём декодеры
            avcc = self._load_avcc()
            self.decoder = Decoder(
                avcc, self.mp4_path,
                thread_type="AUTO", thread_count=0,
                skip_frame=False, gpu_mode="off"
            )
            self.audio_decoders = [AudioDecoder(DEFAULT_ASC), AudioDecoder(DEFAULT_ASC)]

            # 7. Создаём конвейер
            self._pipeline = ChunkPipeline(
                self.mp4_path, window, self.decoder, self.audio_decoders,
                self.buffer_main, self.audio_buffers,
            )

            # 8. Создаём Reader и SeekEngine
            reader = WinSequentialReader(self.mp4_path, 0, False)
            self._seek_engine = SeekEngine(self._lazy_index, self.decoder, reader)

            # 9. Создаём PlaybackEngine (без AudioOutput – он будет подключён позже)
            self._playback = PlaybackEngine(
                self._pipeline, self._seek_engine, self._sync_mgr,
                None,          # audio_output пока None
                self.buffer_main, self.audio_buffers,
                start_frame_offset=self.start_frame_offset,
                total_frames=self.total_frames,
                fps=self.fps,
            )

            self._ready.set()
            logger.info("StreamController готов (без AudioOutput)")
        except Exception as e:
            self._init_error = str(e)
            logger.exception("Ошибка инициализации StreamController")
            self._ready.set()

    # ------------------------------------------------------------------
    def _ensure_ready(self) -> bool:
        """Проверяет, что все компоненты инициализированы и готовы."""
        if self._closed:
            logger.warning("StreamController закрыт")
            return False
        if not self._ready.is_set():
            logger.warning("StreamController ещё не готов")
            return False
        if self._init_error is not None:
            logger.error(f"StreamController инициализирован с ошибкой: {self._init_error}")
            return False
        if self._playback is None or self._pipeline is None:
            logger.warning("PlaybackEngine или ChunkPipeline не инициализированы")
            return False
        return True

    # ------------------------------------------------------------------
    def start_playback(self):
        if not self._ensure_ready():
            return
        if self._playback_started:
            logger.warning("start_playback вызван повторно, игнорируем")
            return
        self._playback_started = True

        try:
            # Открываем окно для актуального стартового кадра (live или начало)
            if self._start_from_live and not self._finalized:
                start_frame = max(0, self.total_frames - LIVE_SEEK_OFFSET_FRAMES)
            else:
                start_frame = 0

            window = self._lazy_index.open_window(start_frame)
            if window is None:
                logger.error("Окно индекса не открыто")
                return

            # Обновляем окно в пайплайне
            if self._pipeline:
                self._pipeline.update_window(window)

            # --- Создаём AudioOutput в ГЛАВНОМ ПОТОКЕ (start_playback вызывается из GUI) ---
            if self.audio_output is None:
                try:
                    self.audio_output = AudioOutput(
                        self.audio_buffers,
                        on_clock_update=lambda samples: setattr(self._playback, '_audio_clock', samples)
                    )
                    self._playback.set_audio_output(self.audio_output)
                    # Устанавливаем активные треки
                    self.audio_output.set_active_tracks(self.active_tracks)
                    logger.info("AudioOutput создан и подключён в главном потоке")
                except Exception as e:
                    logger.error(f"Не удалось создать AudioOutput: {e}")
                    self.audio_output = None
            # ----------------------------------------------------------------------------

            # Останавливаем пайплайн, если он был запущен (на всякий случай)
            if self._pipeline:
                self._pipeline.stop()

            self._playback.start_playback(start_frame, window.window_start_frame)
            self.playing = False
            self._paused = True
        except Exception as e:
            logger.exception("Ошибка в start_playback")

    def resume(self):
        if not self._ensure_ready():
            return
        try:
            self._playback.resume()
            self.playing = True
            self._paused = False
        except Exception as e:
            logger.exception("Ошибка в resume")

    def pause(self):
        if not self._ensure_ready():
            return
        try:
            self._playback.pause()
            self.playing = False
            self._paused = True
        except Exception as e:
            logger.exception("Ошибка в pause")

    def stop(self):
        if not self._ensure_ready():
            return
        try:
            self._playback.stop()
            self.playing = False
            self._paused = False
        except Exception as e:
            logger.exception("Ошибка в stop")

    def toggle_pause(self):
        if self.playing:
            self.pause()
        elif self._paused:
            self.resume()

    def seek_absolute(self, frame_idx: int):
        if not self._ensure_ready():
            return
        try:
            if self._lazy_index is None:
                return
            window = self._lazy_index.open_window(frame_idx)
            if window is None:
                logger.error(f"Не удалось открыть окно для кадра {frame_idx}")
                return
            self._playback.seek(frame_idx, window)
        except Exception as e:
            logger.exception(f"Ошибка в seek_absolute({frame_idx})")

    def seek_relative(self, delta_sec: float):
        if not self._ensure_ready():
            return
        try:
            frame = pts_to_video_frame(self._playback.audio_clock)
            target = frame + int(round(delta_sec * self.fps))
            target = max(0, min(target, self.total_frames - 1))
            self.seek_absolute(target)
        except Exception as e:
            logger.exception(f"Ошибка в seek_relative({delta_sec})")

    def set_seek_speed(self, direction: int):
        if not self._ensure_ready():
            return
        try:
            self._playback.set_speed(direction)
        except Exception as e:
            logger.exception(f"Ошибка в set_seek_speed({direction})")

    def reset_seek_speed(self):
        if not self._ensure_ready():
            return
        try:
            self._playback.reset_speed()
        except Exception as e:
            logger.exception("Ошибка в reset_seek_speed")

    def get_seek_speed_display(self) -> str:
        if not self._ensure_ready():
            return "▶ x1"
        try:
            return self._playback.get_speed_display()
        except Exception:
            return "▶ x1"

    def get_display_frame(self) -> Optional[np.ndarray]:
        if not self._ensure_ready():
            return None
        try:
            return self._playback.get_display_frame()
        except Exception as e:
            logger.exception("Ошибка в get_display_frame")
            return None

    @property
    def audio_clock(self) -> int:
        if not self._ensure_ready():
            return 0
        try:
            return self._playback.audio_clock
        except Exception:
            return 0

    def get_local_timecode_str(self) -> str:
        if not self._ensure_ready():
            return "00:00:00;00"
        try:
            return self._playback.get_local_timecode_str()
        except Exception:
            return "00:00:00;00"

    def get_real_timecode_str(self) -> str:
        if not self._ensure_ready():
            return "00:00:00;00"
        try:
            return self._playback.get_real_timecode_str()
        except Exception:
            return "00:00:00;00"

    def set_active_tracks(self, tracks: List[int]):
        self.active_tracks = [t for t in tracks if t in (2, 3)]
        if self.audio_output:
            try:
                self.audio_output.set_active_tracks(self.active_tracks)
            except Exception as e:
                logger.exception("Ошибка в set_active_tracks")

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self._playback:
                self._playback.close()
            if self.decoder:
                self.decoder.close()
            for ad in self.audio_decoders:
                ad.close()
            if self._lazy_index:
                self._lazy_index.close()
            if self.audio_output:
                self.audio_output.stop()
        except Exception as e:
            logger.exception("Ошибка при закрытии StreamController")