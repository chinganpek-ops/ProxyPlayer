#!/usr/bin/env python3
"""
player_controller.py – контроллер воспроизведения DaletProxy V6.
Синхронизация: видео ведомое, аудио ведущее (audio_clock).
audio_delay_ms применяется как постоянное смещение audio_clock в get_display_frame.
Гарантированно отображаются все кадры без пропусков.
Кадры не выбрасываются, а показываются с максимально возможной скоростью,
если отставание накопилось более 400 мс.
Таймеры удаляются безопасно через deleteLater() из любого потока.
Добавлено принудительное завершение всех потоков при закрытии плеера.
"""

import time
import threading
import socket
from pathlib import Path
from typing import Optional, Dict, List
import logging

import numpy as np
import av
from PyQt5.QtCore import QObject, pyqtSignal, QTimer

from timebase import (
    SAMPLES_PER_VIDEO_FRAME, video_frame_to_pts, pts_to_video_frame,
    AUDIO_SAMPLE_RATE
)
from frame_buffer import FrameRingBuffer
from audio_buffer import MultiTrackAudioBuffer
from audio_output import AudioOutput
from decoder import Decoder
from tcd_parser import parse_tcd
from moov_builder import (
    DTYPE_193, SEGMENT_SIZE,
    fast_video_records,
    build_audio_tracks, build_audio_chunks_v2,
    build_chunks_in_range, build_audio_chunks_in_range,  # новые функции
    DEFAULT_TRACK_FILTER,
    build_idr_map,
    get_idr_indices_from_mmap,
    incremental_append_video,
    incremental_get_chunks,
    incremental_build_audio_chunks_v2,
)
from idx_cache import (
    prepare_mirror, open_idx_mmap, remap_idx,
    cleanup_cache,
)
from win_sequential_reader import WinSequentialReader
from chunk_reader import ChunkReader
from hybrid_seek import HybridSeekWorker
from utils import get_real_size, _WorkerThread

logger = logging.getLogger(__name__)

DEFAULT_AVCC = bytes.fromhex(
    "014d001fffe1002e674d401f9652816824dff80200016a50101014000003"
    "000400000300cb8180009600000301e848fc6383b428532c01000568e9093520"
)
DEFAULT_ASC = b'\x11\x88'

LIVE_SEEK_OFFSET_FRAMES = 1600
MAX_VIDEO_LAG = SAMPLES_PER_VIDEO_FRAME * 10          # 19200 сэмплов (400 мс)
DROPPABLE_LAG = 4800                                   # 100 мс – можно пропустить
FUTURE_HORIZON = SAMPLES_PER_VIDEO_FRAME // 2          # 960 – кадр «не в будущем»
SEEK_SPEEDS = [2.0, 4.0, 8.0]                          # множители для JKL

INITIAL_CHUNKS = 1200  # количество чанков для быстрого старта


class PlayerController(QObject):
    ready = pyqtSignal()
    error = pyqtSignal(str)
    index_updated = pyqtSignal()
    seek_finished = pyqtSignal(int)
    audio_clock_changed = pyqtSignal(int)
    placeholder_hide = pyqtSignal()
    recording_finished = pyqtSignal()

    def __init__(self, ref_path: Path, idx_path: Path, mp4_path: Path,
                 fps: float = 25.0, buffer_size: int = 600,
                 free_slots_required: int = 25,
                 group_chunks: int = 30,
                 max_retries: int = 3,
                 normal_playback_speed_factor: float = 1.25,
                 rate_limit: int = 0,
                 thread_type: str = "AUTO",
                 thread_count: int = 0,
                 skip_frame: bool = False,
                 gpu_mode: str = "off",
                 use_mmap_reader: bool = False,
                 avcc_override: Optional[bytes] = None,
                 audio_delay_ms: int = 0):
        super().__init__()

        self.ref_path = ref_path
        self.idx_path = idx_path
        self.mp4_path = mp4_path
        self.fps = fps
        self.buffer_size = buffer_size
        self.free_slots_required = free_slots_required
        self.group_chunks = group_chunks
        self.max_retries = max_retries
        self.normal_playback_speed_factor = normal_playback_speed_factor
        self.rate_limit = rate_limit
        self.thread_type = thread_type
        self.thread_count = thread_count
        self.skip_frame = skip_frame
        self.gpu_mode = gpu_mode
        self.use_mmap_reader = use_mmap_reader
        self._avcc_override = avcc_override

        self.audio_delay_samples = int(audio_delay_ms * AUDIO_SAMPLE_RATE / 1000)

        self._debug_enabled = True
        if self._debug_enabled:
            self._debug_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._debug_addr = ('127.0.0.1', 18081)
        else:
            self._debug_sock = None

        self._ref_path = ref_path
        self._wrec_path = None
        if ref_path and ref_path.exists():
            wrec_candidate = mp4_path.parent / (mp4_path.name + 'wrec')
            if wrec_candidate.exists():
                self._wrec_path = wrec_candidate
        self._finalized = not ((self._ref_path and self._ref_path.exists()) and 
                               (self._wrec_path and self._wrec_path.exists()))

        self.buffer_main = FrameRingBuffer(max_frames=buffer_size)
        self.buffer_background = FrameRingBuffer(max_frames=buffer_size)
        self.buffer_standby = FrameRingBuffer(max_frames=buffer_size)
        self.buffer = self.buffer_main

        self.audio_buffers = MultiTrackAudioBuffer(capacity_samples=1440000)
        if self._debug_enabled:
            self.audio_buffers.set_debug_callback(self._send_debug)

        self.audio_output: Optional[AudioOutput] = None
        self.active_tracks = [2, 3]

        self.audio_decoders: List[Optional] = [None] * 16
        self.asc = b'\x11\x88'
        self.audio_chunks: List[Dict[int, List[dict]]] = []

        self.chunk_reader: Optional[ChunkReader] = None

        self.playing = False
        self._paused = False
        self._audio_clock = 0
        self._audio_clock_lock = threading.Lock()

        self._cached_first_frame: Optional[np.ndarray] = None

        self._init_thread: Optional[_WorkerThread] = None
        self._build_thread: Optional[threading.Thread] = None  # для фонового построения индекса

        self.decoder: Optional[Decoder] = None
        self.avcc: Optional[bytes] = None
        self.start_frame_offset = 0
        self.total_frames = 0
        self.video_records = np.empty(0, dtype=DTYPE_193)
        self.chunk_offsets = np.array([], dtype=np.int64)
        self.chunk_sizes = np.array([], dtype=np.int64)
        self.mdat_end = 0

        self.all_193 = np.empty(0, dtype=DTYPE_193)
        self._idx_last_size = 0

        self._seek_lock = threading.Lock()
        self._last_network_speed = 0.0

        self._cache_check_timer = QTimer(self)
        self._cache_check_timer.timeout.connect(self._check_mmap_growth)

        self._current_frame_idx = 0
        self._ready = threading.Event()
        self._init_error: Optional[str] = None

        self.idr_frames = np.array([], dtype=np.int64)

        self._seek_reader: Optional[WinSequentialReader] = None
        self._mirror_path: Optional[Path] = None

        self._stats_timer = QTimer(self)
        self._stats_timer.timeout.connect(self._send_buffer_stats)
        self._stats_timer.start(2000)

        self._drift_correction = {
            'enabled': True,
            'nominal_rate': AUDIO_SAMPLE_RATE,
            'measure_interval': 2.0,
            'last_sys_time': None,
            'last_audio_clock': None,
            'measured_rate': None,
            'start_sys_time': None,
            'start_audio_clock': None,
        }
        self._last_drift_update = 0.0

        # JKL-перемотка
        self._seek_speed = 1.0
        self._seek_speed_index = -1
        self._seek_direction = 0
        self._last_seek_time = 0.0
        self._seek_accumulator = 0.0
        self._normal_playing_state = False

        # Live-старт
        self._start_from_live = True

        if not self.mp4_path or not self.mp4_path.exists():
            self._init_error = f"MP4 файл не существует: {self.mp4_path}"
            QTimer.singleShot(0, self._on_init_finished)
            return
        if self.idx_path and not self.idx_path.exists():
            self._init_error = f"IDX файл не существует: {self.idx_path}"
            QTimer.singleShot(0, self._on_init_finished)
            return

        logger.info(f"Инициализация PlayerController V6 для {mp4_path.name}")
        self._init_thread = _WorkerThread(target=self._background_init, name="InitThread")
        self._init_thread.finished.connect(self._on_init_finished)
        self._init_thread.start()

    # ------------------------------------------------------------------
    def _send_debug(self, event_type, details=""):
        if not self._debug_enabled or not self._debug_sock:
            return
        try:
            msg = f"{event_type}|{details}".encode('utf-8')
            self._debug_sock.sendto(msg, self._debug_addr)
        except Exception:
            pass

    def _send_buffer_stats(self):
        if not self._debug_enabled or not self._debug_sock:
            return
        vb = self.buffer_main.count
        r2 = int(self.audio_buffers.buffers[2].read_pos)
        r3 = int(self.audio_buffers.buffers[3].read_pos)
        first = self.buffer_main.peek_first()
        vpts = first[0] if first else -1
        aclock = self._audio_clock
        self._send_debug("BUFFER_STATS",
                         f"video={vb} audio2={int(self.audio_buffers.buffers[2].available_read)} "
                         f"audio3={int(self.audio_buffers.buffers[3].available_read)} "
                         f"read2={r2} read3={r3} vpts={vpts} aclock={aclock}")

    def _background_init(self):
        try:
            self._mirror_path = prepare_mirror(self.idx_path)
            self.all_193, mm_c9 = open_idx_mmap(self._mirror_path)
            self._idx_last_size = self._mirror_path.stat().st_size

            if len(self.all_193) == 0:
                self._init_error = "Индекс пуст"
                return

            self.mdat_end = get_real_size(str(self.mp4_path))

            self.avcc = self._avcc_override if self._avcc_override else DEFAULT_AVCC
            self.asc = DEFAULT_ASC

            self.video_records = fast_video_records(self.all_193)
            if len(self.video_records) == 0:
                self._init_error = "Не удалось найти видеозаписи"
                return

            self.total_frames = self._calculate_total_frames()

            # Определяем стартовый чанк (live или начало)
            if self._start_from_live and not self._finalized:
                start_frame_idx = max(0, self.total_frames - LIVE_SEEK_OFFSET_FRAMES)
            else:
                start_frame_idx = 0
            start_chunk = start_frame_idx // 12

            # Быстрое построение первого окна чанков (1200 от стартовой позиции)
            self.chunk_offsets, self.chunk_sizes = build_chunks_in_range(
                self.video_records, self.mdat_end, start_chunk, INITIAL_CHUNKS
            )
            audio_tracks = build_audio_tracks(mm_c9, track_filter=DEFAULT_TRACK_FILTER)
            self.audio_chunks = build_audio_chunks_in_range(
                audio_tracks, start_chunk, len(self.chunk_offsets)
            )

            # Инициализация декодера и читателя
            tcd_path = self.mp4_path.with_suffix('.tcd')
            if not tcd_path.exists():
                tcd_path = self.idx_path.parent / f"{self.mp4_path.stem}.tcd"
            start_h, start_m, start_s, start_f = parse_tcd(tcd_path)
            self.start_frame_offset = int((start_h * 3600 + start_m * 60 + start_s) * self.fps + start_f)

            self.decoder = Decoder(self.avcc, self.mp4_path,
                                   thread_type=self.thread_type,
                                   thread_count=self.thread_count,
                                   skip_frame=self.skip_frame,
                                   gpu_mode=self.gpu_mode)
            try:
                init_packet = av.Packet(self.avcc)
                self.decoder.codec.decode(init_packet)
            except Exception as e:
                logger.warning(f"Ошибка инициализации декодера: {e}")

            self._create_chunk_reader()
            self._seek_reader = WinSequentialReader(self.mp4_path, rate_limit=0, overlapped=False)
            self._build_idr_list()
            self._cache_first_frame()

            # Запускаем фоновое построение оставшихся чанков
            self._start_background_index_build(mm_c9, start_chunk)

            self._send_debug("INIT_DONE", f"frames={self.total_frames}")

        except Exception as e:
            self._init_error = str(e)
            logger.exception("Ошибка инициализации")

    def _start_background_index_build(self, mm_c9, start_chunk):
        """Запускает фоновый поток для построения полных чанков и аудиоданных."""
        def build_full():
            try:
                # Полные чанки и аудио
                full_offsets, full_sizes = incremental_get_chunks(self.video_records, self.mdat_end)
                audio_tracks = build_audio_tracks(mm_c9, track_filter=DEFAULT_TRACK_FILTER)
                full_audio = build_audio_chunks_v2(audio_tracks, len(full_offsets))

                # Обновляем данные в chunk_reader
                if self.chunk_reader:
                    self.chunk_reader.update_data(
                        self.video_records, full_offsets, full_sizes, full_audio,
                        mdat_end=self.mdat_end
                    )
                self._send_debug("BACKGROUND_INDEX_READY", str(len(full_offsets)))
            except Exception as e:
                logger.error(f"Ошибка фонового построения индекса: {e}")

        self._build_thread = threading.Thread(target=build_full, daemon=True)
        self._build_thread.start()

    def _calculate_total_frames(self) -> int:
        normal_mask = self.all_193['f2'] < 256
        if np.any(normal_mask):
            max_f3 = int(np.max(self.all_193['f3'][normal_mask]))
            if max_f3 > 0:
                return max_f3 + 1
        if self._seek_reader:
            max_pts = int(np.max(self.all_193['f3'])) if len(self.all_193) > 0 else 0
            if max_pts > 0:
                return max_pts // SAMPLES_PER_VIDEO_FRAME + 1
        return len(self.video_records)

    def _build_idr_list(self):
        if len(self.video_records) == 0:
            self.idr_frames = np.array([], dtype=np.int64)
            return
        from moov_builder import get_idr_indices_from_mmap
        self.idr_frames = get_idr_indices_from_mmap(self.video_records)

    def _create_chunk_reader(self):
        if self.chunk_reader:
            self.chunk_reader.stop()
            self.chunk_reader = None
        self.chunk_reader = ChunkReader(
            mp4_path=self.mp4_path,
            video_records=self.video_records,
            chunk_offsets=self.chunk_offsets,
            chunk_sizes=self.chunk_sizes,
            audio_chunks=self.audio_chunks,
            video_buffer=self.buffer_main,
            audio_buffers=self.audio_buffers,
            video_decoder=self.decoder,
            audio_decoders=self.audio_decoders,
            asc=self.asc,
            fps=self.fps,
            group_chunks=self.group_chunks,
            max_retries=self.max_retries,
            free_slots_required=self.free_slots_required,
            normal_playback_speed_factor=self.normal_playback_speed_factor,
            rate_limit=self.rate_limit,
            bitrate=0,
            mdat_end=self.mdat_end,
            mm_c9=None,
            dynamic_chunks=False,
            mm_193=None,
            use_mmap_reader=self.use_mmap_reader,
            debug_callback=self._send_debug,
        )

    def _start_mmap_monitoring(self):
        self._cache_check_timer.start(10000)

    def _check_mmap_growth(self):
        if not self._mirror_path or not self.idx_path:
            return
        threading.Thread(target=self._check_mmap_growth_bg, daemon=True).start()

    def _check_mmap_growth_bg(self):
        if self._finalized:
            return
        try:
            self._mirror_path = prepare_mirror(self.idx_path)
            new_size = self._mirror_path.stat().st_size
            if new_size > self._idx_last_size:
                self._update_index_from_mmap(new_size)
                self._send_debug("INDEX_GROWTH", f"{self._idx_last_size} -> {new_size}")
        except Exception as e:
            logger.error(f"Ошибка проверки роста зеркала: {e}")

    def _on_recording_finished(self):
        if not self.playing and not self._paused:
            return
        logger.info("Запись остановлена")
        self._send_debug("RECORDING_FINISHED")
        self.stop()
        self.recording_finished.emit()

    def _update_index_from_mmap(self, new_size: int):
        try:
            self.all_193, mm_c9 = remap_idx(self._mirror_path)
            self._idx_last_size = new_size
            if len(self.all_193) == 0:
                return
            self.video_records = incremental_append_video(self.video_records, self.all_193)
            self.total_frames = self._calculate_total_frames()
            self.mdat_end = get_real_size(str(self.mp4_path))
            self.chunk_offsets, self.chunk_sizes = incremental_get_chunks(
                self.video_records, self.mdat_end
            )
            audio_tracks = build_audio_tracks(mm_c9, track_filter=DEFAULT_TRACK_FILTER)
            self.audio_chunks = incremental_build_audio_chunks_v2(
                self.audio_chunks, audio_tracks, len(self.chunk_offsets)
            )
            self._build_idr_list()
            if self.chunk_reader:
                self.chunk_reader.update_data(
                    self.video_records, self.chunk_offsets,
                    self.chunk_sizes, self.audio_chunks, mdat_end=self.mdat_end
                )
            self.index_updated.emit()
            self._send_debug("INDEX_UPDATED", str(self.total_frames))
        except Exception as e:
            logger.error(f"Ошибка инкрементального обновления: {e}")

    def _on_init_finished(self):
        if self._init_error:
            self.error.emit(self._init_error)
        else:
            self._ready.set()
            self._start_mmap_monitoring()
            self.ready.emit()
            self._start_priority_buffer_fill()

    def _start_priority_buffer_fill(self):
        if not self.chunk_reader:
            return

        # Определяем стартовый кадр (live или начало)
        if self._start_from_live and not self._finalized:
            start_frame_idx = max(0, self.total_frames - LIVE_SEEK_OFFSET_FRAMES)
            if len(self.idr_frames) > 0:
                idr_positions = self.idr_frames[self.idr_frames <= start_frame_idx]
                if len(idr_positions) > 0:
                    start_frame_idx = idr_positions[-1]
        else:
            start_frame_idx = self.idr_frames[0] if len(self.idr_frames) > 0 else 0

        start_chunk = start_frame_idx // 12

        # --- Мгновенный захват первого кадра через HybridSeekWorker ---
        if not self._cached_first_frame:
            try:
                worker = HybridSeekWorker(
                    mp4_path=self.mp4_path,
                    video_records=self.video_records,
                    mdat_end=self.mdat_end,
                    decoder=self.decoder,
                    buffer=self.buffer_standby,
                    frame_idx=start_frame_idx,
                )
                worker.run()
                if self.buffer_standby.count > 0:
                    first = self.buffer_standby.peek_first()
                    if first is not None:
                        self.buffer_main.update_keep_last(first[1], first[0])
                        self._send_debug("FIRST_FRAME_GRABBED", f"pts={first[0]}")
                    self.buffer_standby.clear()
            except Exception as e:
                logger.warning(f"Не удалось быстро захватить первый кадр: {e}")

        # Запускаем фоновую загрузку чанков
        self.chunk_reader.start(start_chunk=start_chunk)
        self.chunk_reader.enable_audio_write()
        self.chunk_reader.resume_loading()

        threading.Thread(target=self._wait_for_buffer_ready, daemon=True).start()

    def _wait_for_buffer_ready(self):
        while not self._ready.is_set():
            time.sleep(0.1)
        timeout = 30
        waited = 0
        while waited < timeout:
            if self.buffer_main.count >= self.free_slots_required:
                self.placeholder_hide.emit()
                first = self.buffer_main.peek_first()
                if first is not None:
                    self.buffer_main.update_keep_last(first[1], first[0])
                self._send_debug("BUFFER_READY")
                return
            time.sleep(0.2)
            waited += 0.2
        self.placeholder_hide.emit()
        first = self.buffer_main.peek_first()
        if first is not None:
            self.buffer_main.update_keep_last(first[1], first[0])
        self._send_debug("BUFFER_TIMEOUT")

    def _cache_first_frame(self):
        if len(self.video_records) == 0 or len(self.idr_frames) == 0:
            return
        first_idr = self.video_records[self.idr_frames[0]]
        try:
            abs_offset = int(first_idr['f1']) - 4 + int(first_idr['f2']) * SEGMENT_SIZE
            size = 5 * 1024 * 1024
            data = self._seek_reader.read_sequential(abs_offset, min(size, self.mdat_end - abs_offset))
            if not data:
                return
            filtered = self.decoder.filter_avcc(data)
            if not filtered:
                return
            frames = self.decoder.decode_sample(filtered)
            if frames:
                self._cached_first_frame = frames[0]
        except Exception as e:
            logger.warning(f"Не удалось кэшировать первый кадр: {e}")

    # ------------------------------------------------------------------
    def set_active_tracks(self, tracks: List[int]):
        self.active_tracks = [t for t in tracks if t in (2, 3)]
        if self.audio_output:
            self.audio_output.set_active_tracks(self.active_tracks)

    def start_playback(self):
        if not self._ready.is_set():
            return
        logger.info("Загрузка и пауза на первом кадре")
        self._send_debug("PLAY_START")

        # Определяем стартовый кадр (live или начало)
        if self._start_from_live and not self._finalized:
            start_frame_idx = max(0, self.total_frames - LIVE_SEEK_OFFSET_FRAMES)
            if len(self.idr_frames) > 0:
                idr_positions = self.idr_frames[self.idr_frames <= start_frame_idx]
                if len(idr_positions) > 0:
                    start_frame_idx = idr_positions[-1]
        else:
            start_frame_idx = self.idr_frames[0] if len(self.idr_frames) > 0 else 0
        start_chunk = start_frame_idx // 12
        self._current_frame_idx = start_frame_idx
        video_pts = video_frame_to_pts(start_frame_idx)

        self.buffer_main.clear()
        self.buffer = self.buffer_main
        self.audio_buffers.clear_all()

        if len(self.audio_chunks) > 0 and self.audio_chunks[0]:
            first_audio_pts = None
            for track_id in (2, 3):
                entries = self.audio_chunks[0].get(track_id)
                if entries:
                    pts = entries[0]['pts']
                    if first_audio_pts is None or pts < first_audio_pts:
                        first_audio_pts = pts
            if first_audio_pts is not None and video_pts >= first_audio_pts:
                self.audio_buffers.reset_all_read_to(video_pts)

        with self._audio_clock_lock:
            self._audio_clock = video_pts

        if not self.audio_output:
            self.audio_output = AudioOutput(self.audio_buffers,
                                            debug_callback=self._send_debug if self._debug_enabled else None)
            self.audio_output.audio_clock_changed.connect(self._on_audio_clock_changed)
        self.audio_output.set_active_tracks(self.active_tracks)
        self.audio_output.reset_clock(video_pts)

        if self.chunk_reader:
            self.chunk_reader.video_buffer = self.buffer_main
        if self.chunk_reader and not self.chunk_reader._worker_thread:
            self.chunk_reader.start(start_chunk=start_chunk)
            self.chunk_reader.enable_audio_write()
            self.chunk_reader.resume_loading()

        self.playing = False
        self._paused = True
        self._send_debug("READY_PAUSED")

    def resume(self):
        if not self._paused:
            return
        logger.info("Возобновление")
        self.playing = True
        self._paused = False
        if self.audio_output:
            self.audio_output.reset_clock(self._audio_clock)
        if self.chunk_reader:
            self.chunk_reader.enable_audio_write()
        now = time.monotonic()
        dc = self._drift_correction
        dc['start_sys_time'] = now
        dc['start_audio_clock'] = self._audio_clock
        dc['last_sys_time'] = now
        dc['last_audio_clock'] = self._audio_clock
        self._start_audio_when_video_ready()

    def _start_audio_when_video_ready(self):
        self._stop_video_ready_timer()
        first = self.buffer_main.peek_first()
        if first is not None and first[0] <= self._audio_clock + SAMPLES_PER_VIDEO_FRAME // 2:
            self._start_audio()
            return
        self._video_ready_timer = QTimer(self)
        self._video_ready_timer.timeout.connect(self._check_video_ready)
        self._video_ready_timer.start(10)

    def _check_video_ready(self):
        if not self.playing:
            self._stop_video_ready_timer()
            return
        first = self.buffer_main.peek_first()
        if first is not None and first[0] <= self._audio_clock + SAMPLES_PER_VIDEO_FRAME // 2:
            self._stop_video_ready_timer()
            self._start_audio()
        if not hasattr(self, '_video_ready_start'):
            self._video_ready_start = time.monotonic()
        elif time.monotonic() - self._video_ready_start > 5.0:
            self._stop_video_ready_timer()
            logger.warning("Таймаут ожидания видео-кадра, принудительный запуск AudioOutput")
            self._start_audio()

    def _stop_video_ready_timer(self):
        if hasattr(self, '_video_ready_timer') and self._video_ready_timer is not None:
            timer = self._video_ready_timer
            self._video_ready_timer = None
            timer.stop()
            timer.deleteLater()

    def _start_audio(self):
        if self.audio_output and not self.audio_output._active:
            self.audio_output.start()
            self._send_debug("AUDIO_START")

    # ------------------------------------------------------------------
    # JKL-перемотка
    # ------------------------------------------------------------------
    def set_seek_speed(self, direction: int):
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
            if self.audio_output:
                self.audio_output.set_volume(0.0)
        self._seek_accumulator = 0.0
        self._last_seek_time = time.monotonic()
        self._send_debug("SEEK_SPEED", f"dir={direction} speed={self._seek_speed}x")

    def reset_seek_speed(self):
        if self._seek_speed == 1.0 and self._seek_direction == 0:
            return
        self._seek_speed = 1.0
        self._seek_speed_index = -1
        self._seek_direction = 0
        self._seek_accumulator = 0.0
        if self.audio_output:
            self.audio_output.set_volume(0.8)
        if self._normal_playing_state and not self.playing:
            self.resume()
        elif not self._normal_playing_state and self.playing:
            self.pause()
        self._send_debug("SEEK_SPEED_RESET", "normal speed")

    def get_seek_speed_display(self) -> str:
        if self._seek_direction == 0:
            return "▶ x1"
        direction = "<<" if self._seek_direction < 0 else ">>"
        return f"{direction} x{self._seek_speed:.0f}"

    def _fast_seek_to_frame(self, frame_idx: int):
        if frame_idx < 0 or frame_idx >= self.total_frames:
            return
        self._current_frame_idx = frame_idx
        new_clock = video_frame_to_pts(frame_idx)
        with self._audio_clock_lock:
            self._audio_clock = new_clock
        self.buffer.drop_until(new_clock)

        first = self.buffer.peek_first()
        if first is not None:
            self.buffer.update_keep_last(first[1], first[0])

        if self.chunk_reader:
            self.chunk_reader.soft_reset(frame_idx // 12)
        if self.audio_output:
            self.audio_output.reset_clock(new_clock)
        self._send_debug("FAST_SEEK", f"frame={frame_idx}")

    # ------------------------------------------------------------------
    # Коррекция дрейфа audio_clock
    # ------------------------------------------------------------------
    def _update_drift_correction(self):
        if not self._drift_correction['enabled']:
            return
        now = time.monotonic()
        dc = self._drift_correction
        audio_clk = self._audio_clock
        if dc['start_sys_time'] is None:
            dc['start_sys_time'] = now
            dc['start_audio_clock'] = audio_clk
            dc['last_sys_time'] = now
            dc['last_audio_clock'] = audio_clk
            return
        if now - dc['last_sys_time'] >= dc['measure_interval']:
            dt = now - dc['last_sys_time']
            da = audio_clk - dc['last_audio_clock']
            if dt > 0.1 and da > 0:
                measured_rate = da / dt
                if dc['measured_rate'] is None:
                    dc['measured_rate'] = measured_rate
                else:
                    alpha = 0.3
                    dc['measured_rate'] = (1 - alpha) * dc['measured_rate'] + alpha * measured_rate
                dc['start_sys_time'] = now
                dc['start_audio_clock'] = audio_clk
            dc['last_sys_time'] = now
            dc['last_audio_clock'] = audio_clk

    def _get_adjusted_audio_clock(self) -> float:
        dc = self._drift_correction
        if not dc['enabled'] or dc['measured_rate'] is None:
            return float(self._audio_clock)
        if abs(dc['measured_rate'] - dc['nominal_rate']) / dc['nominal_rate'] < 0.0001:
            return float(self._audio_clock)
        now = time.monotonic()
        adjusted = dc['start_audio_clock'] + (now - dc['start_sys_time']) * dc['nominal_rate']
        max_deviation = int(0.5 * AUDIO_SAMPLE_RATE)
        if abs(adjusted - self._audio_clock) > max_deviation:
            logger.warning("Слишком большая коррекция дрейфа, сброс")
            dc['measured_rate'] = None
            return float(self._audio_clock)
        return adjusted

    def pause(self):
        if not self.playing:
            return
        self.playing = False
        self._paused = True
        if self.audio_output:
            self.audio_output.stop()
        self._stop_video_ready_timer()
        if self.chunk_reader:
            self.chunk_reader.disable_audio_write()
        logger.info("Пауза")
        self._send_debug("PAUSE")

    def stop(self):
        self.playing = False
        self._paused = False
        if self.chunk_reader:
            self.chunk_reader.stop()
        if self.audio_output:
            self.audio_output.stop()
        self._stop_video_ready_timer()
        logger.info("Остановка")
        self._send_debug("STOP")

    def toggle_pause(self):
        if self.playing:
            self.pause()
        elif self._paused:
            self.resume()

    def seek_to_live_position(self, offset_frames: int = LIVE_SEEK_OFFSET_FRAMES):
        target = max(0, self.total_frames - offset_frames)
        self.seek_absolute(target)

    def seek_absolute(self, frame_idx: int):
        if self.total_frames == 0:
            return
        if not self._finalized:
            max_allowed = max(0, self.total_frames - LIVE_SEEK_OFFSET_FRAMES)
            frame_idx = max(0, min(frame_idx, max_allowed))
        else:
            frame_idx = max(0, min(frame_idx, self.total_frames - 1))

        logger.info(f"Перемотка к кадру {frame_idx}")
        self._send_debug("SEEK_START", str(frame_idx))
        self._stop_video_ready_timer()

        with self._seek_lock:
            if self.chunk_reader:
                self.chunk_reader.stop()
                self.chunk_reader.disable_audio_write()
                self.chunk_reader.pause_loading()
                self.chunk_reader.flush_audio_queue()
            if self.audio_output:
                self.audio_output.stop()
            self.audio_buffers.clear_all()

            worker = HybridSeekWorker(
                mp4_path=self.mp4_path,
                video_records=self.video_records,
                mdat_end=self.mdat_end,
                decoder=self.decoder,
                buffer=self.buffer_background,
                frame_idx=frame_idx,
            )
            worker.run()

            if self.buffer_background.count == 0:
                self._send_debug("SEEK_FAILED", str(frame_idx))
                return

            self.buffer_main, self.buffer_background = self.buffer_background, self.buffer_main
            self.buffer = self.buffer_main
            self.buffer_background.clear()

            first = self.buffer.peek_first()
            if first is not None:
                self.buffer.update_keep_last(first[1], first[0])

            new_clock = video_frame_to_pts(frame_idx)
            if self.audio_output:
                self.audio_output.reset_clock(new_clock)
            with self._audio_clock_lock:
                self._audio_clock = new_clock
            self._current_frame_idx = frame_idx

            if self.chunk_reader:
                self.chunk_reader.video_buffer = self.buffer_main
                self.chunk_reader.start(start_chunk=frame_idx // 12)
                self.chunk_reader.enable_audio_write()
                self.chunk_reader.resume_loading()

            self._drift_correction['measured_rate'] = None
            self._drift_correction['start_sys_time'] = None

            self.playing = False
            self._paused = True

            self.seek_finished.emit(frame_idx)
            self._send_debug("SEEK_DONE", str(frame_idx))

    def seek_relative(self, delta_sec: float):
        if self.total_frames == 0:
            return
        delta_frames = int(round(delta_sec * self.fps))
        if delta_frames == 0:
            return
        target = self._current_frame_idx + delta_frames
        target = max(0, min(target, self.total_frames - 1))
        self.seek_absolute(target)

    # ------------------------------------------------------------------
    def get_display_frame(self) -> Optional[np.ndarray]:
        if self._cached_first_frame is not None:
            frame = self._cached_first_frame
            self._cached_first_frame = None
            self.buffer.update_keep_last(frame, 0)
            return frame

        if self._seek_direction != 0 and self._seek_speed > 1.0:
            now = time.monotonic()
            dt = now - self._last_seek_time
            self._last_seek_time = now
            frames_to_skip = self._seek_speed * self.fps * dt
            self._seek_accumulator += frames_to_skip
            if self._seek_accumulator >= 1.0:
                skip_frames = int(self._seek_accumulator)
                self._seek_accumulator -= skip_frames
                target_frame = self._current_frame_idx + (skip_frames * self._seek_direction)
                target_frame = max(0, min(target_frame, self.total_frames - 1))
                self._fast_seek_to_frame(target_frame)
            return self.buffer.get_keep_last()

        if not self.playing:
            return self.buffer.get_keep_last()

        self._update_drift_correction()
        adjusted_clock = self._get_adjusted_audio_clock()
        current_clock = adjusted_clock - self.audio_delay_samples

        self.buffer.drop_until(int(current_clock) - MAX_VIDEO_LAG)

        first = self.buffer.peek_first()
        if first is not None:
            pts, frame = first
            delta = pts - current_clock
            if delta < -MAX_VIDEO_LAG:
                self.buffer.advance()
                return self.get_display_frame()
            else:
                self.buffer.update_keep_last(frame, pts)
                self._send_debug("FRAME_DISPLAYED", f"pts={pts} aclock={self._audio_clock}")
                self.buffer.advance()
                return frame
        else:
            if self.chunk_reader and self.chunk_reader._worker_thread is None:
                self.chunk_reader.start(start_chunk=pts_to_video_frame(int(current_clock)) // 12)

            if self.total_frames > 0 and self.playing:
                if not self._finalized:
                    ref_exists = self._ref_path and self._ref_path.exists()
                    wrec_exists = self._wrec_path and self._wrec_path.exists()
                    if not ref_exists and not wrec_exists:
                        self._finalized = True
                        self._send_debug("FINALIZED")

                if self._finalized:
                    current_frame = pts_to_video_frame(int(current_clock))
                    if current_frame >= self.total_frames - 1:
                        video_empty = self.buffer_main.count == 0
                        audio_empty = True
                        if self.audio_buffers:
                            for track_id in self.active_tracks:
                                if track_id in self.audio_buffers.buffers:
                                    if self.audio_buffers.buffers[track_id].available_read > 0:
                                        audio_empty = False
                                        break
                        if video_empty and audio_empty:
                            self._send_debug("TRANS_COMPLETED", f"frame={current_frame}")
                            self._on_recording_finished()
            else:
                if self.playing and self.buffer_main.count == 0:
                    self._send_debug("BUFFER_EMPTY")
            return self.buffer.get_keep_last()

    def _on_audio_clock_changed(self, samples: int):
        with self._audio_clock_lock:
            self._audio_clock = samples
        self.audio_clock_changed.emit(samples)

    @property
    def audio_clock(self) -> int:
        with self._audio_clock_lock:
            return self._audio_clock

    def get_local_timecode_str(self) -> str:
        idx = pts_to_video_frame(self._audio_clock)
        total_seconds = idx / self.fps
        h = int(total_seconds // 3600)
        m = int((total_seconds % 3600) // 60)
        s = int(total_seconds % 60)
        f = int(round((total_seconds - int(total_seconds)) * self.fps))
        return f"{h:02d}:{m:02d}:{s:02d};{f:02d}"

    def get_real_timecode_str(self) -> str:
        idx = pts_to_video_frame(self._audio_clock)
        abs_frame = self.start_frame_offset + idx
        total_seconds = abs_frame / self.fps
        h = int(total_seconds // 3600)
        m = int((total_seconds % 3600) // 60)
        s = int(total_seconds % 60)
        f = int(round((total_seconds - int(total_seconds)) * self.fps))
        return f"{h:02d}:{m:02d}:{s:02d};{f:02d}"

    def get_display_timecode(self) -> str:
        return self.get_real_timecode_str()

    @property
    def network_speed(self) -> float:
        if self.chunk_reader and self.chunk_reader.network_speed > 0:
            self._last_network_speed = self.chunk_reader.network_speed
        return self._last_network_speed

    def close(self):
        self.playing = False
        self._paused = False
        
        if self._cache_check_timer:
            self._cache_check_timer.stop()
            self._cache_check_timer.deleteLater()
            self._cache_check_timer = None
        if self._stats_timer:
            self._stats_timer.stop()
            self._stats_timer.deleteLater()
            self._stats_timer = None
        self._stop_video_ready_timer()
        
        if self.chunk_reader:
            self.chunk_reader.stop()
        if self.audio_output:
            self.audio_output.stop()
        
        if self.decoder:
            self.decoder.close()
        if self._seek_reader:
            self._seek_reader.close()
        
        if self._debug_enabled and self._debug_sock:
            self._debug_sock.close()
            self._debug_sock = None
        
        try:
            if self.idx_path:
                cleanup_cache(self.idx_path)
        except Exception as e:
            logger.warning(f"Ошибка очистки кэша: {e}")