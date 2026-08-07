"""
lazy_index.py – оконный доступ к индексу для ProxyPlayer v1.
Загружает только необходимую часть video_records и audio_tracks
вокруг текущей позиции, используя memory-mapped .idx файл.
Все индексы внутри IndexWindow — локальные (от 0 до N-1).
"""

import threading
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Callable
from collections import defaultdict

import numpy as np

from config.timebase import SAMPLES_PER_CHUNK, FRAMES_PER_CHUNK
from index.moov_builder import (
    DTYPE_193, DTYPE_AUDIO, SEGMENT_SIZE,
    _abs_offset, _filter_normal_records,
    get_idr_indices_from_mmap,
    build_audio_tracks,
    build_chunks_from_cached_offsets,
    build_audio_chunks_in_range,
    DEFAULT_TRACK_FILTER,
)
from index.idx_cache import open_idx_mmap

logger = logging.getLogger(__name__)


class IndexWindow:
    """
    Срез индекса, охватывающий диапазон кадров [start_frame, end_frame).
    Все индексы внутри окна — локальные (от 0 до N-1).
    """

    def __init__(
        self,
        video_records: np.ndarray,
        audio_tracks: np.ndarray,
        chunk_offsets: np.ndarray,
        chunk_sizes: np.ndarray,
        audio_chunks: List[Dict],
        start_frame: int,
        end_frame: int,
        idr_frames: np.ndarray,
        cached_offsets: np.ndarray = None,
    ):
        self.video_records = video_records          # локальные индексы 0..N-1
        self.audio_tracks = audio_tracks            # локальные
        self.chunk_offsets = chunk_offsets          # локальные
        self.chunk_sizes = chunk_sizes              # локальные
        self.audio_chunks = audio_chunks            # локальные индексы 0..M-1
        self.window_start_frame = start_frame       # глобальный номер первого кадра
        self.window_end_frame = end_frame           # глобальный номер последнего кадра + 1
        self.window_start_chunk = start_frame // FRAMES_PER_CHUNK  # глобальный номер первого чанка
        self.idr_frames = idr_frames                # локальные индексы IDR в окне
        self.cached_offsets = cached_offsets        # предвычисленные абсолютные смещения

    @property
    def total_chunks(self) -> int:
        """Количество чанков в окне (локальное)."""
        return len(self.chunk_offsets)

    def contains_frame(self, frame_idx: int) -> bool:
        """Проверяет, попадает ли глобальный кадр в окно."""
        return self.window_start_frame <= frame_idx < self.window_end_frame

    def global_to_local_chunk(self, global_chunk: int) -> int:
        """Переводит глобальный индекс чанка в локальный."""
        return global_chunk - self.window_start_chunk

    def local_to_global_frame(self, local_frame: int) -> int:
        """Переводит локальный индекс кадра в глобальный."""
        return local_frame + self.window_start_frame


class LazyIndex:
    """
    Управляет окном индекса, подгружая данные по мере необходимости.
    Использует mmap для доступа к полному .idx файлу.
    """

    def __init__(self, mirror_path: Path, mp4_path: Path, mdat_end: int):
        self.mirror_path = mirror_path
        self.mp4_path = mp4_path
        self.mdat_end = mdat_end

        # mmap полного индекса (только для чтения)
        self._all_193, self._all_c9 = open_idx_mmap(mirror_path)

        # Полные video_records (глобальные индексы)
        self._video_records_full = _filter_normal_records(self._all_193)
        self._audio_tracks_full: Optional[np.ndarray] = None  # ленивая загрузка

        self._window: Optional[IndexWindow] = None
        self._lock = threading.Lock()

        # Параметры окна
        self.default_window_seconds = 300.0  # 5 минут

    @property
    def window(self) -> Optional[IndexWindow]:
        return self._window

    def open_window(
        self, center_frame: int, window_seconds: float = None
    ) -> IndexWindow:
        """
        Открывает окно вокруг center_frame.
        Если окно уже содержит этот кадр, возвращает текущее.
        Возвращает IndexWindow с локальными индексами.
        """
        with self._lock:
            if self._window and self._window.contains_frame(center_frame):
                logger.debug(f"Кадр {center_frame} уже в окне")
                return self._window

            if window_seconds is None:
                window_seconds = self.default_window_seconds

            half_samples = int(window_seconds * 48000 / 2)
            half_frames = half_samples // 1920

            total_frames = len(self._video_records_full)
            start_frame = max(0, center_frame - half_frames)
            end_frame = min(total_frames, center_frame + half_frames)

            # Гарантируем минимальный размер окна
            min_frames = 100 * FRAMES_PER_CHUNK
            if end_frame - start_frame < min_frames:
                end_frame = min(total_frames, start_frame + min_frames)

            self._window = self._build_window(start_frame, end_frame)
            logger.info(
                f"Окно открыто: кадры {start_frame}-{end_frame} "
                f"(чанки {self._window.window_start_chunk}-"
                f"{self._window.window_start_chunk + self._window.total_chunks})"
            )
            return self._window

    def move_window_async(
        self, center_frame: int, callback: Callable[[IndexWindow], None]
    ):
        """Асинхронно перемещает окно, вызывая callback по завершении."""
        def _move():
            win = self.open_window(center_frame)
            callback(win)

        thread = threading.Thread(target=_move, daemon=True)
        thread.start()

    def expand_window(self, new_end_frame: int) -> IndexWindow:
        """Расширяет окно вперёд (при росте файла)."""
        with self._lock:
            if self._window is None:
                return self.open_window(new_end_frame - 1000)

            old_end = self._window.window_end_frame
            if new_end_frame <= old_end:
                return self._window

            # Расширяем окно
            start_frame = self._window.window_start_frame
            end_frame = min(len(self._video_records_full), new_end_frame)
            self._window = self._build_window(start_frame, end_frame)
            logger.info(f"Окно расширено до кадра {end_frame}")
            return self._window

    def close(self):
        """Освобождает ресурсы."""
        self._window = None
        self._all_193 = None
        self._all_c9 = None

    # ------------------------------------------------------------------
    def _build_window(self, start_frame: int, end_frame: int) -> IndexWindow:
        """Строит IndexWindow с локальными индексами для указанного диапазона."""
        # 1. Видео-записи в диапазоне (срез — view, не копия)
        video_slice = self._video_records_full[start_frame:end_frame]

        # 2. Кэшируем абсолютные смещения ОДИН раз
        cached_offsets = _abs_offset(video_slice) if len(video_slice) > 0 else np.array([], dtype=np.uint64)

        # 3. Аудио-записи (ленивая загрузка полных треков при первом обращении)
        if self._audio_tracks_full is None:
            self._audio_tracks_full = build_audio_tracks(
                self._all_c9, track_filter=DEFAULT_TRACK_FILTER
            )

        # Определяем PTS-границы для аудио в этом окне
        start_pts = (start_frame // FRAMES_PER_CHUNK) * SAMPLES_PER_CHUNK
        end_pts = ((end_frame + FRAMES_PER_CHUNK - 1) // FRAMES_PER_CHUNK) * SAMPLES_PER_CHUNK

        audio_mask = (
            (self._audio_tracks_full['pts'] >= start_pts)
            & (self._audio_tracks_full['pts'] < end_pts)
        )
        audio_slice = self._audio_tracks_full[audio_mask]

        # 4. Чанки для видео в окне (используем кэшированные смещения)
        chunk_offsets, chunk_sizes = build_chunks_from_cached_offsets(
            video_slice, cached_offsets, self.mdat_end
        )

        # 5. Аудио-чанки (предвычисленный индекс, без searchsorted)
        start_chunk = start_frame // FRAMES_PER_CHUNK
        audio_chunks = build_audio_chunks_in_range(
            audio_slice, start_chunk, len(chunk_offsets)
        )

        # 6. IDR в окне (локальные индексы)
        idr_in_window = get_idr_indices_from_mmap(video_slice)

        return IndexWindow(
            video_records=video_slice,
            audio_tracks=audio_slice,
            chunk_offsets=chunk_offsets,
            chunk_sizes=chunk_sizes,
            audio_chunks=audio_chunks,
            start_frame=start_frame,
            end_frame=end_frame,
            idr_frames=idr_in_window,
            cached_offsets=cached_offsets,
        )