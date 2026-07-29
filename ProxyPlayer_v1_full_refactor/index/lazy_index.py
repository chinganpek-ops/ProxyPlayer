"""
lazy_index.py – оконный доступ к индексу для ProxyPlayer v1.
Загружает только необходимую часть video_records и audio_tracks
вокруг текущей позиции, используя memory-mapped .idx файл.
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
    DEFAULT_TRACK_FILTER,
)
from index.idx_cache import open_idx_mmap

logger = logging.getLogger(__name__)


class IndexWindow:
    """
    Срез индекса, охватывающий диапазон кадров [start_frame, end_frame).
    Содержит готовые структуры для ChunkPipeline.
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
    ):
        self.video_records = video_records
        self.audio_tracks = audio_tracks
        self.chunk_offsets = chunk_offsets
        self.chunk_sizes = chunk_sizes
        self.audio_chunks = audio_chunks
        self.window_start_frame = start_frame
        self.window_end_frame = end_frame
        self.idr_frames = idr_frames

    def contains_frame(self, frame_idx: int) -> bool:
        """Проверяет, попадает ли кадр в окно."""
        return self.window_start_frame <= frame_idx < self.window_end_frame

    def chunk_range(self) -> Tuple[int, int]:
        """Возвращает диапазон чанков в окне [start_chunk, end_chunk)."""
        start_chunk = self.window_start_frame // FRAMES_PER_CHUNK
        end_chunk = (self.window_end_frame + FRAMES_PER_CHUNK - 1) // FRAMES_PER_CHUNK
        return start_chunk, end_chunk


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

        # Полные video_records и audio_tracks не загружаются
        self._video_records_full = _filter_normal_records(self._all_193)
        self._audio_tracks_full: Optional[np.ndarray] = None  # ленивая загрузка

        self._window: Optional[IndexWindow] = None
        self._lock = threading.Lock()

        # Параметры окна
        self.default_window_seconds = 300.0  # 5 минут
        self.min_window_chunks = 100  # минимальный размер окна в чанках

    @property
    def window(self) -> Optional[IndexWindow]:
        return self._window

    def open_window(
        self, center_frame: int, window_seconds: float = None
    ) -> IndexWindow:
        """
        Открывает окно вокруг center_frame.
        Если окно уже содержит этот кадр, возвращает текущее.
        """
        with self._lock:
            if self._window and self._window.contains_frame(center_frame):
                logger.debug(f"Кадр {center_frame} уже в окне")
                return self._window

            if window_seconds is None:
                window_seconds = self.default_window_seconds

            half_samples = int(window_seconds * 48000 / 2)
            half_frames = half_samples // 1920

            start_frame = max(0, center_frame - half_frames)
            end_frame = min(len(self._video_records_full), center_frame + half_frames)

            # Гарантируем минимальный размер окна
            if end_frame - start_frame < self.min_window_chunks * FRAMES_PER_CHUNK:
                end_frame = min(
                    len(self._video_records_full),
                    start_frame + self.min_window_chunks * FRAMES_PER_CHUNK,
                )

            self._window = self._build_window(start_frame, end_frame)
            logger.info(
                f"Окно открыто: кадры {start_frame}-{end_frame} "
                f"(чанки {self._window.chunk_range()})"
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

    def close(self):
        """Освобождает ресурсы (mmap будет закрыт при удалении объекта)."""
        self._window = None
        self._all_193 = None
        self._all_c9 = None

    # ------------------------------------------------------------------
    def _build_window(self, start_frame: int, end_frame: int) -> IndexWindow:
        """Строит IndexWindow для указанного диапазона кадров."""
        # 1. Видео-записи в диапазоне
        video_slice = self._video_records_full[start_frame:end_frame]

        # 2. Аудио-записи (ленивая загрузка полных треков при первом обращении)
        if self._audio_tracks_full is None:
            self._audio_tracks_full = build_audio_tracks(
                self._all_c9, track_filter=DEFAULT_TRACK_FILTER
            )

        # Определяем PTS-границы для аудио в этом окне
        start_pts = start_frame * SAMPLES_PER_CHUNK
        end_pts = end_frame * SAMPLES_PER_CHUNK

        audio_mask = (
            (self._audio_tracks_full['pts'] >= start_pts)
            & (self._audio_tracks_full['pts'] < end_pts)
        )
        audio_slice = self._audio_tracks_full[audio_mask]

        # 3. Чанки для видео в окне
        chunk_offsets, chunk_sizes = self._build_chunks_for_slice(video_slice)

        # 4. Аудио-чанки
        audio_chunks = self._build_audio_chunks_for_slice(audio_slice, start_frame)

        # 5. IDR в окне
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
        )

    def _build_chunks_for_slice(
        self, video_slice: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Строит чанки для заданного среза видео-записей."""
        if len(video_slice) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        offs = _abs_offset(video_slice)
        n = len(video_slice)
        nchunks = (n + FRAMES_PER_CHUNK - 1) // FRAMES_PER_CHUNK

        chunk_offs = np.empty(nchunks, dtype=np.int64)
        chunk_sizes = np.empty(nchunks, dtype=np.int64)

        for i in range(nchunks):
            start_idx = i * FRAMES_PER_CHUNK
            end_idx = min(start_idx + FRAMES_PER_CHUNK, n)

            chunk_offs[i] = offs[start_idx]
            if end_idx < n:
                next_off = offs[end_idx]
            else:
                next_off = np.uint64(self.mdat_end)
            chunk_sizes[i] = max(0, int(next_off) - int(chunk_offs[i]))

        return chunk_offs, chunk_sizes

    def _build_audio_chunks_for_slice(
        self, audio_slice: np.ndarray, window_start_frame: int
    ) -> List[Dict]:
        """Строит аудио-чанки для среза audio_tracks."""
        start_chunk = window_start_frame // FRAMES_PER_CHUNK
        num_chunks = (
            max(0, audio_slice['pts'].max() - audio_slice['pts'].min())
            // SAMPLES_PER_CHUNK
            + 1
        )

        if len(audio_slice) == 0 or num_chunks == 0:
            return [{} for _ in range(num_chunks)]

        pts_array = audio_slice['pts']
        chunks = []
        for chunk_idx in range(start_chunk, start_chunk + num_chunks):
            start_pts = chunk_idx * SAMPLES_PER_CHUNK
            end_pts = start_pts + SAMPLES_PER_CHUNK
            left = np.searchsorted(pts_array, start_pts, side='left')
            right = np.searchsorted(pts_array, end_pts, side='left')

            chunk_entries = defaultdict(list)
            for i in range(left, right):
                entry = {
                    'abs_offset': int(audio_slice[i]['abs_offset']),
                    'size1': int(audio_slice[i]['size1']),
                    'size2': int(audio_slice[i]['size2']),
                    'pts': int(audio_slice[i]['pts']),
                    'track': int(audio_slice[i]['track']),
                }
                chunk_entries[entry['track']].append(entry)

            chunks.append(dict(chunk_entries))

        return chunks