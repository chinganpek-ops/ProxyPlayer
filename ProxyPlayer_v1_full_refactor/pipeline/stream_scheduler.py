"""
stream_scheduler.py – приоритетный планировщик загрузки чанков (локальные индексы).
"""

import threading
import logging
from enum import Enum, auto
from typing import Optional, List, Set

from config.timebase import FRAMES_PER_CHUNK
from pipeline.adaptive_chunk import AdaptiveChunkStrategy

logger = logging.getLogger(__name__)


class PlaybackMode(Enum):
    NORMAL = auto()
    SEEK = auto()
    FAST_FORWARD = auto()
    PAUSE = auto()


class StreamScheduler:
    """Планировщик, работающий с локальными индексами чанков внутри окна."""

    def __init__(self, adaptive_strategy: AdaptiveChunkStrategy = None):
        self._adaptive = adaptive_strategy or AdaptiveChunkStrategy()
        self._mode = PlaybackMode.NORMAL
        self._current_chunk = 0        # локальный индекс
        self._total_chunks = 0         # количество чанков в окне

        self._target_chunk = 0         # локальный
        self._seek_priority_done = False

        self._direction = 0
        self._speed = 1.0

        self._loaded_chunks: Set[int] = set()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def set_normal_mode(self, current_local_chunk: int, total_local_chunks: int):
        with self._lock:
            self._mode = PlaybackMode.NORMAL
            self._current_chunk = current_local_chunk
            self._total_chunks = total_local_chunks

    def set_seek_mode(self, target_local_chunk: int, total_local_chunks: int):
        with self._lock:
            self._mode = PlaybackMode.SEEK
            self._target_chunk = target_local_chunk
            self._current_chunk = target_local_chunk
            self._total_chunks = total_local_chunks
            self._seek_priority_done = False
            self._loaded_chunks.clear()

    def set_fast_forward_mode(self, direction: int, speed: float,
                              current_local_chunk: int, total_local_chunks: int):
        with self._lock:
            self._mode = PlaybackMode.FAST_FORWARD
            self._direction = direction
            self._speed = speed
            self._current_chunk = current_local_chunk
            self._total_chunks = total_local_chunks

    def set_pause_mode(self, current_local_chunk: int, total_local_chunks: int):
        with self._lock:
            self._mode = PlaybackMode.PAUSE
            self._current_chunk = current_local_chunk
            self._total_chunks = total_local_chunks

    # ------------------------------------------------------------------
    def get_next_chunk(self) -> Optional[int]:
        """Возвращает локальный индекс следующего чанка для загрузки."""
        with self._lock:
            return self._get_next_chunk_locked()

    def _get_next_chunk_locked(self) -> Optional[int]:
        if self._total_chunks == 0:
            return None
        if self._mode == PlaybackMode.NORMAL:
            return self._next_normal()
        elif self._mode == PlaybackMode.SEEK:
            return self._next_seek()
        elif self._mode == PlaybackMode.FAST_FORWARD:
            return self._next_fast_forward()
        elif self._mode == PlaybackMode.PAUSE:
            return self._next_pause()
        return None

    # ------------------------------------------------------------------
    def _next_normal(self) -> Optional[int]:
        chunk = self._current_chunk
        while chunk in self._loaded_chunks and chunk < self._total_chunks:
            chunk += 1
        if chunk >= self._total_chunks:
            return None
        self._loaded_chunks.add(chunk)
        self._current_chunk = chunk + 1
        return chunk

    def _next_seek(self) -> Optional[int]:
        if not self._seek_priority_done:
            self._seek_priority_done = True
            chunk = self._target_chunk
            if chunk not in self._loaded_chunks and chunk < self._total_chunks:
                self._loaded_chunks.add(chunk)
                return chunk
        for offset in [1, 2]:
            for direction in [-1, 1]:
                chunk = self._target_chunk + offset * direction
                if 0 <= chunk < self._total_chunks and chunk not in self._loaded_chunks:
                    self._loaded_chunks.add(chunk)
                    return chunk
        self._mode = PlaybackMode.NORMAL
        self._current_chunk = max(self._target_chunk + 1, 0)
        return self._next_normal()

    def _next_fast_forward(self) -> Optional[int]:
        stride = self._adaptive.get_fast_forward_stride(self._speed)
        chunk = self._current_chunk
        attempts = 0
        while attempts < self._total_chunks:
            if chunk in self._loaded_chunks:
                chunk += self._direction * stride
                attempts += 1
                continue
            if 0 <= chunk < self._total_chunks:
                self._loaded_chunks.add(chunk)
                self._current_chunk = chunk + self._direction * stride
                return chunk
            break
        return None

    def _next_pause(self) -> Optional[int]:
        return self._next_normal()

    # ------------------------------------------------------------------
    def mark_chunk_failed(self, chunk_idx: int):
        with self._lock:
            self._loaded_chunks.discard(chunk_idx)

    def reset(self):
        with self._lock:
            self._loaded_chunks.clear()
            self._mode = PlaybackMode.NORMAL
            self._current_chunk = 0
            self._total_chunks = 0

    @property
    def loaded_count(self) -> int:
        with self._lock:
            return len(self._loaded_chunks)