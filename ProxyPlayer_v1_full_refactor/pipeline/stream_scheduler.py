"""
stream_scheduler.py – приоритетный планировщик загрузки чанков для ProxyPlayer v1.
Определяет, какой чанк загружать следующим в зависимости от режима:
- NORMAL: последовательно вперёд с опережением
- SEEK: приоритет целевому чанку, затем соседние
- FAST_FORWARD: каждый N-й чанк в зависимости от скорости
- PAUSE: заполнение буфера вперёд
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
    """
    Планировщик загрузки чанков.
    
    В зависимости от режима воспроизведения определяет оптимальный порядок
    загрузки чанков, чтобы минимизировать задержки и обеспечить плавное
    воспроизведение.
    """

    def __init__(self, adaptive_strategy: AdaptiveChunkStrategy = None):
        self._adaptive = adaptive_strategy or AdaptiveChunkStrategy()

        self._mode = PlaybackMode.NORMAL
        self._current_chunk = 0
        self._total_chunks = 0

        # Для режима SEEK
        self._target_chunk = 0
        self._seek_priority_done = False

        # Для режима FAST_FORWARD
        self._direction = 0  # 1 = вперёд, -1 = назад
        self._speed = 1.0

        # Множество уже загруженных чанков (для избежания повторов)
        self._loaded_chunks: Set[int] = set()

        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Установка режимов
    # ------------------------------------------------------------------
    def set_normal_mode(self, current_chunk: int, total_chunks: int):
        """Переключение в режим нормального воспроизведения."""
        with self._lock:
            self._mode = PlaybackMode.NORMAL
            self._current_chunk = current_chunk
            self._total_chunks = total_chunks
            logger.debug(f"Режим NORMAL: текущий чанк {current_chunk}")

    def set_seek_mode(self, target_chunk: int, total_chunks: int):
        """Переключение в режим приоритетной загрузки после seek."""
        with self._lock:
            self._mode = PlaybackMode.SEEK
            self._target_chunk = target_chunk
            self._current_chunk = target_chunk
            self._total_chunks = total_chunks
            self._seek_priority_done = False
            self._loaded_chunks.clear()
            logger.debug(f"Режим SEEK: целевой чанк {target_chunk}")

    def set_fast_forward_mode(self, direction: int, speed: float,
                              current_chunk: int, total_chunks: int):
        """Переключение в режим ускоренной перемотки."""
        with self._lock:
            self._mode = PlaybackMode.FAST_FORWARD
            self._direction = direction
            self._speed = speed
            self._current_chunk = current_chunk
            self._total_chunks = total_chunks
            logger.debug(f"Режим FAST_FORWARD: направление {direction}, скорость x{speed}")

    def set_pause_mode(self, current_chunk: int, total_chunks: int):
        """Переключение в режим паузы (заполнение буфера)."""
        with self._lock:
            self._mode = PlaybackMode.PAUSE
            self._current_chunk = current_chunk
            self._total_chunks = total_chunks
            logger.debug(f"Режим PAUSE: текущий чанк {current_chunk}")

    # ------------------------------------------------------------------
    # Получение следующего чанка
    # ------------------------------------------------------------------
    def get_next_chunk(self) -> Optional[int]:
        """
        Возвращает индекс следующего чанка для загрузки.
        Вызывается ChunkPipeline, когда готов принять новую работу.
        """
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
    # Стратегии для каждого режима
    # ------------------------------------------------------------------
    def _next_normal(self) -> Optional[int]:
        """Последовательная загрузка вперёд."""
        chunk = self._current_chunk
        # Пропускаем уже загруженные
        while chunk in self._loaded_chunks and chunk < self._total_chunks:
            chunk += 1
        if chunk >= self._total_chunks:
            return None
        self._loaded_chunks.add(chunk)
        self._current_chunk = chunk + 1
        return chunk

    def _next_seek(self) -> Optional[int]:
        """
        Приоритетная загрузка: сначала целевой чанк,
        затем по два чанка влево и вправо от цели.
        """
        if not self._seek_priority_done:
            # Первый вызов – целевой чанк
            self._seek_priority_done = True
            chunk = self._target_chunk
            if chunk not in self._loaded_chunks and chunk < self._total_chunks:
                self._loaded_chunks.add(chunk)
                return chunk

        # После цели – соседние чанки
        for offset in [1, 2]:   # сначала ±1, потом ±2
            for direction in [-1, 1]:
                chunk = self._target_chunk + offset * direction
                if 0 <= chunk < self._total_chunks and chunk not in self._loaded_chunks:
                    self._loaded_chunks.add(chunk)
                    return chunk

        # Если соседние загружены – переходим в нормальный режим
        self._mode = PlaybackMode.NORMAL
        self._current_chunk = max(self._target_chunk + 1, 0)
        return self._next_normal()

    def _next_fast_forward(self) -> Optional[int]:
        """
        Загрузка каждого N-го чанка в направлении перемотки.
        N = stride = max(1, int(speed)).
        """
        stride = self._adaptive.get_fast_forward_stride(self._speed)
        chunk = self._current_chunk

        # Ищем следующий незагруженный чанк с учётом шага
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
        """Заполнение буфера вперёд без ограничения скорости."""
        return self._next_normal()

    # ------------------------------------------------------------------
    # Статистика
    # ------------------------------------------------------------------
    @property
    def pending_chunks(self) -> List[int]:
        """Возвращает список ещё не загруженных чанков (для отладки)."""
        with self._lock:
            all_chunks = set(range(self._total_chunks))
            return sorted(all_chunks - self._loaded_chunks)

    @property
    def loaded_count(self) -> int:
        """Количество уже загруженных чанков."""
        with self._lock:
            return len(self._loaded_chunks)

    def mark_chunk_failed(self, chunk_idx: int):
        """Убирает чанк из списка загруженных при ошибке (для повтора)."""
        with self._lock:
            self._loaded_chunks.discard(chunk_idx)

    def reset(self):
        """Полный сброс состояния."""
        with self._lock:
            self._loaded_chunks.clear()
            self._mode = PlaybackMode.NORMAL
            self._current_chunk = 0
            self._total_chunks = 0