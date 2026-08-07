"""
frame_buffer.py – потокобезопасный кольцевой буфер видеокадров (целые PTS).
Поддержка циклического сдвига для трёхбуферной очереди.
Добавлены методы peek_all() и push_front() для точного выбора ближайшего кадра.
"""

import threading
import logging
from typing import List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class FrameRingBuffer:
    """
    Кольцевой буфер для хранения кадров (pts: int, np.ndarray).
    Потокобезопасен: использует threading.Lock + threading.Condition.
    Поддерживает циклический сдвиг для трёхбуферной очереди.
    """

    def __init__(self, max_frames: int = 800, enforce_contiguous: bool = True):
        if max_frames < 2:
            raise ValueError("max_frames должен быть >= 2")
        self.max_frames = max_frames
        self._enforce_contiguous = enforce_contiguous

        self._buffer: List[Optional[Tuple[int, np.ndarray]]] = [None] * max_frames
        self._rindex = 0
        self._windex = 0
        self._size = 0
        self._latest_pts = 0
        self._keep_last: Optional[Tuple[int, np.ndarray]] = None

        self._mutex = threading.Lock()
        self._cond = threading.Condition(self._mutex)

        # Статистика
        self._total_pushed = 0
        self._total_dropped = 0
        self._total_contiguous_conversions = 0

        logger.info(f"FrameRingBuffer создан: max_frames={max_frames}")

    # ------------------------------------------------------------------
    # Добавление кадров
    # ------------------------------------------------------------------
    def push(self, frame: np.ndarray, pts: int):
        """Добавляет кадр в буфер. Блокируется при заполнении."""
        frame = self._ensure_contiguous(frame)
        with self._mutex:
            while self._size >= self.max_frames:
                self._cond.wait(timeout=0.1)
            self._buffer[self._windex] = (pts, frame)
            self._windex = (self._windex + 1) % self.max_frames
            self._size += 1
            self._latest_pts = pts
            self._total_pushed += 1
            self._cond.notify()

    def try_push(self, frame: np.ndarray, pts: int) -> bool:
        """Неблокирующая попытка добавления кадра."""
        frame = self._ensure_contiguous(frame)
        with self._mutex:
            if self._size >= self.max_frames:
                return False
            self._buffer[self._windex] = (pts, frame)
            self._windex = (self._windex + 1) % self.max_frames
            self._size += 1
            self._latest_pts = pts
            self._total_pushed += 1
            self._cond.notify()
            return True

    # ------------------------------------------------------------------
    # Чтение кадров
    # ------------------------------------------------------------------
    def peek_first(self) -> Optional[Tuple[int, np.ndarray]]:
        """Возвращает первый кадр без удаления."""
        with self._mutex:
            if self._size == 0:
                return None
            entry = self._buffer[self._rindex]
            return entry

    def peek_all(self) -> List[Tuple[int, np.ndarray]]:
        """Возвращает список всех кадров в буфере без их удаления."""
        with self._mutex:
            result = []
            idx = self._rindex
            for _ in range(self._size):
                entry = self._buffer[idx]
                if entry is not None:
                    result.append(entry)
                idx = (idx + 1) % self.max_frames
            return result

    def advance(self):
        """Удаляет первый кадр из очереди."""
        with self._mutex:
            if self._size > 0:
                self._rindex = (self._rindex + 1) % self.max_frames
                self._size -= 1
                self._total_dropped += 1
                self._cond.notify()

    def push_front(self, frame: np.ndarray, pts: int):
        """Вставляет кадр в начало буфера."""
        frame = self._ensure_contiguous(frame)
        with self._mutex:
            if self._size >= self.max_frames:
                logger.warning("push_front: буфер заполнен, кадр не вставлен")
                return
            # Сдвигаем rindex назад на одну позицию
            self._rindex = (self._rindex - 1) % self.max_frames
            self._buffer[self._rindex] = (pts, frame)
            self._size += 1

    def drop_until(self, pts_threshold: int):
        """Удаляет все кадры с PTS < pts_threshold."""
        with self._mutex:
            dropped = 0
            while self._size > 0:
                entry = self._buffer[self._rindex]
                if entry is None:
                    self._rindex = (self._rindex + 1) % self.max_frames
                    self._size -= 1
                    self._total_dropped += 1
                    dropped += 1
                    continue
                if entry[0] < pts_threshold:
                    self._rindex = (self._rindex + 1) % self.max_frames
                    self._size -= 1
                    self._total_dropped += 1
                    dropped += 1
                else:
                    break
            if dropped > 0:
                self._cond.notify_all()

    # ------------------------------------------------------------------
    # Keep-last кадр (для паузы или пустого буфера)
    # ------------------------------------------------------------------
    def update_keep_last(self, frame: np.ndarray, pts: int):
        """Сохраняет кадр как последний показанный."""
        frame = self._ensure_contiguous(frame)
        with self._mutex:
            self._keep_last = (pts, frame)

    def get_keep_last(self) -> Optional[np.ndarray]:
        """Возвращает последний сохранённый кадр."""
        with self._mutex:
            if self._keep_last:
                return self._keep_last[1]
            return None

    def get_keep_last_pts(self) -> int:
        """Возвращает PTS последнего показанного кадра или 0."""
        with self._mutex:
            if self._keep_last:
                return self._keep_last[0]
            return 0

    # ------------------------------------------------------------------
    # Операции для трёхбуферной очереди
    # ------------------------------------------------------------------
    def steal_keep_last(self) -> Optional[Tuple[int, np.ndarray]]:
        """
        Атомарно забирает keep_last кадр и очищает его.
        Используется при циклическом сдвиге буферов.
        """
        with self._mutex:
            kl = self._keep_last
            self._keep_last = None
            return kl

    def copy_last_frame_to(self, target: 'FrameRingBuffer'):
        """
        Копирует последний кадр (keep_last) в целевой буфер.
        Если keep_last отсутствует, пытается взять последний кадр из очереди.
        """
        frame = self.get_keep_last()
        if frame is not None:
            target.update_keep_last(frame, self._latest_pts)
        else:
            entry = self.peek_first()
            if entry is not None:
                target.update_keep_last(entry[1], entry[0])

    # ------------------------------------------------------------------
    # Очистка и состояние
    # ------------------------------------------------------------------
    def clear(self):
        """Полная очистка буфера."""
        with self._mutex:
            self._rindex = 0
            self._windex = 0
            self._size = 0
            self._latest_pts = 0
            self._keep_last = None
            for i in range(self.max_frames):
                self._buffer[i] = None
            self._cond.notify_all()

    @property
    def count(self) -> int:
        with self._mutex:
            return self._size

    @property
    def is_empty(self) -> bool:
        with self._mutex:
            return self._size == 0

    @property
    def free_slots(self) -> int:
        with self._mutex:
            return self.max_frames - self._size

    def latest_pts(self) -> int:
        with self._mutex:
            return self._latest_pts

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------
    def _ensure_contiguous(self, frame: np.ndarray) -> np.ndarray:
        if self._enforce_contiguous and not frame.flags['C_CONTIGUOUS']:
            self._total_contiguous_conversions += 1
            return np.ascontiguousarray(frame)
        return frame