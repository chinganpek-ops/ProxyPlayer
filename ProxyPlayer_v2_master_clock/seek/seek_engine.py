"""
seek_engine.py – асинхронный seek с отменой для ProxyPlayer v2.
Версия с поддержкой отмены чтения через cancel_event.
Исправления:
- Корректный расчёт диапазона чтения.
- Передача события отмены в WinSequentialReader.
- Обработка ошибок декодирования с сохранением причины.
"""

import threading
import logging
from typing import Optional, Callable

import numpy as np

from buffer.frame_buffer import FrameRingBuffer
from decode.decoder import Decoder
from index.lazy_index import LazyIndex
from file_io.win_sequential_reader import WinSequentialReader
from config.timebase import video_frame_to_pts
from index.moov_builder import _abs_offset

logger = logging.getLogger(__name__)

LOOKAHEAD_FRAMES = 12  # сколько кадров декодировать после целевого


class SeekRequest:
    """Представляет выполняющийся запрос перемотки с возможностью отмены."""

    def __init__(self, generation: int):
        self.generation = generation
        self._cancelled = threading.Event()
        self._done = threading.Event()
        self._success = False
        self._error: Optional[str] = None

    def cancel(self):
        """Отменяет текущий seek."""
        self._cancelled.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    @property
    def cancel_event(self) -> threading.Event:
        """Событие отмены, которое можно передать в reader для прерывания чтения."""
        return self._cancelled

    def _mark_done(self, success: bool, error: str = None):
        self._success = success
        self._error = error
        self._done.set()

    def wait(self, timeout: float = None) -> bool:
        """Блокирует поток до завершения seek. Возвращает True, если seek успешен."""
        self._done.wait(timeout)
        return self._success

    @property
    def error(self) -> Optional[str]:
        return self._error


class SeekEngine:
    """
    Асинхронный движок перемотки. Выполняет поиск ближайшего IDR,
    декодирует кадры и заполняет буфер. Поддерживает отмену.
    """

    def __init__(
        self,
        lazy_index: LazyIndex,
        decoder: Decoder,
        reader: WinSequentialReader,
    ):
        self._lazy_index = lazy_index
        self._decoder = decoder
        self._reader = reader
        self._current_request: Optional[SeekRequest] = None
        self._generation = 0
        self._lock = threading.Lock()

    def seek_async(
        self,
        frame_idx: int,
        on_complete: Callable[[FrameRingBuffer], None],
        on_error: Callable[[str], None] = None,
    ) -> SeekRequest:
        """
        Запускает асинхронный seek к кадру frame_idx.
        Предыдущий запрос немедленно отменяется (через cancel_event),
        но поток не ожидается, чтобы не блокировать интерфейс.
        """
        with self._lock:
            self._generation += 1
            gen = self._generation
            # Отменяем предыдущий запрос, если он есть
            if self._current_request and not self._current_request._done.is_set():
                self._current_request.cancel()
            request = SeekRequest(gen)
            self._current_request = request

        def _run():
            try:
                buffer = self._seek_sync(frame_idx, request)
                if request.is_cancelled:
                    return
                with self._lock:
                    if gen != self._generation:
                        return  # устаревший запрос
                request._mark_done(True)
                on_complete(buffer)
            except Exception as e:
                logger.exception("Seek error")
                request._mark_done(False, str(e))
                if on_error:
                    on_error(str(e))

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return request

    def seek_sync(self, frame_idx: int) -> FrameRingBuffer:
        """Синхронный seek (блокирует поток). Используется для тестов и совместимости."""
        request = SeekRequest(0)
        return self._seek_sync(frame_idx, request)

    def cancel_current(self):
        """Отменяет текущий выполняющийся seek."""
        with self._lock:
            if self._current_request and not self._current_request._done.is_set():
                self._current_request.cancel()
                self._current_request = None

    # ------------------------------------------------------------------
    def _seek_sync(self, frame_idx: int, request: SeekRequest) -> FrameRingBuffer:
        """Основная логика seek. Может быть отменена через request."""
        window = self._lazy_index.open_window(frame_idx)
        if window is None or len(window.video_records) == 0:
            raise RuntimeError("Не удалось открыть окно индекса")

        # Ищем ближайший IDR в окне
        idr_indices = window.idr_frames
        if len(idr_indices) == 0:
            raise RuntimeError("В окне не найдены IDR-кадры")

        # Переводим глобальный frame_idx в локальный индекс внутри окна
        local_target = frame_idx - window.window_start_frame
        pos = np.searchsorted(idr_indices, local_target, side='right') - 1
        if pos < 0:
            pos = 0
        local_idr = idr_indices[pos]

        # Декодируем от IDR до целевого кадра + запас
        end_local = min(len(window.video_records) - 1,
                        max(local_idr + LOOKAHEAD_FRAMES, local_target))

        # Определяем границы чтения в файле
        first_rec = window.video_records[local_idr]
        start_offset = int(_abs_offset(first_rec))

        if end_local + 1 < len(window.video_records):
            next_rec = window.video_records[end_local + 1]
            end_offset = int(_abs_offset(next_rec))
        else:
            end_offset = self._lazy_index.mdat_end

        read_size = end_offset - start_offset
        if read_size <= 0:
            raise RuntimeError("Некорректный размер данных для seek")

        # Проверка отмены перед чтением
        if request.is_cancelled:
            raise RuntimeError("Seek отменён")

        # Передаём событие отмены в reader
        self._reader.set_cancel_event(request.cancel_event)
        try:
            raw_data = self._reader.read_sequential(start_offset, read_size)
        finally:
            self._reader.clear_cancel_event()

        if request.is_cancelled:
            # Операция отменена во время чтения, возвращаем пустой буфер
            return FrameRingBuffer(max_frames=2)

        if not raw_data:
            return FrameRingBuffer(max_frames=2)

        # Декодируем кадры
        buffer = FrameRingBuffer(max_frames=300)
        first_decode_error = None

        for i in range(local_idr, end_local + 1):
            if request.is_cancelled:
                raise RuntimeError("Seek отменён")

            rec = window.video_records[i]
            off = int(_abs_offset(rec))
            rel_start = off - start_offset

            if i + 1 < len(window.video_records):
                next_off = int(_abs_offset(window.video_records[i + 1]))
                size = next_off - off
            else:
                size = len(raw_data) - rel_start

            if rel_start < 0 or size <= 0:
                continue

            sample = raw_data[rel_start:rel_start + size]
            try:
                filtered = self._decoder.filter_avcc(sample)
            except Exception as e:
                if first_decode_error is None:
                    first_decode_error = e
                logger.debug(f"Ошибка фильтрации AVCC для кадра {i}: {e}")
                continue
            if not filtered:
                continue

            try:
                frames = self._decoder.decode_sample(filtered)
                for frame in frames:
                    pts = video_frame_to_pts(window.window_start_frame + i)
                    if not buffer.try_push(frame, pts):
                        logger.debug(f"Буфер заполнен на кадре {i}, прекращаем декодирование")
                        request.cancel()
                        break
                if request.is_cancelled:
                    break
            except Exception as e:
                if first_decode_error is None:
                    first_decode_error = e
                logger.debug(f"Ошибка декодирования кадра {i}: {e}")
                continue

        if buffer.count == 0:
            error_msg = "Не удалось декодировать ни одного кадра при seek"
            if first_decode_error is not None:
                error_msg += f": {first_decode_error}"
            raise RuntimeError(error_msg)

        return buffer