"""
seek_engine.py – асинхронный seek с отменой для ProxyPlayer v2.
Оптимизирован для стабильной работы с одним воркер-потоком.

Изменения (исходные):
- Один воркер-поток с очередью команд (никаких параллельных seek).
- Новый запрос отменяет предыдущий, но не создаёт новый поток.
- Каждый запрос использует свой WinSequentialReader (изолированные чтения).
- Поддержка поколений: старые колбэки игнорируются.

ИЗМЕНЕНИЯ (правки продакшен-ревью):
- _seek_sync_internal(): переполнение буфера кадров seek-а (max_frames=300)
  раньше сигнализировалось через request.cancel() — тот же механизм, что и
  реальная отмена запроса новым seek. Из-за этого _process_seek() видел
  request.is_cancelled=True и просто возвращался, ни разу не вызвав
  on_complete()/on_error() — уже готовый и валидный результат seek терялся
  молча. Теперь переполнение буфера обозначается отдельным локальным
  флагом buffer_full, не трогающим состояние request. request.is_cancelled
  теперь означает ровно то, для чего он существует — отмену новым seek.

Остальная логика (бинарный поиск IDR, чтение диапазона, декодирование) не
менялась.
"""

import threading
import logging
import queue
from typing import Optional, Callable

import numpy as np

from buffer.frame_buffer import FrameRingBuffer
from decode.decoder import Decoder
from index.lazy_index import LazyIndex
from file_io.win_sequential_reader import WinSequentialReader
from config.timebase import video_frame_to_pts
from index.moov_builder import _abs_offset

logger = logging.getLogger(__name__)

LOOKAHEAD_FRAMES = 12


class SeekRequest:
    """Представляет выполняющийся запрос перемотки с возможностью отмены."""

    def __init__(self, generation: int):
        self.generation = generation
        self._cancelled = threading.Event()
        self._done = threading.Event()
        self._success = False
        self._error: Optional[str] = None

    def cancel(self):
        self._cancelled.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    @property
    def cancel_event(self) -> threading.Event:
        return self._cancelled

    def _mark_done(self, success: bool, error: str = None):
        self._success = success
        self._error = error
        self._done.set()

    def wait(self, timeout: float = None) -> bool:
        self._done.wait(timeout)
        return self._success

    @property
    def error(self) -> Optional[str]:
        return self._error


class SeekEngine:
    """
    Асинхронный движок перемотки. Один воркер-поток обрабатывает все запросы.
    Новый seek отменяет текущий и сразу становится в очередь.
    """

    def __init__(
        self,
        lazy_index: LazyIndex,
        decoder: Decoder,
        reader: WinSequentialReader,
    ):
        self._lazy_index = lazy_index
        self._decoder = decoder
        self._base_reader = reader  # базовый ридер, для совместимости (не используется напрямую)
        self._current_request: Optional[SeekRequest] = None
        self._generation = 0
        self._lock = threading.Lock()

        # Очередь команд для воркера
        self._command_queue = queue.Queue()

        # Запускаем воркер
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="SeekWorker")
        self._worker_thread.start()

    def _worker_loop(self):
        """Основной цикл воркера: берёт команды из очереди и выполняет их."""
        while True:
            try:
                command, args, kwargs = self._command_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if command == "seek":
                self._process_seek(*args, **kwargs)
            elif command == "stop":
                break

    def _process_seek(self, request: SeekRequest, frame_idx: int, on_complete, on_error):
        """Выполняет seek-запрос внутри воркера."""
        try:
            buffer = self._seek_sync_internal(frame_idx, request)
            if request.is_cancelled:
                return
            with self._lock:
                if request.generation != self._generation:
                    return  # устаревший запрос
            request._mark_done(True)
            if on_complete:
                on_complete(buffer)
        except Exception as e:
            logger.exception("Seek error")
            request._mark_done(False, str(e))
            if on_error:
                on_error(str(e))

    def seek_async(
        self,
        frame_idx: int,
        on_complete: Callable[[FrameRingBuffer], None],
        on_error: Callable[[str], None] = None,
    ) -> SeekRequest:
        """
        Ставит запрос в очередь. Предыдущий отменяется.
        Возвращает SeekRequest для отслеживания/отмены.
        """
        with self._lock:
            self._generation += 1
            gen = self._generation
            if self._current_request and not self._current_request._done.is_set():
                self._current_request.cancel()
            request = SeekRequest(gen)
            self._current_request = request

        # Отправляем команду воркеру
        self._command_queue.put(("seek", (request, frame_idx, on_complete, on_error), {}))

        return request

    def seek_sync(self, frame_idx: int) -> FrameRingBuffer:
        """Синхронный seek (блокирует поток). Для тестов и совместимости."""
        request = SeekRequest(0)
        return self._seek_sync_internal(frame_idx, request)

    def cancel_current(self):
        """Отменяет текущий выполняющийся seek."""
        with self._lock:
            if self._current_request and not self._current_request._done.is_set():
                self._current_request.cancel()

    def _seek_sync_internal(self, frame_idx: int, request: SeekRequest) -> FrameRingBuffer:
        """Основная логика seek. Может быть отменена через request."""
        window = self._lazy_index.open_window(frame_idx)
        if window is None or len(window.video_records) == 0:
            raise RuntimeError("Не удалось открыть окно индекса")

        idr_indices = window.idr_frames
        if len(idr_indices) == 0:
            raise RuntimeError("В окне не найдены IDR-кадры")

        local_target = frame_idx - window.window_start_frame
        pos = np.searchsorted(idr_indices, local_target, side='right') - 1
        if pos < 0:
            pos = 0
        local_idr = idr_indices[pos]

        end_local = min(len(window.video_records) - 1,
                        max(local_idr + LOOKAHEAD_FRAMES, local_target))

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

        if request.is_cancelled:
            raise RuntimeError("Seek отменён")

        # Создаём отдельный ридер для этого seek
        reader = WinSequentialReader(self._lazy_index.mp4_path, rate_limit=0, overlapped=False)
        try:
            reader.set_cancel_event(request.cancel_event)
            raw_data = reader.read_sequential(start_offset, read_size)
        finally:
            reader.set_cancel_event(threading.Event())  # сбрасываем
            reader.close()

        if request.is_cancelled:
            return FrameRingBuffer(max_frames=2)

        if not raw_data:
            return FrameRingBuffer(max_frames=2)

        buffer = FrameRingBuffer(max_frames=300)
        first_decode_error = None
        buffer_full = False  # достижение предела буфера seek-а — НЕ отмена

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
                continue
            if not filtered:
                continue

            try:
                frames = self._decoder.decode_sample(filtered)
                for frame in frames:
                    pts = video_frame_to_pts(window.window_start_frame + i)
                    if not buffer.try_push(frame, pts):
                        # Буфер seek-а физически заполнен (max_frames=300) —
                        # это естественный предел, а не отмена запроса.
                        # Результат остаётся валидным и должен быть отдан
                        # вызывающему через on_complete().
                        buffer_full = True
                        break
                if buffer_full:
                    break
            except Exception as e:
                if first_decode_error is None:
                    first_decode_error = e
                continue

        if buffer.count == 0:
            error_msg = "Не удалось декодировать ни одного кадра при seek"
            if first_decode_error is not None:
                error_msg += f": {first_decode_error}"
            raise RuntimeError(error_msg)

        return buffer

    def close(self):
        """Останавливает воркер и освобождает ресурсы."""
        self._command_queue.put(("stop", (), {}))
        self._worker_thread.join(timeout=2.0)
