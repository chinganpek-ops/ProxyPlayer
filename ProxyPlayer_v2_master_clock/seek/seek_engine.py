"""
seek_engine.py – пул воркеров для асинхронного seek (v3).

ИСТОРИЯ ИЗМЕНЕНИЙ:
- v1: один воркер-поток с очередью команд.
- v2: буфер seek-а (max_frames=300) и разделение "отмена запроса" vs
  "буфер физически заполнен" (buffer_full).
- v3 (эта версия) — переход на пул воркеров по мотивам диагностики
  зависания SeekWorker на блокирующем сетевом чтении (SMB, ReadFile без
  таймаута) и последующего аварийного завершения процесса при закрытии
  плеера, пока поток ещё жив и держит хэндл. Требования: (1) таймауты —
  и на отмену конкретной задачи, и общий watchdog на случай, если новый
  seek вообще не пришёл, чтобы отменить зависшую; (2) каждый воркер
  ЖЁСТКО привязан к своему FrameRingBuffer на весь срок жизни воркера —
  без гонки данных; (3) передача результата — через очередь-сигнал
  (_result_queue), а не прямым вызовом колбэка из потока воркера;
  (4) вся координация — на отдельном потоке-диспетчере, ни GUI, ни
  сами воркеры не блокируются.

СОВМЕСТИМОСТЬ С НОВЫМ WinSequentialReader:
- В _do_seek() используется новый API: конструктор
  WinSequentialReader(path, buffer_size=..., read_timeout_ms=...,
  use_no_buffering=...), методы seek(offset) и read(size).
- Убраны вызовы set_cancel_event()/read_sequential(), которых нет в
  новом ридере. Отмена обеспечивается таймаутом внутри ридера и
  внешним CancelSynchronousIo из координатора (при необходимости).
  Если read() возвращает b'' – это трактуется как отмена/ошибка/EOF.
"""

import threading
import logging
import queue
import time
import ctypes
from ctypes import wintypes
from pathlib import Path
from typing import Optional, Callable, List

import numpy as np

from buffer.frame_buffer import FrameRingBuffer
from decode.decoder import Decoder
from index.lazy_index import LazyIndex
from file_io.win_sequential_reader import WinSequentialReader
from config.timebase import video_frame_to_pts
from index.moov_builder import _abs_offset

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Windows API для принудительной отмены синхронного I/O чужого потока
# ------------------------------------------------------------------
THREAD_TERMINATE = 0x0001
OpenThread = ctypes.windll.kernel32.OpenThread
OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
OpenThread.restype = wintypes.HANDLE
CancelSynchronousIo = ctypes.windll.kernel32.CancelSynchronousIo
CancelSynchronousIo.argtypes = [wintypes.HANDLE]
CancelSynchronousIo.restype = wintypes.BOOL
CloseHandle = ctypes.windll.kernel32.CloseHandle
CloseHandle.argtypes = [wintypes.HANDLE]
CloseHandle.restype = wintypes.BOOL

LOOKAHEAD_FRAMES = 12
DEFAULT_WORKERS = 3
BUFFER_MAX_FRAMES = 300

_ACQUIRE_CANCEL_HARD_TIMEOUT_SEC = 10.0
_TASK_WATCHDOG_TIMEOUT_SEC = 20.0
_HEALTH_CHECK_INTERVAL_SEC = 0.2


class SeekRequest:
    """Хэндл запроса перемотки для внешнего кода."""

    def __init__(self, generation: int,
                 on_complete: Optional[Callable[[FrameRingBuffer], None]] = None,
                 on_error: Optional[Callable[[str], None]] = None):
        self.generation = generation
        self.on_complete = on_complete
        self.on_error = on_error
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

    @property
    def is_done(self) -> bool:
        """Завершён ли запрос (успешно или с ошибкой)."""
        return self._done.is_set()

    def mark_done(self, success: bool, error: str = None):
        """
        Публичное завершение запроса. Раньше вызывалось как _mark_done() из
        SeekEngine — то есть модуль обращался к приватному методу чужого
        объекта. Метод остаётся под старым именем как псевдоним, чтобы
        правка была совместимой.
        """
        self._success = success
        self._error = error
        self._done.set()

    # Псевдоним для совместимости с существующими вызовами.
    _mark_done = mark_done

    def wait(self, timeout: float = None) -> bool:
        self._done.wait(timeout)
        return self._success

    @property
    def error(self) -> Optional[str]:
        return self._error


class SeekWorker:
    """
    Постоянный воркер с жёстко привязанными буфером И декодером.

    Каждый воркер владеет СОБСТВЕННЫМ экземпляром Decoder на весь срок
    жизни. Раньше все воркеры и VideoDecoderStage конвейера использовали
    один общий Decoder, переданный из StreamController. PyAV CodecContext
    не рассчитан на параллельное использование: одновременный вызов
    decode_sample() из нескольких потоков (до трёх SeekWorker плюс
    непрерывно работающий VideoDecoderStage) — обращение к одному
    нативному контексту из разных потоков, что приводит к порче состояния
    декодера и аварийному завершению процесса без исключения Python.

    Декодер создаётся лениво, при первой задаче: если seek за сеанс не
    выполнялся, лишние контексты не создаются. Закрывается в stop().
    """

    def __init__(self, worker_id: int, decoder_factory=None):
        self.id = worker_id
        self.buffer = FrameRingBuffer(max_frames=BUFFER_MAX_FRAMES)
        self._decoder_factory = decoder_factory
        self.decoder = None
        self.state = 'free'                      # 'free' | 'busy' | 'stuck'
        self.thread: Optional[threading.Thread] = None
        self.task_start_ts = 0.0
        self.current_request: Optional[SeekRequest] = None
        self.current_frame_idx = 0

    def get_decoder(self):
        """Возвращает собственный декодер воркера, создавая его при первом обращении."""
        if self.decoder is None and self._decoder_factory is not None:
            self.decoder = self._decoder_factory()
            logger.info("SeekWorker %d: создан собственный декодер", self.id)
        return self.decoder

    def close_decoder(self):
        """Освобождает нативный контекст декодера."""
        if self.decoder is not None:
            try:
                self.decoder.close()
            except Exception:
                logger.exception("SeekWorker %d: ошибка закрытия декодера", self.id)
            self.decoder = None


class _SeekCommand:
    __slots__ = ('frame_idx', 'request')

    def __init__(self, frame_idx: int, request: SeekRequest):
        self.frame_idx = frame_idx
        self.request = request


class _SeekResult:
    __slots__ = ('worker', 'request', 'success', 'cancelled', 'error_msg')

    def __init__(self, worker: SeekWorker, request: SeekRequest,
                 success: bool, cancelled: bool, error_msg: Optional[str]):
        self.worker = worker
        self.request = request
        self.success = success
        self.cancelled = cancelled
        self.error_msg = error_msg


class SeekEngine:
    """Пул воркеров для асинхронного seek."""

    def __init__(self, lazy_index: LazyIndex, decoder: Decoder = None,
                 reader: Optional[WinSequentialReader] = None,
                 num_workers: int = DEFAULT_WORKERS,
                 decoder_factory=None):
        """
        decoder_factory — функция без аргументов, создающая НОВЫЙ Decoder.
        Каждый воркер получает собственный экземпляр: PyAV CodecContext не
        допускает параллельного использования из нескольких потоков, а
        seek-воркеры работают одновременно друг с другом и с
        VideoDecoderStage конвейера.

        Аргумент decoder оставлен для обратной совместимости. Если фабрика
        не передана, все воркеры разделят этот единственный декодер —
        поведение как раньше, с соответствующим риском; в лог пишется
        предупреждение, чтобы такая конфигурация не осталась незамеченной.
        """
        self._lazy_index = lazy_index
        self._decoder = decoder
        self._base_reader = reader  # не используется, оставлен для совместимости

        if decoder_factory is None:
            if decoder is None:
                raise ValueError("Нужен decoder_factory либо decoder")
            logger.warning(
                "SeekEngine создан без decoder_factory: все воркеры будут "
                "использовать ОДИН декодер совместно с конвейером. PyAV не "
                "поддерживает параллельное декодирование на одном контексте — "
                "возможно аварийное завершение процесса."
            )
            decoder_factory = lambda: decoder

        self._decoder_factory = decoder_factory
        self.workers: List[SeekWorker] = [
            SeekWorker(i, decoder_factory) for i in range(max(1, num_workers))
        ]

        self._command_queue: "queue.Queue[_SeekCommand]" = queue.Queue()
        self._result_queue: "queue.Queue[_SeekResult]" = queue.Queue()

        self._lock = threading.Lock()
        self._generation = 0
        self._current_request: Optional[SeekRequest] = None

        self._closing = threading.Event()
        self._coordinator_thread = threading.Thread(
            target=self._coordinator_loop, daemon=True, name="SeekCoordinator"
        )
        self._coordinator_thread.start()

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------
    def seek_async(
        self,
        frame_idx: int,
        on_complete: Callable[[FrameRingBuffer], None],
        on_error: Optional[Callable[[str], None]] = None,
    ) -> SeekRequest:
        """Ставит запрос в очередь координатора. Предыдущий запрос отменяется."""
        with self._lock:
            self._generation += 1
            gen = self._generation
            if self._current_request is not None and not self._current_request._done.is_set():
                self._current_request.cancel()
            request = SeekRequest(gen, on_complete=on_complete, on_error=on_error)
            self._current_request = request

        self._command_queue.put(_SeekCommand(frame_idx, request))
        return request

    def seek_sync(self, frame_idx: int) -> FrameRingBuffer:
        """Синхронный seek (блокирует вызывающий поток). Для тестов/совместимости."""
        done = threading.Event()
        result = {}

        def _on_complete(buf):
            result['buffer'] = buf
            done.set()

        def _on_error(msg):
            result['error'] = msg
            done.set()

        self.seek_async(frame_idx, _on_complete, _on_error)
        done.wait()
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result.get('buffer', FrameRingBuffer(max_frames=2))

    def get_workers_state(self) -> List[dict]:
        """
        Состояние пула воркеров для диагностики.

        Заменяет чтение self.workers и приватных полей воркера напрямую:
        телеметрия и тесты получают снимок, не завися от внутреннего
        устройства SeekWorker.
        """
        out = []
        for w in self.workers:
            thread = w.thread
            out.append({
                "id": w.id,
                "state": w.state,
                "buffer_count": getattr(w.buffer, "count", None),
                "alive": bool(thread.is_alive()) if thread is not None else False,
                "frame_idx": w.current_frame_idx,
                "has_decoder": w.decoder is not None,
            })
        return out

    def get_state(self) -> dict:
        """Общее состояние движка перемотки."""
        return {
            "generation": self._generation,
            "workers": self.get_workers_state(),
            "command_queue": self._command_queue.qsize(),
            "result_queue": self._result_queue.qsize(),
            "closing": self._closing.is_set(),
        }

    def cancel_current(self):
        """Отменяет текущий выполняющийся/ожидающий запрос."""
        with self._lock:
            if self._current_request is not None and not self._current_request._done.is_set():
                self._current_request.cancel()

    def close(self):
        """Останавливает координатор и по возможности прерывает все воркеры."""
        self._closing.set()
        self._command_queue.put(None)  # будим координатор без ожидания poll-таймаута
        if self._coordinator_thread.is_alive():
            self._coordinator_thread.join(timeout=3.0)

        # Финальная попытка добить воркеры — best effort, не блокируем вечно.
        for worker in self.workers:
            if worker.thread is not None and worker.thread.is_alive():
                if worker.current_request is not None:
                    worker.current_request.cancel()
                self._cancel_thread_io(worker)
                worker.thread.join(timeout=1.0)
                if worker.thread.is_alive():
                    logger.warning("SeekEngine.close(): SeekWorker %d не завершился", worker.id)

        # Освобождаем нативные контексты декодеров. Не трогаем декодеры
        # воркеров, чьи потоки ещё живы: закрытие контекста под работающим
        # decode_sample() — та же проблема параллельного доступа.
        for worker in self.workers:
            if worker.thread is None or not worker.thread.is_alive():
                worker.close_decoder()

    # ------------------------------------------------------------------
    # Координатор — единственный поток, управляющий состоянием пула
    # ------------------------------------------------------------------
    def _coordinator_loop(self):
        logger.info("SeekEngine: координатор запущен (воркеров: %d)", len(self.workers))
        while not self._closing.is_set():
            try:
                cmd = self._command_queue.get(timeout=_HEALTH_CHECK_INTERVAL_SEC)
            except queue.Empty:
                cmd = None

            self._drain_results()

            if cmd is None:
                if self._closing.is_set():
                    break
                self._check_watchdog()
                continue

            self._handle_command(cmd)
            self._check_watchdog()

        self._drain_results()
        logger.info("SeekEngine: координатор остановлен")

    @staticmethod
    def _fail_request(request: SeekRequest, reason: str):
        """
        Единая точка отказа: помечает запрос завершённым И уведомляет
        вызывающего через on_error.

        Появилась после регрессии: в нескольких местах запрос завершался
        молча (`_mark_done(...)` + `return` без вызова колбэка). Вызывающий
        при этом не получал НИЧЕГО — ни успеха, ни ошибки — и ждал до
        собственного watchdog-таймаута (в UI это выглядело как «плеер
        оживает через 25 секунд после перемотки»). Любой путь, на котором
        запрос не будет выполнен, обязан проходить через этот метод.
        """
        if request._done.is_set():
            return
        request._mark_done(False, reason)
        if request.on_error:
            try:
                request.on_error(reason)
            except Exception:
                logger.exception("Ошибка в on_error seek-запроса")

    def _handle_command(self, cmd: _SeekCommand):
        request = cmd.request
        if request.is_cancelled:
            self._fail_request(request, "Seek отменён")
            return

        with self._lock:
            if request.generation != self._generation:
                # Вытеснен более новым запросом. Уведомляем: вызывающий
                # должен снять состояние «идёт перемотка», иначе UI
                # останется заблокированным.
                self._fail_request(request, "Seek вытеснен более новым запросом")
                return

        worker = self._acquire_worker()
        if worker is None:
            logger.error("SeekEngine: нет доступных воркеров для seek(frame=%d)", cmd.frame_idx)
            self._fail_request(request, "Нет доступных воркеров для seek")
            return

        worker.buffer.clear()
        worker.current_request = request
        worker.current_frame_idx = cmd.frame_idx
        worker.task_start_ts = time.monotonic()
        worker.state = 'busy'

        t = threading.Thread(
            target=self._run_worker_task, args=(worker, cmd),
            daemon=True, name=f"SeekWorker-{worker.id}",
        )
        worker.thread = t
        t.start()

    def _acquire_worker(self) -> Optional[SeekWorker]:
        for w in self.workers:
            if w.state == 'free':
                return w

        busy = [w for w in self.workers if w.state == 'busy']
        if not busy:
            return None

        busy.sort(key=lambda w: w.task_start_ts)
        victim = busy[0]
        logger.info("SeekEngine: все воркеры заняты, вытесняем SeekWorker %d", victim.id)
        self._force_cancel_and_wait(victim)

        if victim.thread is None:
            victim.state = 'free'
            return victim

        return self._retire_stuck_worker(victim)

    def _force_cancel_and_wait(self, worker: SeekWorker,
                               hard_timeout: float = _ACQUIRE_CANCEL_HARD_TIMEOUT_SEC):
        """Блокирующее ожидание завершения потока воркера с эскалацией отмены."""
        if worker.current_request is not None and not worker.current_request._done.is_set():
            worker.current_request.cancel()

        if worker.thread is None:
            return

        deadline = time.monotonic() + hard_timeout
        while time.monotonic() < deadline:
            self._cancel_thread_io(worker)
            worker.thread.join(timeout=0.5)
            if not worker.thread.is_alive():
                worker.thread = None
                return

        logger.error(
            "SeekWorker %d: не удалось прервать чтение за %.0f сек "
            "(CancelSynchronousIo не помог) — воркер выводится из пула",
            worker.id, hard_timeout,
        )

    def _retire_stuck_worker(self, worker: SeekWorker) -> SeekWorker:
        """Выводит зависший воркер из пула навсегда, создаёт замену."""
        worker.state = 'stuck'
        idx = self.workers.index(worker)
        new_id = max(w.id for w in self.workers) + 1
        replacement = SeekWorker(new_id, self._decoder_factory)
        self.workers[idx] = replacement
        logger.error(
            "SeekWorker %d окончательно выведен из пула (завис), добавлен SeekWorker %d",
            worker.id, new_id,
        )
        return replacement

    def _drain_results(self):
        while True:
            try:
                result = self._result_queue.get_nowait()
            except queue.Empty:
                break
            self._finalize_worker(result)

    def _finalize_worker(self, result: _SeekResult):
        worker = result.worker

        if worker.thread is not None:
            worker.thread.join(timeout=2.0)
            if worker.thread.is_alive():
                # Воркер прислал результат, но поток ОС ещё жив: в пул его
                # не возвращаем (следующий _acquire_worker при
                # необходимости пройдёт эскалацию отмены). Но запрос обязан
                # получить ответ — иначе вызывающий зависнет до собственного
                # таймаута. Раньше здесь был молчаливый return.
                logger.warning(
                    "SeekWorker %d: результат получен, но поток ещё не завершился",
                    worker.id,
                )
                self._fail_request(result.request,
                                   "поток seek-воркера не завершился вовремя")
                return
            worker.thread = None

        request = result.request
        try:
            if request._done.is_set():
                return

            with self._lock:
                is_current = (request.generation == self._generation)

            if result.cancelled or not is_current:
                # Раньше здесь был _mark_done без вызова on_error, и
                # вызывающий не узнавал об отмене вообще — это и приводило
                # к «зависшей» перемотке в UI до watchdog-таймаута.
                reason = ("Seek отменён" if result.cancelled
                          else "Seek вытеснен более новым запросом")
                self._fail_request(request, reason)
                return

            if result.success:
                request._mark_done(True)
                if request.on_complete:
                    try:
                        request.on_complete(worker.buffer)
                    except Exception as e:
                        # Исключение ВНУТРИ колбэка (например, рассинхрон
                        # сигнатур в цепочке применения результата) раньше
                        # только логировалось: запрос считался успешным, но
                        # вызывающий не получал ни успеха, ни ошибки и висел
                        # до собственного watchdog-таймаута. Теперь сбой
                        # колбэка доводится до вызывающего немедленно.
                        logger.exception("Ошибка в on_complete seek-запроса")
                        if request.on_error:
                            try:
                                request.on_error(f"ошибка применения результата seek: {e}")
                            except Exception:
                                logger.exception("Ошибка в on_error seek-запроса")
            else:
                request._mark_done(False, result.error_msg)
                if request.on_error:
                    try:
                        request.on_error(result.error_msg or "seek failed")
                    except Exception:
                        logger.exception("Ошибка в on_error seek-запроса")
        finally:
            # Порядок важен: сначала колбэк выше синхронно забрал кадры в
            # буфер плеера (_on_seek_complete переносит их через
            # peek_first/advance), и только теперь буфер воркера можно
            # освободить. Раньше очистка была ТОЛЬКО в _handle_command перед
            # следующей задачей — то есть декодированные кадры оставались в
            # памяти воркера до его следующего использования. При трёх
            # воркерах это до трёх удерживаемых наборов кадров одновременно,
            # без всякой пользы: данные уже скопированы.
            #
            # Очищаем после того, как воркер отпущен потоком (thread is None
            # проверено выше), поэтому конкурентной записи в буфер быть не
            # может.
            try:
                worker.buffer.clear()
            except Exception:
                logger.exception("SeekWorker %d: ошибка очистки буфера", worker.id)
            worker.current_request = None
            worker.state = 'free'

    def _check_watchdog(self):
        now = time.monotonic()
        for worker in list(self.workers):
            if worker.state != 'busy' or worker.thread is None:
                continue
            if now - worker.task_start_ts <= _TASK_WATCHDOG_TIMEOUT_SEC:
                continue

            request = worker.current_request
            logger.warning(
                "SeekWorker %d: задача (frame=%s) выполняется дольше %.0f сек, "
                "принудительная отмена по watchdog",
                worker.id, worker.current_frame_idx, _TASK_WATCHDOG_TIMEOUT_SEC,
            )

            self._force_cancel_and_wait(worker)

            if worker.thread is not None and worker.thread.is_alive():
                self._retire_stuck_worker(worker)
            else:
                worker.thread = None
                worker.state = 'free'

            if request is not None and not request._done.is_set():
                request._mark_done(False, "Таймаут seek-операции (watchdog)")
                if request.on_error:
                    try:
                        request.on_error("Таймаут seek-операции (watchdog)")
                    except Exception:
                        logger.exception("Ошибка в on_error после watchdog-таймаута")
            # Буфер снятой по таймауту задачи тоже освобождаем — её кадры
            # никому не будут отданы. Но только если поток воркера уже
            # завершился: у зависшего (переведённого в 'stuck') воркера
            # буфер трогать нельзя, там может продолжаться запись.
            if worker.thread is None and worker.state != 'stuck':
                try:
                    worker.buffer.clear()
                except Exception:
                    logger.exception("SeekWorker %d: ошибка очистки буфера", worker.id)
            worker.current_request = None

    # ------------------------------------------------------------------
    # Работа воркера (выполняется на ОТДЕЛЬНОМ потоке SeekWorker-N)
    # ------------------------------------------------------------------
    def _run_worker_task(self, worker: SeekWorker, cmd: _SeekCommand):
        request = cmd.request
        success = False
        error_msg = None
        try:
            self._do_seek(worker, request, cmd.frame_idx)
            success = worker.buffer.count > 0
        except Exception as e:
            logger.exception("SeekWorker %d: ошибка seek(frame=%d)", worker.id, cmd.frame_idx)
            error_msg = str(e)
        finally:
            cancelled = request.is_cancelled
            self._result_queue.put(_SeekResult(worker, request, success, cancelled, error_msg))

    def _do_seek(self, worker: SeekWorker, request: SeekRequest, frame_idx: int):
        """
        Основная логика seek — пишет напрямую в worker.buffer и декодирует
        СВОИМ декодером воркера (см. докстринг SeekWorker): общий декодер
        конвейера здесь использовать нельзя.
        """
        decoder = worker.get_decoder()
        if decoder is None:
            raise RuntimeError(f"SeekWorker {worker.id}: декодер недоступен")
        window = self._lazy_index.open_window(frame_idx)
        if window is None or len(window.video_records) == 0:
            raise RuntimeError("Не удалось открыть окно индекса")

        idr_indices = window.idr_frames
        if len(idr_indices) == 0:
            raise RuntimeError("В окне не найдены IDR-кадры")

        local_target = max(0, min(frame_idx - window.window_start_frame,
                                  len(window.video_records) - 1))
        pos = np.searchsorted(idr_indices, local_target, side='right') - 1
        if pos < 0:
            pos = 0
        local_idr = int(idr_indices[pos])
        local_idr = max(0, min(local_idr, len(window.video_records) - 1))

        end_local = min(len(window.video_records) - 1,
                        max(local_idr + LOOKAHEAD_FRAMES, local_target))
        if end_local < local_idr:
            end_local = local_idr

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
            return

        # СОВМЕСТИМОСТЬ С НОВЫМ WinSequentialReader
        # Создаём ридер с параметрами по умолчанию (buffer_size=1MB,
        # read_timeout_ms=5000, use_no_buffering=False для SMB).
        reader = WinSequentialReader(
            self._lazy_index.mp4_path,
            buffer_size=1024 * 1024,
            read_timeout_ms=5000,
            use_no_buffering=False,
        )
        try:
            # Новый API: seek(offset) + read(size)
            reader.seek(start_offset)
            raw_data = reader.read(read_size)
        finally:
            reader.close()

        if request.is_cancelled or not raw_data:
            return

        first_decode_error = None
        buffer_full = False
        for i in range(local_idr, end_local + 1):
            if request.is_cancelled:
                return
            if i < 0 or i >= len(window.video_records):
                continue

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
            if rel_start + size > len(raw_data):
                size = len(raw_data) - rel_start
                if size <= 0:
                    continue

            sample = raw_data[rel_start:rel_start + size]
            try:
                filtered = decoder.filter_avcc(sample)
                if not filtered:
                    continue
                frames = decoder.decode_sample(filtered)
                for frame in frames:
                    if request.is_cancelled:
                        return
                    pts = video_frame_to_pts(window.window_start_frame + i)
                    if not worker.buffer.try_push(frame, pts):
                        buffer_full = True
                        break
                if buffer_full:
                    break
            except Exception as e:
                if first_decode_error is None:
                    first_decode_error = e
                continue

        if worker.buffer.count == 0 and not request.is_cancelled:
            msg = "Не удалось декодировать ни одного кадра при seek"
            if first_decode_error is not None:
                msg += f": {first_decode_error}"
            raise RuntimeError(msg)

    def _cancel_thread_io(self, worker: SeekWorker):
        """Прерывает блокирующий синхронный I/O в потоке воркера."""
        if worker.thread is None or not worker.thread.is_alive():
            return
        native_id = worker.thread.native_id
        if not native_id:
            return
        h_thread = OpenThread(THREAD_TERMINATE, False, native_id)
        if h_thread:
            try:
                CancelSynchronousIo(h_thread)
            finally:
                CloseHandle(h_thread)