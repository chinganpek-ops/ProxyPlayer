"""
win_sequential_reader.py – асинхронный ридер для PyAV через Windows API.

Финальная версия с безупречным кэшированием и полной статистикой.
Исправления:
- Кэш для NO_BUFFERING сохраняет полные выровненные данные.
- Добавлен счётчик cache_misses.
- Добавлен метод clear_cache().
- Расширены метрики производительности (cache_hit_rate, error_rate, timeout_rate, iops).
- Эвристика для определения паттерна доступа при кэшировании.

ПРАВКИ ПРОДАКШЕН-РЕВЬЮ (три исправления, подробности — в докстрингах методов):
1. close() теперь взводит событие отмены ДО захвата self._lock. Раньше сигнал
   подавался только внутри _close_resources(), уже под локом, который read()
   удерживает всё время блокирующего сетевого чтения — из-за этого close() из
   другого потока ждал ровно то, что должен был прервать, а готовый механизм
   отмены в _read_at() оставался недостижимым. Это и есть причина, по которой
   поток чтения не закрывался при закрытии окна плеера.
2. INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value вместо -1. Со старым
   значением проверка результата CreateFileW никогда не срабатывала, и
   неудачное открытие файла проходило незамеченным.
3. read() дочитывает частичные результаты в цикле. Один ReadFile по SMB штатно
   возвращает меньше запрошенного; раньше короткий результат отдавался наверх
   как есть, и вызывающий молча терял кадры.

Кэширование, статистика, выравнивание для NO_BUFFERING и логика _read_at()
не менялись.
"""

import time
import asyncio
import ctypes
import threading
from ctypes import wintypes
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Константы Windows API
# ---------------------------------------------------------------------------
GENERIC_READ = 0x80000000
FILE_SHARE_READ = 0x00000001
FILE_SHARE_WRITE = 0x00000002
OPEN_EXISTING = 3
FILE_FLAG_OVERLAPPED = 0x40000000
FILE_FLAG_NO_BUFFERING = 0x20000000
FILE_FLAG_RANDOM_ACCESS = 0x10000000
# ФИКС: CreateFileW.restype = wintypes.HANDLE — беззнаковый указатель, и при
# ошибке ctypes отдаёт 0xFFFFFFFFFFFFFFFF, а не -1. Со старым значением (-1)
# сравнение `self._handle == INVALID_HANDLE_VALUE` в _open() было ВСЕГДА
# ложным: неудачное открытие файла не детектировалось, исключение не
# бросалось, и невалидный хэндл уходил дальше в ReadFile — ошибка всплывала
# позже и не по адресу ("не читается файл" вместо "не удалось открыть").
INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value

ERROR_IO_PENDING = 997
ERROR_OPERATION_ABORTED = 995
ERROR_HANDLE_EOF = 38
ERROR_IO_INCOMPLETE = 996

WAIT_OBJECT_0 = 0
WAIT_TIMEOUT = 0x00000102

kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)


class OVERLAPPED(ctypes.Structure):
    _fields_ = [
        ("Internal", ctypes.c_size_t),
        ("InternalHigh", ctypes.c_size_t),
        ("Offset", wintypes.DWORD),
        ("OffsetHigh", wintypes.DWORD),
        ("hEvent", wintypes.HANDLE),
    ]


# Прототипы функций Windows API
CreateFileW = kernel32.CreateFileW
CreateFileW.argtypes = [
    wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
    wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
    wintypes.HANDLE
]
CreateFileW.restype = wintypes.HANDLE

ReadFile = kernel32.ReadFile
ReadFile.argtypes = [
    wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(OVERLAPPED)
]
ReadFile.restype = wintypes.BOOL

WaitForSingleObject = kernel32.WaitForSingleObject
WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
WaitForSingleObject.restype = wintypes.DWORD

WaitForMultipleObjects = kernel32.WaitForMultipleObjects
WaitForMultipleObjects.argtypes = [
    wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE),
    wintypes.BOOL, wintypes.DWORD
]
WaitForMultipleObjects.restype = wintypes.DWORD

CancelIoEx = kernel32.CancelIoEx
CancelIoEx.argtypes = [wintypes.HANDLE, ctypes.POINTER(OVERLAPPED)]
CancelIoEx.restype = wintypes.BOOL

GetOverlappedResult = kernel32.GetOverlappedResult
GetOverlappedResult.argtypes = [
    wintypes.HANDLE, ctypes.POINTER(OVERLAPPED),
    ctypes.POINTER(wintypes.DWORD), wintypes.BOOL
]
GetOverlappedResult.restype = wintypes.BOOL

CreateEventW = kernel32.CreateEventW
CreateEventW.argtypes = [
    wintypes.LPVOID, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR
]
CreateEventW.restype = wintypes.HANDLE

SetEvent = kernel32.SetEvent
SetEvent.argtypes = [wintypes.HANDLE]
SetEvent.restype = wintypes.BOOL

ResetEvent = kernel32.ResetEvent
ResetEvent.argtypes = [wintypes.HANDLE]
ResetEvent.restype = wintypes.BOOL

CloseHandle = kernel32.CloseHandle
CloseHandle.argtypes = [wintypes.HANDLE]
CloseHandle.restype = wintypes.BOOL

GetFileSizeEx = kernel32.GetFileSizeEx
GetFileSizeEx.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_longlong)]
GetFileSizeEx.restype = wintypes.BOOL

GetLastError = kernel32.GetLastError


class WinSequentialReader:
    """
    Файлоподобный асинхронный ридер для PyAV через Windows API.
    Поддерживает методы read(), readinto(), seek(), tell(), seekable(), close().
    Ведёт статистику чтения и имеет эффективный кэш для последовательного доступа.
    """

    def __init__(
        self,
        path: Path,
        buffer_size: int = 1024 * 1024,
        read_timeout_ms: int = 50000,
        use_no_buffering: bool = False,
        max_cache_size: int = 10 * 1024 * 1024,  # 10 МБ
    ):
        if buffer_size <= 0:
            raise ValueError("buffer_size должен быть положительным")
        if read_timeout_ms <= 0:
            raise ValueError("read_timeout_ms должен быть положительным")
        if max_cache_size <= 0:
            raise ValueError("max_cache_size должен быть положительным")

        self._path = str(path)
        self._buffer_size = buffer_size
        self._read_timeout_ms = read_timeout_ms
        self._use_no_buffering = use_no_buffering
        self._max_cache_size = max_cache_size

        self._handle = None
        self._overlapped = None
        self._buffer = None
        self._position = 0
        self._file_size = 0
        self._is_closed = False
        self._lock = threading.RLock()

        self._io_event = None
        self._cancel_event = None
        self._events = None

        # Кэш для последовательного чтения
        self._cache_offset = None
        self._cache_data = None
        self._last_read_offset = None  # Для эвристики паттерна доступа

        # Статистика
        self._stats = {
            'reads': 0,
            'bytes_read': 0,
            'timeouts': 0,
            'errors': 0,
            'total_read_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
        }

        self._open()

    def _open(self):
        """Открывает файл с асинхронным режимом и случайным доступом."""
        try:
            flags = FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS
            if self._use_no_buffering:
                flags |= FILE_FLAG_NO_BUFFERING

            self._handle = CreateFileW(
                self._path,
                GENERIC_READ,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                None,
                OPEN_EXISTING,
                flags,
                None
            )
            if self._handle == INVALID_HANDLE_VALUE:
                raise OSError(f"Не удалось открыть {self._path}: {GetLastError()}")

            self._io_event = CreateEventW(None, True, False, None)
            self._cancel_event = CreateEventW(None, True, False, None)

            if not self._io_event or not self._cancel_event:
                raise OSError("Не удалось создать события")

            self._events = (wintypes.HANDLE * 2)(self._io_event, self._cancel_event)

            self._overlapped = OVERLAPPED()
            self._overlapped.hEvent = self._io_event

            self._buffer = ctypes.create_string_buffer(self._buffer_size)

            size = ctypes.c_longlong(0)
            if not GetFileSizeEx(self._handle, ctypes.byref(size)):
                error = GetLastError()
                raise OSError(f"Не удалось получить размер файла: {error}")
            self._file_size = size.value

        except Exception:
            self._close_resources()
            raise

    def refresh_file_size(self) -> int:
        """
        Переспрашивает у ОС актуальный размер файла и обновляет self._file_size.

        Нужно для растущих файлов: _file_size определяется один раз в _open(),
        а ReaderStage держит один ридер на весь сеанс воспроизведения. На
        файле, растущем 8+ часов, read() клампит запрошенный размер по
        границе `self._file_size - self._position` — то есть по размеру на
        момент открытия. Дойдя до этой границы, read() начинает молча
        возвращать b'' (данные "закончились"), хотя физически файл давно
        вырос: чтение упирается в стену, чанки помечаются неудачными,
        воспроизведение встаёт.

        Вызывается по событию роста индекса (см. ChunkPipeline.notify_file_grew),
        а не по таймеру: порядок записи гарантирует, что к моменту появления
        новых записей в .idx соответствующие байты уже в mdat.

        Возвращает актуальный размер (или прежний, если запрос не удался —
        ошибку наверх не пробрасываем, попробуем на следующем событии).
        """
        with self._lock:
            if self._is_closed or not self._handle:
                return self._file_size

            size = ctypes.c_longlong(0)
            if not GetFileSizeEx(self._handle, ctypes.byref(size)):
                logger.warning(
                    "refresh_file_size: GetFileSizeEx не удался (%s), "
                    "оставлен прежний размер %d", GetLastError(), self._file_size,
                )
                return self._file_size

            new_size = size.value
            if new_size != self._file_size:
                logger.debug("Размер файла обновлён: %d -> %d", self._file_size, new_size)
                self._file_size = new_size
            return self._file_size

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        """Позиционирование для PyAV."""
        with self._lock:
            if whence == 0:
                self._position = offset
            elif whence == 1:
                self._position += offset
            elif whence == 2:
                self._position = self._file_size + offset
            self._position = max(0, min(self._position, self._file_size))
            return self._position

    def tell(self) -> int:
        with self._lock:
            return self._position

    def read(self, size: int = -1) -> bytes:
        """
        Синхронное чтение (для PyAV).

        ФИКС (молчаливая потеря данных при частичном чтении): раньше здесь был
        ОДИН вызов _read_at(), то есть один ReadFile. По SMB на больших блоках
        ядро штатно возвращает меньше запрошенного — read() отдавал короткий
        буфер, а вызывающий (DemuxerStage) молча отбрасывал кадры, не попавшие
        в отданные байты (проверка rel_start + size > len(raw_data)): ни ошибки,
        ни записи в лог, просто пропавшие кадры и звук. В старом ридере от этого
        защищал цикл в read_sequential() по 1 МБ, при переходе на новый API он
        потерялся.

        Теперь чтение идёт циклом до тех пор, пока не набран полный запрошенный
        размер либо пока _read_at() не вернёт 0 байт. Ноль трактуется как конец
        доступных данных (EOF/отмена/таймаут/ошибка — все они уже залогированы
        внутри _read_at); если при этом набрано меньше запрошенного, пишем
        предупреждение, чтобы недобор было видно в логах, а не только по
        косвенным симптомам.
        """
        with self._lock:
            if self._is_closed:
                return b''

            if size < 0:
                size = self._file_size - self._position
            size = min(size, self._file_size - self._position)

            if size <= 0:
                return b''

            requested = size
            start_time = time.perf_counter()

            chunks = []
            got = 0
            while got < requested:
                if self._is_closed:
                    break
                part = self._read_at(self._position + got, requested - got)
                if not part:
                    # 0 байт: EOF, отмена, таймаут или ошибка — причина уже
                    # залогирована внутри _read_at(). Прекращаем дочитывание.
                    break
                chunks.append(part)
                got += len(part)

            elapsed = time.perf_counter() - start_time

            result = chunks[0] if len(chunks) == 1 else b''.join(chunks)

            self._stats['reads'] += 1
            self._stats['bytes_read'] += len(result)
            self._stats['total_read_time'] += elapsed

            if result:
                self._position += len(result)

            if len(result) < requested:
                logger.warning(
                    "Недочитано: запрошено %d байт с позиции %d, получено %d",
                    requested, self._position - len(result), len(result),
                )

            return result

    def readinto(self, buffer) -> int:
        """Читает данные в предоставленный буфер (для PyAV)."""
        data = self.read(len(buffer))
        if data:
            buffer[:len(data)] = data
        return len(data)

    def _should_cache(self, offset: int, size: int) -> bool:
        """Эвристика: кэшируем только при последовательном доступе."""
        if self._last_read_offset is None:
            self._last_read_offset = offset
            return True

        # Последовательный доступ: offset >= предыдущего конца чтения
        is_sequential = offset >= self._last_read_offset
        self._last_read_offset = offset + size
        return is_sequential

    def _read_at(self, offset: int, size: int) -> bytes:
        """Чтение через overlapped I/O с таймаутом и отменой."""
        if self._is_closed or not self._handle:
            return b''

        # Проверяем кэш
        if (
            self._cache_offset is not None
            and self._cache_data is not None
            and offset >= self._cache_offset
            and offset + size <= self._cache_offset + len(self._cache_data)
        ):
            start = offset - self._cache_offset
            self._stats['cache_hits'] += 1
            return self._cache_data[start:start + size]

        # Промах кэша
        self._stats['cache_misses'] += 1

        # Выравнивание для NO_BUFFERING
        sector_size = 4096 if self._use_no_buffering else 1
        aligned_offset = (offset // sector_size) * sector_size
        aligned_size = ((size + sector_size - 1) // sector_size) * sector_size
        need_offset = offset - aligned_offset

        # Проверяем, что выровненный размер не превышает размер буфера
        if aligned_size > self._buffer_size:
            self._buffer = ctypes.create_string_buffer(aligned_size)

        self._overlapped.Offset = aligned_offset & 0xFFFFFFFF
        self._overlapped.OffsetHigh = (aligned_offset >> 32) & 0xFFFFFFFF

        ResetEvent(self._io_event)

        bytes_read = wintypes.DWORD(0)
        success = ReadFile(
            self._handle,
            self._buffer,
            aligned_size,
            ctypes.byref(bytes_read),
            ctypes.byref(self._overlapped)
        )

        if not success:
            error = GetLastError()
            if error == ERROR_HANDLE_EOF:
                return b''
            if error == ERROR_IO_INCOMPLETE:
                pass
            elif error != ERROR_IO_PENDING:
                logger.error(f"Ошибка ReadFile: {error}")
                self._stats['errors'] += 1
                return b''

            wait_result = WaitForMultipleObjects(
                2, self._events, False, self._read_timeout_ms
            )

            if wait_result == WAIT_OBJECT_0:
                pass
            elif wait_result == WAIT_OBJECT_0 + 1:
                self._cancel_and_wait(bytes_read)
                return b''
            elif wait_result == WAIT_TIMEOUT:
                self._cancel_and_wait(bytes_read)
                self._stats['timeouts'] += 1
                logger.warning(f"Таймаут чтения {self._read_timeout_ms} мс")
                return b''
            else:
                logger.error(f"Ошибка ожидания: {GetLastError()}")
                self._stats['errors'] += 1
                return b''

        if not GetOverlappedResult(
            self._handle, ctypes.byref(self._overlapped),
            ctypes.byref(bytes_read), False
        ):
            error = GetLastError()
            if error == ERROR_HANDLE_EOF:
                return b''
            if error != ERROR_OPERATION_ABORTED:
                logger.error(f"Ошибка GetOverlappedResult: {error}")
                self._stats['errors'] += 1
            return b''

        if bytes_read.value == 0:
            return b''

        # Полные выровненные данные
        raw_data = bytes(self._buffer.raw[:bytes_read.value])

        # Кэшируем полные выровненные данные (если последовательный доступ)
        if (
            len(raw_data) <= self._max_cache_size
            and self._should_cache(offset, size)
        ):
            self._cache_offset = aligned_offset
            self._cache_data = raw_data

        # Возвращаем данные с учётом need_offset
        if self._use_no_buffering and need_offset > 0:
            end = min(need_offset + size, len(raw_data))
            return raw_data[need_offset:end]
        return raw_data

    def _cancel_and_wait(self, bytes_read):
        """Отменяет операцию и ждёт её завершения без блокировки."""
        CancelIoEx(self._handle, ctypes.byref(self._overlapped))
        for _ in range(20):
            if GetOverlappedResult(
                self._handle, ctypes.byref(self._overlapped),
                ctypes.byref(bytes_read), False
            ):
                break
            error = GetLastError()
            if error != ERROR_IO_INCOMPLETE:
                break
            time.sleep(0.01)

    def get_stats(self) -> dict:
        """Возвращает статистику чтения."""
        with self._lock:
            return self._stats.copy()

    def reset_stats(self):
        """Сбрасывает статистику чтения."""
        with self._lock:
            self._stats = {
                'reads': 0,
                'bytes_read': 0,
                'timeouts': 0,
                'errors': 0,
                'total_read_time': 0.0,
                'cache_hits': 0,
                'cache_misses': 0,
            }

    def clear_cache(self):
        """Очищает кэш последовательного чтения."""
        with self._lock:
            self._cache_offset = None
            self._cache_data = None
            self._last_read_offset = None

    def get_performance_metrics(self) -> dict:
        """Возвращает расширенные метрики производительности."""
        stats = self.get_stats()

        if stats['reads'] > 0:
            stats['avg_read_size'] = stats['bytes_read'] / stats['reads']
            stats['avg_read_time_ms'] = (stats['total_read_time'] / stats['reads']) * 1000

            if stats['total_read_time'] > 0:
                stats['throughput_mbps'] = (
                    stats['bytes_read'] / stats['total_read_time'] / 1024 / 1024
                )
                stats['iops'] = stats['reads'] / stats['total_read_time']

            stats['error_rate'] = stats['errors'] / stats['reads']
            stats['timeout_rate'] = stats['timeouts'] / stats['reads']

        total_cache_accesses = stats.get('cache_hits', 0) + stats.get('cache_misses', 0)
        if total_cache_accesses > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / total_cache_accesses

        return stats

    def close(self):
        """
        Безопасное закрытие с отменой всех операций.

        ФИКС (корректное закрытие потоков чтения): раньше SetEvent(_cancel_event)
        вызывался только внутри _close_resources(), то есть УЖЕ ПОСЛЕ захвата
        self._lock. Но read() держит этот же RLock всё время блокирующего
        сетевого чтения — значит close() из другого потока вставал в очередь за
        локом и ждал ровно того, что должен был прервать. Механизм отмены в
        _read_at() (WaitForMultipleObjects по [io_event, cancel_event] с
        последующим CancelIoEx) существовал, но был недостижим: сигнал не мог
        быть подан, пока чтение не отпустит лок само.

        Теперь событие отмены взводится ПЕРВЫМ делом, без лока: висящий read()
        немедленно просыпается на WaitForMultipleObjects, вызывает CancelIoEx,
        возвращает b'' и отпускает лок — после чего close() спокойно доводит
        уборку ресурсов под локом, как и раньше.
        """
        # Ранний сигнал отмены — до любых блокировок (см. докстринг).
        # Локальная копия: _close_resources() в другом потоке может обнулить
        # поле между проверкой и использованием.
        cancel_event = self._cancel_event
        if cancel_event:
            SetEvent(cancel_event)

        with self._lock:
            if self._is_closed:
                return
            self._is_closed = True
            self._close_resources()

    def _close_resources(self):
        """Внутреннее освобождение ресурсов (без блокировки)."""
        if self._cancel_event:
            SetEvent(self._cancel_event)

        if self._handle and self._handle != INVALID_HANDLE_VALUE:
            if self._overlapped:
                CancelIoEx(self._handle, ctypes.byref(self._overlapped))
                bytes_read = wintypes.DWORD(0)
                GetOverlappedResult(
                    self._handle, ctypes.byref(self._overlapped),
                    ctypes.byref(bytes_read), True
                )
            CloseHandle(self._handle)
            self._handle = None

        if self._io_event:
            CloseHandle(self._io_event)
            self._io_event = None

        if self._cancel_event:
            CloseHandle(self._cancel_event)
            self._cancel_event = None

        self._events = None
        self._overlapped = None
        self._buffer = None
        self._cache_offset = None
        self._cache_data = None
        self._last_read_offset = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class AsyncWinSequentialReader(WinSequentialReader):
    """Асинхронная обёртка для использования с asyncio."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await asyncio.to_thread(self.close)

    async def aread(self, size: int = -1) -> bytes:
        return await asyncio.to_thread(self.read, size)

    async def aseek(self, offset: int, whence: int = 0) -> int:
        return await asyncio.to_thread(self.seek, offset, whence)

    async def atell(self) -> int:
        return await asyncio.to_thread(self.tell)