"""
win_sequential_reader.py – низкоуровневое чтение файлов через Windows API
с флагом FILE_FLAG_SEQUENTIAL_SCAN для оптимизации последовательного доступа.
Поддерживает синхронный и асинхронный (overlapped) режимы.
Добавлена поддержка отмены операции чтения через threading.Event.
"""

import time
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
FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
FILE_FLAG_OVERLAPPED = 0x40000000
INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value

WAIT_TIMEOUT = 0x00000102
WAIT_OBJECT_0 = 0x00000000
INFINITE = 0xFFFFFFFF

kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

CreateFileW = kernel32.CreateFileW
CreateFileW.argtypes = [
    wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
    wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
    wintypes.HANDLE
]
CreateFileW.restype = wintypes.HANDLE

class OVERLAPPED(ctypes.Structure):
    _fields_ = [
        ("Internal", wintypes.ULONG),
        ("InternalHigh", wintypes.ULONG),
        ("Offset", wintypes.DWORD),
        ("OffsetHigh", wintypes.DWORD),
        ("hEvent", wintypes.HANDLE),
    ]

ReadFile = kernel32.ReadFile
ReadFile.argtypes = [
    wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(OVERLAPPED)
]
ReadFile.restype = wintypes.BOOL

SetFilePointerEx = kernel32.SetFilePointerEx
SetFilePointerEx.argtypes = [
    wintypes.HANDLE, ctypes.c_longlong, wintypes.LPVOID, wintypes.DWORD
]
SetFilePointerEx.restype = wintypes.BOOL

CloseHandle = kernel32.CloseHandle
CloseHandle.argtypes = [wintypes.HANDLE]
CloseHandle.restype = wintypes.BOOL

GetLastError = kernel32.GetLastError

CreateEventW = kernel32.CreateEventW
CreateEventW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
CreateEventW.restype = wintypes.HANDLE

WaitForSingleObject = kernel32.WaitForSingleObject
WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
WaitForSingleObject.restype = wintypes.DWORD

CancelIo = kernel32.CancelIo
CancelIo.argtypes = [wintypes.HANDLE]
CancelIo.restype = wintypes.BOOL

GetOverlappedResult = kernel32.GetOverlappedResult
GetOverlappedResult.argtypes = [
    wintypes.HANDLE, ctypes.POINTER(OVERLAPPED), ctypes.POINTER(wintypes.DWORD), wintypes.BOOL
]
GetOverlappedResult.restype = wintypes.BOOL

ERROR_IO_PENDING = 997


class WinSequentialReader:
    """
    Чтение файла через Windows API с флагом FILE_FLAG_SEQUENTIAL_SCAN.
    Параметры:
        path: путь к файлу
        read_timeout_ms: таймаут асинхронного чтения
        overlapped: использовать ли асинхронный ввод-вывод (по умолчанию False)
    Поддержка отмены через set_cancel_event()/clear_cancel_event().
    """

    def __init__(
        self,
        path: Path,
        rate_limit: int = 0,          # оставлен для обратной совместимости
        read_timeout_ms: int = 3000,
        overlapped: bool = False,
    ):
        self._path = str(path)
        self._handle = None
        self._read_timeout_ms = read_timeout_ms
        self._overlapped = overlapped
        self._cancel_event = threading.Event()  # событие отмены
        self._open()

    def _open(self):
        """Открывает файл с оптимизированным для последовательного чтения флагом."""
        if self._handle:
            self.close()
        flags = FILE_FLAG_SEQUENTIAL_SCAN
        if self._overlapped:
            flags |= FILE_FLAG_OVERLAPPED
        logger.debug(f"Открытие файла: {self._path}, overlapped={self._overlapped}")
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
            err = GetLastError()
            logger.error(f"Не удалось открыть файл {self._path}: код ошибки {err}")
            raise OSError(f"Не удалось открыть файл {self._path}: код ошибки {err}")
        logger.debug(f"Файл успешно открыт, HANDLE={self._handle}")

    def set_cancel_event(self, event: threading.Event):
        """Устанавливает событие отмены. Чтение будет прервано при его активации."""
        self._cancel_event = event

    def clear_cancel_event(self):
        """Сбрасывает событие отмены."""
        self._cancel_event.clear()

    def _create_overlapped(self, offset: int) -> OVERLAPPED:
        """Создаёт структуру OVERLAPPED с событием для асинхронного чтения."""
        hEvent = CreateEventW(None, True, False, None)
        ov = OVERLAPPED()
        ov.Offset = offset & 0xFFFFFFFF
        ov.OffsetHigh = (offset >> 32) & 0xFFFFFFFF
        ov.hEvent = hEvent
        return ov

    def read(self, size: int, offset: int) -> bytes:
        """
        Читает size байт с позиции offset.
        При overlapped=True выполняется асинхронно, иначе синхронно.
        """
        if self._handle is None or self._handle == INVALID_HANDLE_VALUE:
            logger.error("Попытка чтения с невалидным HANDLE")
            return b''
        size = int(size)
        buf = ctypes.create_string_buffer(size)
        bytes_read = wintypes.DWORD(0)

        if self._overlapped:
            ov = self._create_overlapped(offset)
            try:
                success = ReadFile(self._handle, buf, size, ctypes.byref(bytes_read), ctypes.byref(ov))
                if not success:
                    last_err = GetLastError()
                    if last_err != ERROR_IO_PENDING:
                        logger.error(f"Ошибка асинхронного чтения: код {last_err}")
                        return b''
                    # Ожидание завершения с учётом отмены
                    while True:
                        if self._cancel_event.is_set():
                            CancelIo(self._handle)
                            return b''
                        wait_result = WaitForSingleObject(ov.hEvent, 100)  # 100 мс
                        if wait_result == WAIT_TIMEOUT:
                            continue
                        elif wait_result == WAIT_OBJECT_0:
                            break
                        else:
                            logger.error(f"Ошибка ожидания события: {GetLastError()}")
                            return b''
                    if not GetOverlappedResult(self._handle, ctypes.byref(ov), ctypes.byref(bytes_read), False):
                        logger.error(f"Ошибка получения результата overlapped: {GetLastError()}")
                        return b''
                logger.debug(f"Прочитано {bytes_read.value} байт асинхронно с {offset}")
                return buf.raw[:bytes_read.value]
            finally:
                if ov.hEvent:
                    CloseHandle(ov.hEvent)
        else:
            # Синхронное чтение
            self.seek(offset)
            # Проверка отмены перед чтением
            if self._cancel_event.is_set():
                return b''
            if not ReadFile(self._handle, buf, size, ctypes.byref(bytes_read), None):
                err = GetLastError()
                logger.error(f"Ошибка синхронного чтения: код {err}")
                raise OSError(f"Ошибка синхронного чтения: код ошибки {err}")
            logger.debug(f"Прочитано {bytes_read.value} байт синхронно с {offset}")
            return buf.raw[:bytes_read.value]

    def seek(self, offset: int):
        """Позиционирует файловый указатель на offset (только для синхронного режима)."""
        li = ctypes.c_longlong(int(offset))
        if not SetFilePointerEx(self._handle, li, None, 0):
            err = GetLastError()
            logger.error(f"Ошибка позиционирования: код {err}")
            raise OSError(f"Ошибка позиционирования: код ошибки {err}")
        logger.debug(f"Указатель файла установлен на {offset}")

    def read_sequential(self, offset: int, size: int) -> bytes:
        """
        Читает size байт последовательно, разбивая на блоки.
        Поддерживает отмену через cancel_event: если событие установлено,
        чтение немедленно прекращается и возвращает уже прочитанные данные.
        """
        chunk_size = 1024 * 1024
        data = bytearray()
        remaining = int(size)
        current_offset = int(offset)
        max_retries = 3

        while remaining > 0:
            # Проверка отмены в начале каждой итерации
            if self._cancel_event.is_set():
                logger.debug("Чтение прервано по cancel_event")
                break

            to_read = int(min(chunk_size, remaining))
            block = b''
            for attempt in range(max_retries):
                if self._cancel_event.is_set():
                    break
                try:
                    block = self.read(to_read, current_offset)
                    if block:
                        break
                    else:
                        remaining = 0
                        break
                except OSError as e:
                    logger.warning(f"Попытка {attempt+1}/{max_retries} чтения не удалась: {e}")
                    if attempt < max_retries - 1:
                        # Переоткрываем файл
                        try:
                            self.close()
                        except Exception:
                            pass
                        try:
                            self._open()
                        except Exception as open_err:
                            logger.error(f"Не удалось переоткрыть файл: {open_err}")
                            return bytes(data)
                        time.sleep(0.5)
                    else:
                        logger.error("Все попытки чтения исчерпаны")
                        return bytes(data)
            if not block:
                break

            data.extend(block)
            bytes_read_now = len(block)
            current_offset += bytes_read_now
            remaining -= bytes_read_now
            if bytes_read_now < to_read:
                remaining = 0
        return bytes(data)

    def close(self):
        """Закрывает HANDLE и освобождает ресурсы."""
        if self._handle and self._handle != INVALID_HANDLE_VALUE:
            logger.debug(f"Закрытие HANDLE {self._handle}")
            CloseHandle(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()