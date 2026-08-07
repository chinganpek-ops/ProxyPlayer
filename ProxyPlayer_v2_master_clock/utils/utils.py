"""
utils.py – общие утилиты (production):
- get_real_size: актуальный размер файла через WinAPI (с обходом кэша SMB).
- find_ref_path: поиск .ref файла для Dalet MP4 (сначала <stem>.mp4.ref, затем <stem>.ref).
- resolve_id_to_mp4: поиск MP4-файла по ID в homedir (формат ID_YYYY-MM-DDTHH-MM-SS.mmm.mp4).
- _WorkerThread: фоновый поток, выполняющий переданную функцию.
- Вспомогательные WinAPI-обёртки для CancelSynchronousIo и др.
Подробное логирование.
"""

import ctypes
from ctypes import wintypes
import logging
import re
from pathlib import Path
from typing import Optional
from PyQt5.QtCore import QThread

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Windows API для определения размера файла (минуя кэш SMB)
# ---------------------------------------------------------------------------
GetFileSizeEx = ctypes.windll.kernel32.GetFileSizeEx
GetFileSizeEx.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_longlong)]
GetFileSizeEx.restype = wintypes.BOOL

# ---------------------------------------------------------------------------
# Windows API для прерывания синхронного I/O (CancelSynchronousIo)
# ---------------------------------------------------------------------------
CancelSynchronousIo = ctypes.windll.kernel32.CancelSynchronousIo
CancelSynchronousIo.argtypes = [wintypes.HANDLE]
CancelSynchronousIo.restype = wintypes.BOOL

OpenThread = ctypes.windll.kernel32.OpenThread
OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
OpenThread.restype = wintypes.HANDLE

CloseHandle = ctypes.windll.kernel32.CloseHandle
CloseHandle.argtypes = [wintypes.HANDLE]
CloseHandle.restype = wintypes.BOOL

# Константы
THREAD_SUSPEND_RESUME = 0x0002          # право приостанавливать/возобновлять поток
ERROR_OPERATION_ABORTED = 995


def get_real_size(path: str) -> int:
    """
    Возвращает актуальный размер файла через GetFileSizeEx (игнорирует кэш SMB).
    При ошибке возвращает 0 и логирует ошибку.
    """
    handle = ctypes.windll.kernel32.CreateFileW(
        path, 0x80000000, 0x00000001 | 0x00000002, None, 3, 0, None
    )
    if handle == -1:
        logger.error(f"Не удалось открыть файл для GetFileSizeEx: {path}")
        return 0
    size = ctypes.c_longlong(0)
    if GetFileSizeEx(handle, ctypes.byref(size)):
        ctypes.windll.kernel32.CloseHandle(handle)
        logger.debug(f"GetFileSizeEx({path}) = {size.value}")
        return size.value
    else:
        err = ctypes.GetLastError()
        logger.error(f"GetFileSizeEx вернул ошибку: {err} для {path}")
        ctypes.windll.kernel32.CloseHandle(handle)
        return 0


def find_ref_path(mp4_path: Path) -> Path:
    """
    Возвращает путь к .ref файлу для данного .mp4.
    Приоритет: <stem>.mp4.ref, затем <stem>.ref.
    """
    stem = mp4_path.stem
    parent = mp4_path.parent
    # Сначала ищем <stem>.mp4.ref
    ref = parent / f"{stem}.mp4.ref"
    if ref.exists():
        return ref
    # Запасной вариант <stem>.ref
    return parent / f"{stem}.ref"


def resolve_id_to_mp4(file_id: str, homedir: str) -> Optional[Path]:
    """
    Ищет MP4-файл по ID в homedir.
    Формат имени: ID_YYYY-MM-DDTHH-MM-SS.mmm.mp4 (например, 1808854_2026-07-03T13-06-17.606.mp4).
    Возвращает полный путь или None.
    """
    if not homedir:
        logger.error("homedir не задан, поиск по ID невозможен")
        return None
    home = Path(homedir)
    if not home.is_dir():
        logger.error(f"homedir не существует: {home}")
        return None
    if not re.match(r'^\d+$', file_id):
        logger.warning(f"ID '{file_id}' не является числом")
        return None

    # Шаблон: ID_ГГГГ-ММ-ДДTЧЧ-ММ-СС.миллисекунды.mp4
    pattern = re.compile(
        re.escape(file_id) + r'_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d{3}\.mp4$',
        re.IGNORECASE
    )

    # Быстрый поиск в корне homedir
    try:
        for entry in home.iterdir():
            if entry.is_file() and pattern.match(entry.name):
                logger.info(f"Найден файл по ID {file_id}: {entry}")
                return entry
    except PermissionError:
        logger.warning(f"Нет доступа к папке {home}")

    # Рекурсивный поиск
    try:
        for found in home.rglob(f"{file_id}_*.mp4"):
            if found.is_file() and pattern.match(found.name):
                logger.info(f"Найден файл (рекурсивно): {found}")
                return found
    except PermissionError:
        logger.warning(f"Ошибка доступа при рекурсивном поиске в {home}")

    return None


class _WorkerThread(QThread):
    """
    Фоновый поток, выполняющий переданную функцию.
    После завершения испускает сигнал finished.
    """
    def __init__(self, target, name=None):
        super().__init__()
        self._target = target
        self.setObjectName(name or "WorkerThread")

    def run(self):
        logger.debug(f"Поток {self.objectName()} запущен")
        try:
            self._target()
        except Exception as e:
            logger.exception(f"Исключение в потоке {self.objectName()}: {e}")
        finally:
            logger.debug(f"Поток {self.objectName()} завершён")