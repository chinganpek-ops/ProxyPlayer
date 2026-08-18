"""
idx_cache.py – локальный кэш индекса (.idx) с поддержкой memory-mapped файлов.
Версия для ProxyPlayer v1 с архитектурой IndexService.

Основные изменения:
- prepare_mirror() теперь вызывается только IndexService (процессом-менеджером).
- Плееры используют open_idx_mmap() напрямую, получая путь к уже готовому зеркалу.
- Добавлена get_mirror_path() – возвращает путь к зеркалу без попытки докачки.

ИЗМЕНЕНИЯ (правки продакшен-ревью):
- _get_base_name(): имя зеркала теперь строится через SHA1-хэш полного пути,
  а не обрезкой строки до последних 50 символов. Раньше два разных .idx-файла
  с одинаковым хвостом пути (например, на разных шарах/серверах) могли
  схлопнуться в один и тот же файл зеркала и портить данные друг друга.
- mirror_ipc_name(): та же хэш-функция используется для имени локального
  IPC-канала (QLocalServer/QLocalSocket) между IndexService и плеерами —
  так плееру не нужно знать PID процесса IndexService заранее, оба процесса
  вычисляют одно и то же имя из пути к .idx.
- Логика открытия/докачки/mmap не менялась.
"""

import os
import hashlib
import mmap
import logging
import threading
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

from file_io.win_sequential_reader import WinSequentialReader
from index.moov_builder import DTYPE_193, DTYPE_C9

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Константы путей
# ------------------------------------------------------------------
CACHE_DIR_NAME = "ProxyPlayer"
CACHE_SUBDIR = "cache"
MIRROR_SUBDIR = "idx_mirror"

TAIL_SAFETY = 67          # перекрытие для целостности записей при инкрементальной докачке

# ------------------------------------------------------------------
# Блокировки и кэш mmap
# ------------------------------------------------------------------
_locks: dict[Path, threading.Lock] = {}
_locks_lock = threading.Lock()

_mmap_cache: dict[Path, mmap.mmap] = {}
_mmap_cache_lock = threading.Lock()


def _get_cache_dir() -> Path:
    if os.name == 'nt':
        base = Path(os.environ.get('APPDATA', Path.home() / 'AppData' / 'Roaming'))
    else:
        base = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
    cache_dir = base / CACHE_DIR_NAME / CACHE_SUBDIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _get_mirror_dir() -> Path:
    mirror_dir = _get_cache_dir() / MIRROR_SUBDIR
    mirror_dir.mkdir(parents=True, exist_ok=True)
    return mirror_dir


def _path_digest(idx_path: Path) -> str:
    """
    Единый источник правды для "уникального имени по пути" — используется и
    для имени файла зеркала, и для имени IPC-канала IndexService, чтобы оба
    процесса детерминированно приходили к одному и тому же идентификатору,
    зная только путь к .idx (без обмена PID или доп. согласований).
    """
    resolved = str(idx_path.resolve())
    return hashlib.sha1(resolved.encode('utf-8')).hexdigest()[:16]


def _get_base_name(idx_path: Path) -> str:
    """
    Уникальное имя зеркала по полному пути.

    Хэш от resolve()-пути гарантирует отсутствие коллизий между разными
    файлами (в отличие от прежней обрезки строки по последним 50 символам,
    где два .idx с длинным общим хвостом пути схлопывались в одно зеркало).
    Человекочитаемый stem оставлен только для удобства просмотра папки
    кэша глазами — на уникальность имени он не влияет.
    """
    digest = _path_digest(idx_path)
    readable = "".join(c if c.isalnum() or c in "._-" else "_" for c in idx_path.stem)[:40]
    return f"{readable}_{digest}"


def mirror_ipc_name(idx_path: Path) -> str:
    """
    Детерминированное имя локального IPC-канала (QLocalServer/QLocalSocket)
    для связки IndexService <-> плееры по конкретному .idx-файлу.
    Строится из того же хэша, что и имя зеркала, поэтому плееру не нужно
    знать PID процесса IndexService — оба процесса вычисляют одно и то же
    имя из пути к .idx.
    """
    return f"ProxyPlayerIndexService_{_path_digest(idx_path)}"


def _mirror_path(idx_path: Path) -> Path:
    """Единое зеркало для всех процессов (без PID)."""
    return _get_mirror_dir() / f"{_get_base_name(idx_path)}.idx"


def _get_lock(path: Path) -> threading.Lock:
    with _locks_lock:
        if path not in _locks:
            _locks[path] = threading.Lock()
        return _locks[path]


# ------------------------------------------------------------------
# Синхронизация зеркала (только для IndexService)
# ------------------------------------------------------------------
def _sync_mirror(reader: WinSequentialReader, mirror_path: Path, remote_size: int,
                 local_size: int, tail_data: bytes = b'') -> int:
    """Инкрементально докачивает недостающие байты в локальное зеркало."""
    if local_size >= remote_size:
        return local_size

    start_read = max(0, local_size - TAIL_SAFETY)
    to_read = remote_size - start_read
    logger.info(f"Докачка зеркала: с {start_read} байт, объём {to_read} байт")

    raw_data = reader.read_sequential(start_read, to_read)
    if not raw_data:
        logger.error("Не удалось прочитать данные для обновления зеркала")
        return local_size

    if tail_data:
        overlap = local_size - start_read
        if overlap > 0 and len(tail_data) >= overlap:
            raw_data = tail_data[-overlap:] + raw_data[overlap:]
        else:
            raw_data = tail_data + raw_data

    try:
        with open(mirror_path, 'ab' if local_size > 0 else 'wb') as f:
            if local_size == 0:
                f.write(raw_data)
            else:
                tail_len = TAIL_SAFETY if len(raw_data) > TAIL_SAFETY else 0
                if tail_len > 0:
                    f.write(raw_data[tail_len:])
                else:
                    f.write(raw_data)
        new_local_size = mirror_path.stat().st_size
        logger.info(f"Зеркало обновлено: было {local_size} байт, стало {new_local_size}")
        return new_local_size
    except OSError as e:
        logger.error(f"Ошибка записи зеркала {mirror_path}: {e}")
        return local_size


def prepare_mirror(idx_path: Path) -> Path:
    """
    Синхронизирует локальное зеркало с удалённым idx-файлом.
    Вызывается ТОЛЬКО IndexService. Плееры не должны вызывать эту функцию.
    """
    mirror_path = _mirror_path(idx_path)
    lock = _get_lock(mirror_path)
    with lock:
        try:
            cur_size = idx_path.stat().st_size
        except OSError:
            logger.warning(f"Не удалось получить stat для {idx_path}")
            return mirror_path

        local_size = mirror_path.stat().st_size if mirror_path.exists() else 0
        if local_size < cur_size:
            try:
                reader = WinSequentialReader(idx_path, rate_limit=0, overlapped=False)
                local_size = _sync_mirror(reader, mirror_path, cur_size, local_size)
                reader.close()
            except Exception as e:
                logger.error(f"Ошибка синхронизации зеркала: {e}")
        return mirror_path


def get_mirror_path(idx_path: Path) -> Path:
    """
    Возвращает путь к локальному зеркалу БЕЗ попытки докачки.
    Используется плеерами для открытия уже готового зеркала.
    """
    return _mirror_path(idx_path)


# ------------------------------------------------------------------
# mmap-функции (используются всеми)
# ------------------------------------------------------------------
def open_idx_mmap(mirror_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Отображает локальное зеркало .idx в память через mmap.
    Возвращает структурированные массивы (mm_193, mm_c9) для записей 0x193 и 0xC9.
    """
    # Закрываем предыдущий mmap для этого пути, если есть
    with _mmap_cache_lock:
        if mirror_path in _mmap_cache:
            old_mmap = _mmap_cache.pop(mirror_path)
            try:
                old_mmap.close()
            except Exception:
                pass

    f = open(mirror_path, 'rb')
    try:
        file_size = os.fstat(f.fileno()).st_size
        if file_size == 0:
            raise ValueError("Зеркало пустое")

        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        data_start = mm.find(b'\x93\x01\x00\x00')
        if data_start == -1:
            raise ValueError("Сигнатура 0x00000193 не найдена в зеркале")

        total_payload = file_size - data_start
        trim = total_payload % 4
        if trim:
            total_payload -= trim

        arr_u4 = np.ndarray(
            shape=(total_payload // 4,),
            dtype='<u4',
            buffer=mm,
            offset=data_start
        )

        # --- Построение mm_193 ---
        cand_193 = np.where(arr_u4 == 0x193)[0]
        cand_193 = cand_193[cand_193 + 17 <= len(arr_u4)]
        valid_193 = cand_193[(arr_u4[cand_193 + 1] > 0) & (arr_u4[cand_193 + 2] < 256) & (arr_u4[cand_193 + 7] <= 31)]

        if len(valid_193) == 0:
            mm_193 = np.empty(0, dtype=DTYPE_193)
        else:
            indices = valid_193[:, None] + np.arange(17)
            mm_193 = arr_u4[indices].copy().view(DTYPE_193).ravel()

        # --- Построение mm_c9 ---
        cand_c9 = np.where(arr_u4 == 0xC9)[0]
        cand_c9 = cand_c9[cand_c9 + 17 <= len(arr_u4)]
        valid_c9 = cand_c9[arr_u4[cand_c9 + 1] > 0]

        if len(valid_c9) == 0:
            mm_c9 = np.empty(0, dtype=DTYPE_C9)
        else:
            indices = valid_c9[:, None] + np.arange(17)
            mm_c9 = arr_u4[indices].copy().view(DTYPE_C9).ravel()

        with _mmap_cache_lock:
            _mmap_cache[mirror_path] = mm

        logger.info(f"mmap открыт для {mirror_path.name}: 193={len(mm_193)}, C9={len(mm_c9)}")
        return mm_193.copy(), mm_c9.copy()

    except Exception:
        f.close()
        raise
    finally:
        f.close()


def remap_idx(mirror_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Переоткрывает mmap для зеркала (используется при росте файла)."""
    with _mmap_cache_lock:
        if mirror_path in _mmap_cache:
            old_mmap = _mmap_cache.pop(mirror_path)
            try:
                old_mmap.close()
            except Exception:
                pass
    return open_idx_mmap(mirror_path)


# ------------------------------------------------------------------
# Очистка
# ------------------------------------------------------------------
def cleanup_cache(idx_path: Path):
    """Удаляет локальное зеркало для указанного индекса."""
    mirror_path = _mirror_path(idx_path)
    with _mmap_cache_lock:
        if mirror_path in _mmap_cache:
            try:
                _mmap_cache.pop(mirror_path).close()
            except Exception:
                pass

    with _get_lock(mirror_path):
        try:
            mirror_path.unlink(missing_ok=True)
            logger.debug(f"Удалён файл зеркала: {mirror_path}")
        except OSError as e:
            logger.warning(f"Не удалось удалить {mirror_path}: {e}")


def cleanup_all_cache():
    """Полная очистка всей папки кэша."""
    with _mmap_cache_lock:
        for mm in _mmap_cache.values():
            try:
                mm.close()
            except Exception:
                pass
        _mmap_cache.clear()

    cache_dir = _get_cache_dir()
    if cache_dir.exists():
        try:
            import shutil as _shutil
            _shutil.rmtree(cache_dir)
            logger.info("Вся папка кэша удалена")
        except Exception as e:
            logger.error(f"Не удалось удалить папку кэша {cache_dir}: {e}")
