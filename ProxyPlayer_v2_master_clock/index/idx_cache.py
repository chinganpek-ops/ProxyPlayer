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
- НОВОЕ: remap_idx_incremental() + внутренняя _scan_markers(start_element).
  Раньше open_idx_mmap()/remap_idx() всегда сканировали весь файл заново
  (np.where по ВСЕМУ u4-массиву) — для растущего 8+ часового файла это
  означало, что каждый периодический опрос (LazyIndex.refresh_from_disk(),
  раз в 5 сек) пересканировал уже давно обработанные мегабайты, становясь
  дороже по мере роста файла. Для append-only роста (новые записи только
  дописываются в конец, старые байты не меняются — как в вашем случае)
  это лишняя работа: всё, что уже нашли и провалидировали, остаётся
  валидным навсегда. _scan_markers(start_element) теперь сканирует только
  window = arr_u4[start_element:] и возвращает next_start_element для
  следующего вызова (с запасом в 16 последних элементов — кандидат-маркер
  мог быть отброшен не по содержимому, а просто из-за нехватки данных на
  момент скана). open_idx_mmap()/remap_idx() поведение не изменили
  (полный скан с 0, тот же 2-tuple) — используются при первом открытии и
  как аварийный откат.
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

    reader.seek(start_read)
    raw_data = reader.read(to_read)
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
                reader = WinSequentialReader(idx_path)
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

# Сколько последних элементов u4-массива всегда пересканируются заново.
# Кандидат-маркер, найденный в последних 16 элементах предыдущего скана,
# мог быть отброшен не по содержимому, а просто из-за нехватки данных
# (записи 17 x uint32, и cand+17 могло выходить за длину массива на тот
# момент) — такие кандидаты должны быть пересмотрены, когда файл подрастёт.
_SCAN_RETRY_TAIL_ELEMENTS = 16


def _scan_markers(mirror_path: Path, start_element: int = 0) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Открывает mmap зеркала и ищет маркеры 0x193/0xC9 начиная с элемента
    start_element u4-массива (payload после сигнатуры начала данных).

    start_element=0 — полный скан (используется при первом открытии).
    start_element>0 — инкрементальный скан: возвращает ТОЛЬКО новые
    записи, найденные в добавленном хвосте с прошлого вызова. Корректен
    только для append-only файлов, где старые байты никогда не меняются
    (иначе см. LazyIndex._full_rescan — аварийный откат на полный скан).

    Возвращает (mm_193_new, mm_c9_new, next_start_element).
    next_start_element нужно сохранить и передать как start_element в
    следующий вызов, чтобы продолжить дочтение с этого места.
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

        total_elements = len(arr_u4)
        # Если файл вдруг оказался короче, чем то, что мы уже
        # просканировали (пересоздание/усечение вместо чистого роста) —
        # это не наш случай для инкремента; сигнализируем вызывающему
        # через next_start_element < start_element, он решит, что делать
        # (см. LazyIndex.refresh_from_disk / _full_rescan).
        search_start = max(0, min(start_element, total_elements))

        if search_start >= total_elements:
            mm_193 = np.empty(0, dtype=DTYPE_193)
            mm_c9 = np.empty(0, dtype=DTYPE_C9)
        else:
            # ВАЖНО: скан только по НОВОМУ хвосту (window), а не по всему
            # arr_u4 — именно это убирает O(n) полное пересканирование на
            # каждый вызов при растущем файле.
            window = arr_u4[search_start:]

            # --- 0x193 ---
            cand_193 = np.where(window == 0x193)[0]
            cand_193 = cand_193[cand_193 + 17 <= len(window)]
            valid_193 = cand_193[
                (window[cand_193 + 1] > 0) & (window[cand_193 + 2] < 256) & (window[cand_193 + 7] <= 31)
            ]
            if len(valid_193) == 0:
                mm_193 = np.empty(0, dtype=DTYPE_193)
            else:
                indices = valid_193[:, None] + np.arange(17)
                mm_193 = window[indices].view(DTYPE_193).ravel()  # fancy-индексация уже копирует

            # --- 0xC9 ---
            cand_c9 = np.where(window == 0xC9)[0]
            cand_c9 = cand_c9[cand_c9 + 17 <= len(window)]
            valid_c9 = cand_c9[window[cand_c9 + 1] > 0]
            if len(valid_c9) == 0:
                mm_c9 = np.empty(0, dtype=DTYPE_C9)
            else:
                indices = valid_c9[:, None] + np.arange(17)
                mm_c9 = window[indices].view(DTYPE_C9).ravel()

        with _mmap_cache_lock:
            _mmap_cache[mirror_path] = mm

        next_start_element = max(search_start, total_elements - _SCAN_RETRY_TAIL_ELEMENTS)

        logger.info(
            f"idx просканирован для {mirror_path.name} (с элемента {search_start}): "
            f"новых 193={len(mm_193)}, новых C9={len(mm_c9)}, всего элементов={total_elements}"
        )
        return mm_193, mm_c9, next_start_element

    except Exception:
        f.close()
        raise
    finally:
        f.close()


def open_idx_mmap(mirror_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Отображает локальное зеркало .idx в память через mmap и делает полный
    скан. Возвращает структурированные массивы (mm_193, mm_c9) для всех
    записей 0x193 и 0xC9. Используется при первом открытии — для
    последующих обновлений растущего файла см. remap_idx_incremental().
    """
    mm_193, mm_c9, _next = _scan_markers(mirror_path, start_element=0)
    return mm_193, mm_c9


def remap_idx(mirror_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Полный пересчёт индекса с нуля (переоткрывает mmap и сканирует весь
    файл заново). Дорогая операция для большого растущего файла — для
    обычного периодического обновления используйте remap_idx_incremental().
    Оставлен как аварийный откат (см. LazyIndex._full_rescan) и для любых
    сценариев, где нужен гарантированно полный пересчёт.
    """
    mm_193, mm_c9, _next = _scan_markers(mirror_path, start_element=0)
    return mm_193, mm_c9


def remap_idx_incremental(mirror_path: Path, start_element: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Дочитывает ТОЛЬКО новые записи, появившиеся в зеркале после
    start_element (номер элемента u4-массива, возвращённый предыдущим
    вызовом этой же функции или open_idx_mmap-эквивалента).

    Корректно только для append-only роста (старые байты не меняются,
    новые дописываются в конец) — именно так, как описан рост .idx в
    вашем случае: сначала 12 видео-записей, затем аудио к ним, и так
    далее, монотонно в конец файла.

    Возвращает (new_193, new_c9, next_start_element). Если
    next_start_element < start_element — файл стал короче, чем ожидалось
    (похоже на пересоздание/усечение, а не на чистый рост); вызывающий
    должен в этом случае откатиться на полный пересчёт (remap_idx).
    """
    return _scan_markers(mirror_path, start_element=start_element)


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
