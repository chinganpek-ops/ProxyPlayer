"""
lazy_index.py – оконный доступ к индексу для ProxyPlayer v1.
Загружает только необходимую часть video_records и audio_tracks
вокруг текущей позиции, используя memory-mapped .idx файл.
Все индексы внутри IndexWindow — локальные (от 0 до N-1).

ИЗМЕНЕНИЯ (правки продакшен-ревью, итерация 2 — скользящее окно):
- self._lock: threading.Lock -> threading.RLock (было сделано в предыдущей
  итерации, сохраняется: build_slid_window()/expand_window() вызывают друг
  друга/себя из уже захваченной блокировки того же потока).

- ПЕРЕСМОТРЕНО refresh_from_disk(): раньше метод сам расширял активное
  окно (вызывал expand_window()). Из-за этого при живом файле, растущем
  8+ часов, окно, задуманное как лёгкое ~5-минутное (default_window_seconds),
  только растягивалось бы вперёд и рано или поздно покрыло бы всю запись —
  весь смысл "ленивого окна" терялся, а _build_window() пересчитывал бы
  чанки над всё увеличивающимся срезом на каждый опрос.
  Теперь refresh_from_disk() ТОЛЬКО переоткрывает mmap и обновляет
  _video_records_full/_audio_tracks_full/total_frames — окно не трогает.
  Обновление активного окна — отдельная забота (см. ниже), вызывается по
  позиции воспроизведения, а не по факту прихода новых данных.
  Нужно вызывать этот метод регулярно (например, раз в 5-10 сек фоновым
  потоком в StreamController) — без этого is_near_window_end()/
  build_slid_window() будут работать со стухшими total_frames и не увидят
  кадры, уже дозаписанные IndexService в зеркало.

- НОВОЕ: is_near_window_end() / build_slid_window() / commit_window() —
  поддержка скользящего окна. Когда воспроизведение подходит к концу
  текущего окна (см. is_near_window_end), вызывающий (PlaybackEngine)
  строит новое окно через build_slid_window() — окно смещается ВПЕРЁД
  целиком (и start, и end), с перекрытием overlap_chunks относительно
  текущей позиции воспроизведения, а НЕ просто растягивается. Размер окна
  остаётся ограниченным (~default_window_seconds) сколько угодно долго —
  это и есть решение проблемы "окно, растущее без границ" из предыдущего
  абзаца. build_slid_window() НЕ подменяет self._window сразу — это
  сделано намеренно, чтобы вызывающий мог сначала переключить конвейер
  (ChunkPipeline.shift_window()) и только потом зафиксировать новое окно
  через commit_window(); так self._window и активное окно в конвейере
  никогда не рассинхронизируются, даже если между построением окна и его
  применением что-то пойдёт не так.
  Работает одинаково для архива и live: для архива total_frames стабилен,
  для live он подтягивается через refresh_from_disk() (см. выше).

  Общая для open_window()/build_slid_window() математика вынесена в
  _compute_window_bounds(), логика самого построения окна (_build_window)
  не менялась.

- expand_window() оставлен как есть (растягивает только конец) — теперь
  это не единственный, а вспомогательный инструмент для явного точечного
  использования (например, если понадобится специально продлить окно, не
  сдвигая начало); автоматический live-рост через него больше не идёт.

ИЗМЕНЕНИЯ (правки продакшен-ревью, итерация 3 — инкрементальное чтение):
- refresh_from_disk() переписан с "полный пересчёт каждый раз" на
  "дочитать только новый хвост и добавить к накопленному". Раньше метод
  вызывал idx_cache.remap_idx(), который заново сканировал ВЕСЬ .idx-файл
  с нуля на каждый опрос (раз в 5 сек, см. PlaybackEngine._tick_window_
  management) — для 8+ часовой растущей записи это становилось всё
  дороже по мере роста файла. Теперь используется
  idx_cache.remap_idx_incremental(), который сканирует только элементы
  u4-массива после self._next_scan_element (граница предыдущего скана) и
  возвращает только НОВЫЕ записи; они конкатенируются к уже накопленным
  self._all_193/_all_c9/_video_records_full, а не заменяют их целиком.
  Корректность опирается на строго append-only рост .idx (новые записи
  только дописываются в конец, старые байты никогда не меняются) — так
  описан ваш сценарий (сначала 12 видео-записей, затем аудио к ним,
  монотонно в конец файла).
ИЗМЕНЕНИЯ (правки продакшен-ревью, итерация 5 — оконное аудио):
- Сырые аудиозаписи 0xC9 больше НЕ ХРАНЯТСЯ. Раньше LazyIndex держал два
  представления одновременно: сырой _all_c9 (68 байт на запись) и
  производный _audio_tracks_full (28 байт), то есть 96 байт на запись.
  На восьмичасовой записи (около 11 млн аудиозаписей) это почти гигабайт
  в КАЖДОМ процессе плеера. Теперь сырая порция преобразуется в
  компактный вид сразу при поступлении и отпускается: остаётся только
  _audio_tracks_full. Экономия около 71%.
- _audio_tracks_full накапливается ИНКРЕМЕНТАЛЬНО (_append_audio).
  Раньше он сбрасывался в None на каждом обновлении индекса и
  пересобирался целиком при следующем построении окна — полный проход по
  всем аудиозаписям файла каждые несколько секунд. Теперь работа
  пропорциональна приросту. Корректность size2 на границе порций
  обеспечивается дозаполнением хвостовой записи каждой дорожки и
  проверена на эквивалентность обработке одной порцией.
- Срез аудио для окна (_slice_audio) использует двоичный поиск вместо
  булевой маски, когда PTS монотонны. Маска выделяла булев массив по
  числу ВСЕХ аудиозаписей при каждом построении окна.

- Добавлен _full_rescan() — аварийный откат на полный пересчёт индекса с
  нуля, если remap_idx_incremental() вдруг увидит файл короче, чем уже
  просканировано (признак пересоздания/усечения файла, а не чистого
  роста) — на такой случай инкрементальное состояние считается
  недостоверным и пересобирается заново.

ИЗМЕНЕНИЯ (правки продакшен-ревью, итерация 4 — растущий mdat):
- mdat_end больше не «замерзает» на значении, полученном при создании
  LazyIndex. Раньше он задавался один раз (get_real_size в
  StreamController._background_init) и никогда не обновлялся, хотя
  используется как правая граница при расчёте размера ПОСЛЕДНЕГО чанка
  каждого строящегося окна (_build_window) и как end_offset в SeekEngine.
  На файле, растущем 8+ часов, последний чанк любого нового окна считался
  от устаревшей границы — недочитанные чанки у live-края.
  Теперь mdat_end обновляется в _refresh_mdat_end(), вызываемом из
  refresh_from_disk()/_full_rescan() ТОЛЬКО когда в индексе реально
  появились новые записи. Привязка к событию роста индекса, а не к
  таймеру, корректна по порядку записи: данные попадают в mdat ДО того,
  как появятся ссылающиеся на них записи индекса, поэтому увиденные новые
  записи гарантируют, что соответствующие байты MP4 уже на диске.
  Обновление границ файла — задача LazyIndex как владельца индекса;
  PlaybackEngine остаётся чистым потребителем.
"""

import threading
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Callable
from collections import defaultdict

import numpy as np

from config.timebase import SAMPLES_PER_CHUNK, FRAMES_PER_CHUNK
from index.moov_builder import (
    DTYPE_193, DTYPE_AUDIO, SEGMENT_SIZE,
    _abs_offset, _filter_normal_records,
    get_idr_indices_from_mmap,
    build_audio_tracks,
    build_chunks_from_cached_offsets,
    build_audio_chunks_in_range,
    DEFAULT_TRACK_FILTER,
)
from index.idx_cache import remap_idx_incremental
from utils.utils import get_real_size

logger = logging.getLogger(__name__)

# Насколько чанков не хватает до конца окна, чтобы начать строить новое
# (скользящее) окно. 20 чанков * 12 кадров/чанк ≈ 240 кадров ≈ 9.6 сек при
# 25 fps — заведомо больше времени, нужного build_slid_window() на сборку
# (векторизованные numpy-операции над окном в несколько тысяч кадров).
DEFAULT_SLIDE_TRIGGER_MARGIN_CHUNKS = 20

# На сколько чанков НОВОЕ окно должно начинаться раньше текущей позиции
# воспроизведения. Должно с запасом перекрывать глубину очередей конвейера
# (RAW_QUEUE_SIZE=10 + VIDEO_QUEUE_SIZE=10 чанков "в полёте" на момент
# переключения) — иначе пакеты, уже лежащие в очередях со старыми
# глобальными PTS, будут отброшены новым pts_min сразу после свопа окна.
DEFAULT_SLIDE_OVERLAP_CHUNKS = 70


class IndexWindow:
    """
    Срез индекса, охватывающий диапазон кадров [start_frame, end_frame).
    Все индексы внутри окна — локальные (от 0 до N-1).
    """

    def __init__(
        self,
        video_records: np.ndarray,
        audio_tracks: np.ndarray,
        chunk_offsets: np.ndarray,
        chunk_sizes: np.ndarray,
        audio_chunks: List[Dict],
        start_frame: int,
        end_frame: int,
        idr_frames: np.ndarray,
        cached_offsets: np.ndarray = None,
    ):
        self.video_records = video_records          # локальные индексы 0..N-1
        self.audio_tracks = audio_tracks            # локальные
        self.chunk_offsets = chunk_offsets          # локальные
        self.chunk_sizes = chunk_sizes              # локальные
        self.audio_chunks = audio_chunks            # локальные индексы 0..M-1
        self.window_start_frame = start_frame       # глобальный номер первого кадра
        self.window_end_frame = end_frame           # глобальный номер последнего кадра + 1
        self.window_start_chunk = start_frame // FRAMES_PER_CHUNK  # глобальный номер первого чанка
        self.idr_frames = idr_frames                # локальные индексы IDR в окне
        self.cached_offsets = cached_offsets        # предвычисленные абсолютные смещения

    @property
    def total_chunks(self) -> int:
        """Количество чанков в окне (локальное)."""
        return len(self.chunk_offsets)

    def contains_frame(self, frame_idx: int) -> bool:
        """Проверяет, попадает ли глобальный кадр в окно."""
        return self.window_start_frame <= frame_idx < self.window_end_frame

    def global_to_local_chunk(self, global_chunk: int) -> int:
        """Переводит глобальный индекс чанка в локальный."""
        return global_chunk - self.window_start_chunk

    def local_to_global_frame(self, local_frame: int) -> int:
        """Переводит локальный индекс кадра в глобальный."""
        return local_frame + self.window_start_frame


class LazyIndex:
    """
    Управляет окном индекса, подгружая данные по мере необходимости.
    Использует mmap для доступа к полному .idx файлу.
    """

    def __init__(self, mirror_path: Path, mp4_path: Path, mdat_end: int):
        self.mirror_path = mirror_path
        self.mp4_path = mp4_path
        self.mdat_end = mdat_end

        # Инкрементальный скан с 0 при первом открытии эквивалентен
        # полному open_idx_mmap(), но дополнительно даёт next_scan_element —
        # границу, с которой refresh_from_disk() продолжит дочитывать
        # только новый хвост файла (см. idx_cache._scan_markers).
        all_193, all_c9, self._next_scan_element = remap_idx_incremental(
            mirror_path, start_element=0
        )
        self._all_193 = all_193

        # Полные video_records (глобальные индексы)
        self._video_records_full = _filter_normal_records(all_193)

        # Аудио: сырые записи 0xC9 сразу преобразуются в компактный
        # DTYPE_AUDIO и БОЛЬШЕ НЕ ХРАНЯТСЯ. См. _append_audio().
        self._audio_tracks_full: np.ndarray = np.empty(0, dtype=DTYPE_AUDIO)
        self._audio_pts_sorted: Optional[bool] = None
        self._append_audio(all_c9)
        # Сырой массив больше не нужен: всё, что из него требуется,
        # уже перенесено в _audio_tracks_full.
        self._all_c9 = None

        self._window: Optional[IndexWindow] = None
        # RLock, а не Lock: build_slid_window()/expand_window() реентерабельно
        # вызывают друг друга из того же потока, уже держащего блокировку.
        self._lock = threading.RLock()

        # Параметры окна
        self.default_window_seconds = 300.0  # 5 минут

    # ------------------------------------------------------------------
    def _append_audio(self, new_c9) -> int:
        """
        Преобразует новую порцию сырых записей 0xC9 в компактный
        DTYPE_AUDIO и присоединяет к накопленному массиву.

        Зачем это нужно. Раньше LazyIndex держал ОБА представления
        одновременно: сырые записи `_all_c9` (68 байт на запись) и
        производный `_audio_tracks_full` (28 байт). При этом производный
        сбрасывался в None на каждом обновлении индекса и пересобирался
        из полного сырого массива при следующем построении окна — то есть
        полный проход по всем аудиозаписям файла каждые несколько секунд.

        На восьмичасовой записи (порядка 11 млн аудиозаписей) это
        означало около 740 МБ сырых данных плюс 305 МБ производных в
        КАЖДОМ процессе плеера, и пересборку 305 МБ каждые 5 секунд.

        Теперь сырые записи живут ровно столько, сколько нужно на
        преобразование, и отпускаются. Хранится только компактное
        представление, и только приростом.

        Тонкость с size2. Длина второй части пакета вычисляется как
        расстояние до следующей записи ТОЙ ЖЕ дорожки. У последней записи
        каждой дорожки следующей ещё нет, поэтому size2 остаётся нулевым.
        Когда приходит новая порция, эти "хвостовые" записи нужно
        дозаполнить — иначе на каждой границе порций терялась бы вторая
        половина аудиопакета.

        Возвращает число добавленных записей.
        """
        if new_c9 is None or len(new_c9) == 0:
            return 0

        new_audio = build_audio_tracks(new_c9, track_filter=DEFAULT_TRACK_FILTER)
        if len(new_audio) == 0:
            return 0

        old_audio = self._audio_tracks_full
        if old_audio is None or len(old_audio) == 0:
            self._audio_tracks_full = new_audio
            self._audio_pts_sorted = None
            return len(new_audio)

        old_len = len(old_audio)
        merged = np.concatenate([old_audio, new_audio])

        # Дозаполняем size2 у последней записи каждой дорожки из прошлой
        # порции: теперь у неё появился следующий сосед.
        for track_id in DEFAULT_TRACK_FILTER:
            idx = np.where(merged["track"] == track_id)[0]
            if len(idx) < 2:
                continue
            prev_tail = idx[idx < old_len]
            next_head = idx[idx >= old_len]
            if not len(prev_tail) or not len(next_head):
                continue
            a = int(prev_tail[-1])
            b = int(next_head[0])
            gap = int(merged["abs_offset"][b]) - (
                int(merged["abs_offset"][a]) + int(merged["size1"][a])
            )
            merged["size2"][a] = max(0, gap)

        self._audio_tracks_full = merged
        self._audio_pts_sorted = None       # пересчитать при следующем срезе
        return len(new_audio)

    @property
    def audio_records_count(self) -> int:
        """Число накопленных аудиозаписей (для диагностики и телеметрии)."""
        return 0 if self._audio_tracks_full is None else len(self._audio_tracks_full)

    @property
    def window(self) -> Optional[IndexWindow]:
        return self._window

    @property
    def total_frames(self) -> int:
        """Полное количество видео-кадров в индексе на данный момент."""
        return len(self._video_records_full)

    # ------------------------------------------------------------------

    def get_window_bounds(self) -> Optional[dict]:
        """Границы активного окна без выдачи самого окна наружу."""
        with self._lock:
            w = self._window
            if w is None:
                return None
            return {
                "start_frame": w.window_start_frame,
                "end_frame": w.window_end_frame,
                "start_chunk": w.window_start_chunk,
                "total_chunks": w.total_chunks,
            }

    def open_window(
        self, center_frame: int, window_seconds: float = None
    ) -> IndexWindow:
        """
        Открывает окно вокруг center_frame.
        Если окно уже содержит этот кадр, возвращает текущее.
        Возвращает IndexWindow с локальными индексами.
        """
        with self._lock:
            if self._window and self._window.contains_frame(center_frame):
                logger.debug(f"Кадр {center_frame} уже в окне")
                return self._window

            start_frame, end_frame = self._compute_window_bounds(center_frame, window_seconds)
            self._window = self._build_window(start_frame, end_frame)
            logger.info(
                f"Окно открыто: кадры {start_frame}-{end_frame} "
                f"(чанки {self._window.window_start_chunk}-"
                f"{self._window.window_start_chunk + self._window.total_chunks})"
            )
            return self._window

    def move_window_async(
        self, center_frame: int, callback: Callable[[IndexWindow], None]
    ):
        """Асинхронно перемещает окно, вызывая callback по завершении."""
        def _move():
            win = self.open_window(center_frame)
            callback(win)

        thread = threading.Thread(target=_move, daemon=True)
        thread.start()

    def expand_window(self, new_end_frame: int) -> IndexWindow:
        """
        Растягивает конец текущего окна до new_end_frame, не трогая начало.
        Вспомогательный инструмент для точечного использования — см.
        комментарий в шапке файла про то, почему автоматический live-рост
        теперь идёт через build_slid_window()/commit_window(), а не через
        этот метод.
        """
        with self._lock:
            if self._window is None:
                return self.open_window(new_end_frame - 1000)

            old_end = self._window.window_end_frame
            if new_end_frame <= old_end:
                return self._window

            start_frame = self._window.window_start_frame
            end_frame = min(len(self._video_records_full), new_end_frame)
            self._window = self._build_window(start_frame, end_frame)
            logger.info(f"Окно расширено до кадра {end_frame}")
            return self._window

    # ------------------------------------------------------------------
    # Скользящее окно
    # ------------------------------------------------------------------
    def is_near_window_end(
        self, current_frame: int,
        trigger_margin_chunks: int = DEFAULT_SLIDE_TRIGGER_MARGIN_CHUNKS,
    ) -> bool:
        """
        Дешёвая проверка (без построения нового окна): пора ли начинать
        фоновую сборку следующего окна, потому что текущая позиция
        воспроизведения приближается к концу активного окна.

        ПРАВКА (продакшен-ревью, диагностика "GUI периодически блокируется"):
        эта проверка вызывается на КАЖДЫЙ кадр рендера из GUI-потока (через
        PlaybackEngine._tick_window_management). Раньше она брала обычную
        self._lock (RLock) — тот же лок, что refresh_from_disk() держит на
        всё время полного remap mmap + пересчёта массива записей в фоновом
        потоке (для многочасового файла это не мгновенно). Пока фоновый
        поток держал лок, GUI-поток блокировался здесь в ожидании — отсюда
        периодические подвисания интерфейса. Теперь лок берётся
        неблокирующе: если он сейчас занят фоновой операцией, просто
        пропускаем эту проверку до следующего кадра рендера (доли секунды
        при обычном fps) вместо ожидания.
        """
        if not self._lock.acquire(blocking=False):
            return False
        try:
            if self._window is None:
                return False
            remaining_frames = self._window.window_end_frame - current_frame
            return remaining_frames <= trigger_margin_chunks * FRAMES_PER_CHUNK
        finally:
            self._lock.release()

    def build_slid_window(
        self, current_frame: int,
        overlap_chunks: int = DEFAULT_SLIDE_OVERLAP_CHUNKS,
        window_seconds: float = None,
    ) -> Optional['IndexWindow']:
        """
        Строит НОВОЕ окно, сдвинутое вперёд относительно current_frame с
        перекрытием overlap_chunks в прошлое (см. константу
        DEFAULT_SLIDE_OVERLAP_CHUNKS — перекрытие должно с запасом
        покрывать глубину очередей конвейера).

        В отличие от expand_window(), НЕ заменяет self._window — только
        возвращает построенный IndexWindow. Замену должен явно выполнить
        вызывающий через commit_window() ПОСЛЕ того, как конвейер успешно
        переключится на новое окно (ChunkPipeline.shift_window()) — так
        self._window в LazyIndex и активное окно в ChunkPipeline не могут
        разойтись даже при ошибке на промежуточном шаге.

        Возвращает None, если сдвигать некуда (например, current_frame
        достаточно близко к total_frames, чтобы новое окно совпало со
        старым, или окно ещё не открыто).
        """
        with self._lock:
            if self._window is None:
                return None

            new_start = max(0, current_frame - overlap_chunks * FRAMES_PER_CHUNK)
            new_start = (new_start // FRAMES_PER_CHUNK) * FRAMES_PER_CHUNK

            if new_start <= self._window.window_start_frame:
                # Данных для сдвига пока не прибавилось (или пришли
                # раньше, чем ожидалось) — ждём следующего тика.
                return None

            _, new_end = self._compute_window_bounds(
                center_frame=None, window_seconds=window_seconds,
                explicit_start=new_start,
            )

            if new_end <= new_start:
                return None

            new_window = self._build_window(new_start, new_end)
            logger.info(
                f"Новое (скользящее) окно построено: кадры {new_start}-{new_end}, "
                f"текущая позиция={current_frame}"
            )
            return new_window

    def commit_window(self, window: 'IndexWindow'):
        """
        Фиксирует window как активное. Вызывается ПОСЛЕ того, как
        ChunkPipeline.shift_window(window) успешно отработал — см.
        комментарий в build_slid_window().
        """
        with self._lock:
            self._window = window

    def refresh_from_disk(self) -> bool:
        """
        Дочитывает НОВЫЙ хвост зеркала (после того как IndexService
        дозаписал данные на диск) и ДОБАВЛЯЕТ найденные записи к уже
        накопленным _all_193/_all_c9/_video_records_full/total_frames.

        ПРАВКА (продакшен-ревью, инкрементальное чтение): раньше здесь
        вызывался remap_idx(), который каждый раз пересканировал ВЕСЬ
        файл заново (полный np.where по всему u4-массиву) — для 8+
        часового растущего файла это означало, что каждый периодический
        опрос (раз в 5 сек) становился дороже по мере роста файла,
        пересчитывая давно обработанные данные. Для append-only роста
        (новые записи только дописываются в конец, старые байты не
        меняются) это лишняя работа: раз найденная и провалидированная
        запись остаётся валидной навсегда. Теперь используется
        remap_idx_incremental(), который сканирует только новый хвост
        (начиная с self._next_scan_element) и возвращает только НОВЫЕ
        записи — они конкатенируются к уже накопленным массивам, а не
        заменяют их.

        НЕ трогает активное окно (self._window) — это отдельная забота
        is_near_window_end()/build_slid_window()/commit_window(), которая
        реагирует на позицию воспроизведения, а не на сам факт прихода
        новых данных. Нужно вызывать регулярно (например, раз в 5-10 сек
        фоновым потоком) — без этого total_frames протухает, и
        build_slid_window() однажды упрётся в старые данные, даже если
        IndexService уже дозаписал новые кадры в зеркало на диске.

        Возвращает True, если появились новые видео- или аудио-записи,
        иначе False.
        """
        with self._lock:
            try:
                new_193, new_c9, next_scan_element = remap_idx_incremental(
                    self.mirror_path, self._next_scan_element
                )
            except Exception as e:
                logger.error(f"Не удалось обновить mmap зеркала {self.mirror_path}: {e}")
                return False

            if next_scan_element < self._next_scan_element:
                # Файл стал короче, чем мы уже успели просканировать —
                # это не чистый рост (пересоздание/усечение файла на
                # удалённой стороне). Инкрементальное состояние больше не
                # заслуживает доверия — аварийный откат на полный пересчёт.
                logger.warning(
                    "Зеркало %s короче, чем ожидалось (граница %d < %d) — "
                    "похоже на пересоздание файла, выполняю полный пересчёт индекса",
                    self.mirror_path, next_scan_element, self._next_scan_element,
                )
                return self._full_rescan()

            self._next_scan_element = next_scan_element

            if len(new_193) == 0 and len(new_c9) == 0:
                # Новых данных нет (или IndexService ещё не дописал
                # очередную порцию) — ничего не делаем.
                return False

            got_new_video = False
            if len(new_193) > 0:
                self._all_193 = np.concatenate([self._all_193, new_193])
                new_video_records = _filter_normal_records(new_193)
                if len(new_video_records) > 0:
                    self._video_records_full = np.concatenate(
                        [self._video_records_full, new_video_records]
                    )
                    got_new_video = True

            got_new_audio = 0
            if len(new_c9) > 0:
                # Преобразуем порцию в компактный вид и отпускаем сырые
                # записи. Раньше сырой массив дописывался в _all_c9, а
                # производный сбрасывался в None — то есть при следующем
                # построении окна пересобирался ПОЛНОСТЬЮ, проходом по
                # всем аудиозаписям файла. Теперь работа пропорциональна
                # приросту, а память — только компактному представлению.
                got_new_audio = self._append_audio(new_c9)

            # ПРАВКА (продакшен-ревью, растущий mdat): mdat_end задавался
            # ОДИН раз при создании LazyIndex (get_real_size в
            # StreamController._background_init) и больше никогда не
            # обновлялся. Он используется как правая граница при расчёте
            # размера ПОСЛЕДНЕГО чанка каждого строящегося окна
            # (_build_window -> build_chunks_from_cached_offsets) и как
            # end_offset в SeekEngine. На файле, растущем 8+ часов, это
            # означало, что последний чанк любого нового окна считался от
            # устаревшей границы — источник недочитанных чанков у live-края.
            #
            # Обновляем именно здесь, а не по таймеру: порядок записи
            # гарантирует корректность — данные пишутся в mdat ДО того, как
            # появятся ссылающиеся на них записи индекса. Раз мы только что
            # увидели новые записи, соответствующие байты в MP4 уже на диске.
            # Обновление границ файла — задача LazyIndex как владельца
            # индекса; PlaybackEngine остаётся только потребителем.
            self._refresh_mdat_end()

            return got_new_video or got_new_audio > 0

    def _full_rescan(self) -> bool:
        """
        Полный пересчёт индекса с нуля — аварийный откат, если
        инкрементальное состояние оказалось недостоверным (см.
        refresh_from_disk). Активное окно (self._window) не трогает —
        как и refresh_from_disk(), оставляет обновление окна на
        is_near_window_end()/build_slid_window()/commit_window().
        """
        try:
            all_193, all_c9, next_scan_element = remap_idx_incremental(
                self.mirror_path, start_element=0
            )
        except Exception as e:
            logger.error(f"Не удалось выполнить полный пересчёт индекса {self.mirror_path}: {e}")
            return False

        self._all_193 = all_193
        self._next_scan_element = next_scan_element
        self._video_records_full = _filter_normal_records(all_193)
        # Пересобираем аудио с нуля: инкрементальное состояние признано
        # недостоверным, поэтому накопленное отбрасывается целиком.
        self._audio_tracks_full = np.empty(0, dtype=DTYPE_AUDIO)
        self._audio_pts_sorted = None
        self._append_audio(all_c9)
        self._all_c9 = None
        self._refresh_mdat_end()
        return True

    def _refresh_mdat_end(self):
        """
        Переспрашивает актуальный размер MP4 (mdat_end) у файловой системы.
        Вызывается только когда в индексе реально появились новые записи —
        см. комментарий в refresh_from_disk().

        get_real_size() читает размер через GetFileSizeEx на свежем хэндле,
        минуя кэш SMB, поэтому на растущем сетевом файле возвращает
        актуальное значение, а не закэшированное клиентом.

        Ошибку не пробрасываем: неудача (0) означает лишь, что размер сейчас
        неизвестен — оставляем прежнее значение и попробуем на следующем
        обновлении индекса. Ронять из-за этого обновление уже полученных
        записей индекса не нужно.
        """
        try:
            new_end = get_real_size(str(self.mp4_path))
        except Exception as e:
            logger.warning("Не удалось обновить mdat_end для %s: %s", self.mp4_path, e)
            return

        if new_end <= 0:
            logger.warning("get_real_size вернул %s для %s — mdat_end оставлен прежним (%d)",
                           new_end, self.mp4_path, self.mdat_end)
            return

        if new_end > self.mdat_end:
            logger.debug("mdat_end обновлён: %d -> %d", self.mdat_end, new_end)
            self.mdat_end = new_end
        elif new_end < self.mdat_end:
            # Файл усечён/пересоздан — тот же класс аномалии, что и
            # укоротившееся зеркало в refresh_from_disk(). Принимаем новое
            # значение (продолжать считать по старой, большей границе
            # опаснее: чтение уйдёт за реальный конец файла).
            logger.warning("MP4 стал короче: mdat_end %d -> %d (%s)",
                           self.mdat_end, new_end, self.mp4_path)
            self.mdat_end = new_end

    def get_index_state(self) -> dict:
        """
        Состояние индекса: сколько кадров и аудиозаписей накоплено, где
        граница файла и докуда просканировано зеркало.

        Раньше эти величины читались из _video_records_full,
        _next_scan_element и _audio_tracks_full напрямую.
        """
        with self._lock:
            window = self._window
            return {
                "total_frames": len(self._video_records_full),
                "audio_records": (0 if self._audio_tracks_full is None
                                  else len(self._audio_tracks_full)),
                "mdat_end": self.mdat_end,
                "next_scan_element": self._next_scan_element,
                "window_start_frame": window.window_start_frame if window else None,
                "window_end_frame": window.window_end_frame if window else None,
                "window_total_chunks": window.total_chunks if window else 0,
            }

    def get_memory_usage(self) -> dict:
        """
        Фактический объём массивов индекса в байтах.

        Нужен для оценки расхода памяти при нескольких окнах: индекс —
        единственная величина, растущая линейно с длительностью записи.
        """
        def nbytes(arr):
            try:
                return int(arr.nbytes) if arr is not None else 0
            except Exception:
                return 0

        with self._lock:
            return {
                "video_records": nbytes(self._video_records_full),
                "audio_tracks": nbytes(self._audio_tracks_full),
                "raw_193": nbytes(self._all_193),
                "total": (nbytes(self._video_records_full)
                          + nbytes(self._audio_tracks_full)
                          + nbytes(self._all_193)),
            }

    def close(self):
        """Освобождает ресурсы."""
        self._window = None
        self._all_193 = None
        self._all_c9 = None

    # ------------------------------------------------------------------
    def _compute_window_bounds(
        self, center_frame: Optional[int], window_seconds: float = None,
        explicit_start: Optional[int] = None,
    ) -> Tuple[int, int]:
        """
        Общая математика для open_window()/build_slid_window(): считает
        (start_frame, end_frame) для окна заданного размера.

        Если explicit_start передан (используется build_slid_window),
        окно строится вперёд от этой точки, а не вокруг center_frame.
        Иначе окно центрируется вокруг center_frame — как и раньше в
        open_window().
        """
        if window_seconds is None:
            window_seconds = self.default_window_seconds

        total_frames = len(self._video_records_full)
        min_frames = 100 * FRAMES_PER_CHUNK

        if explicit_start is not None:
            start_frame = explicit_start
            full_window_frames = int(window_seconds * 48000) // 1920
            end_frame = min(total_frames, start_frame + full_window_frames)
            if end_frame - start_frame < min_frames:
                end_frame = min(total_frames, start_frame + min_frames)
            return start_frame, end_frame

        half_samples = int(window_seconds * 48000 / 2)
        half_frames = half_samples // 1920

        start_frame = max(0, center_frame - half_frames)
        start_frame = (start_frame // FRAMES_PER_CHUNK) * FRAMES_PER_CHUNK
        end_frame = min(total_frames, center_frame + half_frames)

        if end_frame - start_frame < min_frames:
            end_frame = min(total_frames, start_frame + min_frames)

        return start_frame, end_frame

    def _slice_audio(self, start_pts: int, end_pts: int) -> np.ndarray:
        """
        Возвращает аудиозаписи в диапазоне [start_pts, end_pts).

        Записи отсортированы по abs_offset, а не по pts, поэтому
        двоичный поиск применим не всегда: дорожки чередуются, и в местах
        дефектов разметки порядок по времени может нарушаться. Поэтому
        сначала проверяется монотонность (одно сравнение по массиву), и
        только при её наличии используется searchsorted; иначе —
        обычная маска.

        Разница существенна на длинных файлах: маска выделяет булев
        массив по числу ВСЕХ аудиозаписей (около 11 млн на восьми часах)
        при каждом построении окна, тогда как двоичный поиск даёт срез
        без выделения памяти.
        """
        audio = self._audio_tracks_full
        if audio is None or len(audio) == 0:
            return np.empty(0, dtype=DTYPE_AUDIO)

        pts = audio["pts"]
        if self._audio_pts_sorted is None:
            self._audio_pts_sorted = bool(np.all(pts[1:] >= pts[:-1])) if len(pts) > 1 else True

        if self._audio_pts_sorted:
            lo = int(np.searchsorted(pts, start_pts, side="left"))
            hi = int(np.searchsorted(pts, end_pts, side="left"))
            return audio[lo:hi]

        mask = (pts >= start_pts) & (pts < end_pts)
        return audio[mask]

    def _build_window(self, start_frame: int, end_frame: int) -> IndexWindow:
        """Строит IndexWindow с локальными индексами для указанного диапазона."""
        # 1. Видео-записи в диапазоне (срез — view, не копия)
        video_slice = self._video_records_full[start_frame:end_frame]

        # 2. Кэшируем абсолютные смещения ОДИН раз
        cached_offsets = _abs_offset(video_slice) if len(video_slice) > 0 else np.array([], dtype=np.uint64)

        # 3. Аудио-записи окна.
        #
        # Полный массив накапливается инкрементально в _append_audio(),
        # здесь берётся только срез по PTS-границам окна. Раньше на этом
        # месте при каждом построении окна пересобирался ВЕСЬ список
        # аудиотреков — он сбрасывался в None на каждом обновлении индекса.
        start_pts = (start_frame // FRAMES_PER_CHUNK) * SAMPLES_PER_CHUNK
        end_pts = ((end_frame + FRAMES_PER_CHUNK - 1) // FRAMES_PER_CHUNK) * SAMPLES_PER_CHUNK

        audio_slice = self._slice_audio(start_pts, end_pts)

        # 4. Чанки для видео в окне (используем кэшированные смещения)
        chunk_offsets, chunk_sizes = build_chunks_from_cached_offsets(
            video_slice, cached_offsets, self.mdat_end
        )

        # 5. Аудио-чанки (предвычисленный индекс, без searchsorted)
        start_chunk = start_frame // FRAMES_PER_CHUNK
        audio_chunks = build_audio_chunks_in_range(
            audio_slice, start_chunk, len(chunk_offsets)
        )

        # 6. IDR в окне (локальные индексы)
        idr_in_window = get_idr_indices_from_mmap(video_slice)

        return IndexWindow(
            video_records=video_slice,
            audio_tracks=audio_slice,
            chunk_offsets=chunk_offsets,
            chunk_sizes=chunk_sizes,
            audio_chunks=audio_chunks,
            start_frame=start_frame,
            end_frame=end_frame,
            idr_frames=idr_in_window,
            cached_offsets=cached_offsets,
        )
