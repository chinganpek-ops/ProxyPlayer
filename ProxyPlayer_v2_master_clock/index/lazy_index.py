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
from index.idx_cache import open_idx_mmap, remap_idx

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

        # mmap полного индекса (только для чтения)
        self._all_193, self._all_c9 = open_idx_mmap(mirror_path)

        # Полные video_records (глобальные индексы)
        self._video_records_full = _filter_normal_records(self._all_193)
        self._audio_tracks_full: Optional[np.ndarray] = None  # ленивая загрузка

        self._window: Optional[IndexWindow] = None
        # RLock, а не Lock: build_slid_window()/expand_window() реентерабельно
        # вызывают друг друга из того же потока, уже держащего блокировку.
        self._lock = threading.RLock()

        # Параметры окна
        self.default_window_seconds = 300.0  # 5 минут

    @property
    def window(self) -> Optional[IndexWindow]:
        return self._window

    @property
    def total_frames(self) -> int:
        """Полное количество видео-кадров в индексе на данный момент."""
        return len(self._video_records_full)

    # ------------------------------------------------------------------
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
        """
        with self._lock:
            if self._window is None:
                return False
            remaining_frames = self._window.window_end_frame - current_frame
            return remaining_frames <= trigger_margin_chunks * FRAMES_PER_CHUNK

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
        Переоткрывает mmap зеркала (после того как IndexService дозаписал
        новые данные на диск) и обновляет _video_records_full/
        _audio_tracks_full/total_frames.

        НЕ трогает активное окно (self._window) — это отдельная забота
        is_near_window_end()/build_slid_window()/commit_window(), которая
        реагирует на позицию воспроизведения, а не на сам факт прихода
        новых данных. Нужно вызывать регулярно (например, раз в 5-10 сек
        фоновым потоком) — без этого total_frames протухает, и
        build_slid_window() однажды упрётся в старые данные, даже если
        IndexService уже дозаписал новые кадры в зеркало на диске.

        Возвращает True, если появились новые видео-кадры, иначе False.
        """
        with self._lock:
            try:
                new_all_193, new_all_c9 = remap_idx(self.mirror_path)
            except Exception as e:
                logger.error(f"Не удалось обновить mmap зеркала {self.mirror_path}: {e}")
                return False

            new_video_records_full = _filter_normal_records(new_all_193)

            self._all_193 = new_all_193
            self._all_c9 = new_all_c9

            if len(new_video_records_full) <= len(self._video_records_full):
                # Новых кадров нет (или IndexService ещё не дописал
                # очередную порцию) — ничего не делаем.
                return False

            self._video_records_full = new_video_records_full
            # Полный список аудио-треков кэшируется лениво в _build_window();
            # инвалидируем, чтобы он был пересчитан с учётом новых C9-записей.
            self._audio_tracks_full = None
            return True

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

    def _build_window(self, start_frame: int, end_frame: int) -> IndexWindow:
        """Строит IndexWindow с локальными индексами для указанного диапазона."""
        # 1. Видео-записи в диапазоне (срез — view, не копия)
        video_slice = self._video_records_full[start_frame:end_frame]

        # 2. Кэшируем абсолютные смещения ОДИН раз
        cached_offsets = _abs_offset(video_slice) if len(video_slice) > 0 else np.array([], dtype=np.uint64)

        # 3. Аудио-записи (ленивая загрузка полных треков при первом обращении)
        if self._audio_tracks_full is None:
            self._audio_tracks_full = build_audio_tracks(
                self._all_c9, track_filter=DEFAULT_TRACK_FILTER
            )

        # Определяем PTS-границы для аудио в этом окне
        start_pts = (start_frame // FRAMES_PER_CHUNK) * SAMPLES_PER_CHUNK
        end_pts = ((end_frame + FRAMES_PER_CHUNK - 1) // FRAMES_PER_CHUNK) * SAMPLES_PER_CHUNK

        audio_mask = (
            (self._audio_tracks_full['pts'] >= start_pts)
            & (self._audio_tracks_full['pts'] < end_pts)
        )
        audio_slice = self._audio_tracks_full[audio_mask]

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
