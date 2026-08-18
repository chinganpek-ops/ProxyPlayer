"""
playback_engine.py – движок воспроизведения с тройным буфером и фильтрацией старых кадров.

Исправления (исходные):
- Тройной буфер: display_buffer, fill_buffer, free_buffer.
- При seek конвейер НЕ останавливается, только перенаправляется в новый буфер.
- Звук не прерывается, отклик быстрый.
- Эталонный PTS устанавливается ДО переключения буферов, чтобы старые пакеты
  отбрасывались ещё на этапе вставки.

ИЗМЕНЕНИЯ (правки продакшен-ревью — скользящее окно для растущих файлов):
- Конструктор принимает опциональный lazy_index: Optional[LazyIndex] (плюс
  set_lazy_index() для внедрения после создания, если конструктору его не
  передали). Без lazy_index класс работает ровно как раньше — новая логика
  ниже полностью гейтится проверкой `if self._lazy_index is None: return`.

- НОВОЕ: _tick_window_management() — вызывается из get_display_frame() на
  каждый кадр рендера (сама проверка дешёвая, тяжёлая работа всегда уходит
  в отдельный поток) и делает две независимые вещи:

  1) Держит метаданные LazyIndex свежими: раз в _GROWTH_REFRESH_INTERVAL_SEC
     (5 сек) в фоновом потоке вызывает lazy_index.refresh_from_disk().
     Работает всегда, независимо от режима воспроизведения (seek/JKL/pause) —
     это просто синхронизация с диском, она не трогает активное окно и
     ничего не ломает, даже если вызвана посреди seek. Без этого вызова
     LazyIndex.total_frames/_video_records_full протухнут, и скользящее
     окно (см. ниже) будет упираться в старые данные, даже если
     IndexService уже дозаписал новые кадры в зеркало на диске — то есть
     это прямой ответ на вопрос "чтобы были данные из чего строить новое
     окно". Каждый успешный refresh обновляет self.total_frames — это
     позволяет JKL/UI увидеть выросшую границу файла ещё до фактического
     сдвига окна.

  2) Отслеживает приближение текущей позиции к концу активного окна
     (LazyIndex.is_near_window_end) и, если конвейер в обычном режиме
     воспроизведения (не во время активного seek и не во время JKL-
     перемотки — там позиция скачет нелинейно, и "приближение к концу
     окна" не то же самое событие), запускает в фоновом потоке
     трёхшаговый сдвиг окна:
       a) new_window = lazy_index.build_slid_window(current_frame)
       b) pipeline.shift_window(new_window)   — переключает конвейер
       c) lazy_index.commit_window(new_window) — фиксирует новое окно
     Порядок (b) до (c) обязателен — см. контракт в docstring
     ChunkPipeline.shift_window(): так self._window в LazyIndex и активное
     окно в ChunkPipeline не могут разойтись, даже если на шаге (b)
     что-то пойдёт не так (тогда (c) просто не будет вызван, и на
     следующем тике попытка повторится с той же исходной точки).
     Гонки с параллельным seek/повторным сдвигом исключены счётчиками
     _sliding_in_progress/_growth_refresh_in_progress (threading.Event) —
     пока предыдущая попытка не завершилась, новая не запускается.
     Сдвиг окна не трогает буферы кадров/audio_clock/_current_frame_idx —
     PTS кадров глобальны и не зависят от локальной адресации окна, поэтому
     переключение незаметно для отображения (в отличие от seek).

- seek(): добавлен флаг _seek_in_progress (True на время выполнения
  асинхронного seek, False после завершения/ошибки/устаревания) — нужен
  только для гейта скользящего окна в п.2 выше, на исход самого seek не
  влияет.
- _on_seek_complete(): добавлен вызов lazy_index.commit_window(window) —
  seek тоже полностью меняет активное окно (как и update_window() в
  ChunkPipeline), поэтому LazyIndex.window должен указывать на то же окно,
  что стало активно в конвейере; без этого следующий тик скользящего окна
  строил бы новое окно от устаревшего self._window.
"""

import time
import threading
import logging
from typing import Optional

import numpy as np

from buffer.frame_buffer import FrameRingBuffer
from core.sync_manager import SyncManager
from pipeline.chunk_pipeline import ChunkPipeline
from seek.seek_engine import SeekEngine
from core.master_clock import MasterClock
from index.lazy_index import LazyIndex
from config.timebase import (
    AUDIO_SAMPLE_RATE,
    SAMPLES_PER_VIDEO_FRAME,
    FRAMES_PER_CHUNK,
    video_frame_to_pts,
    pts_to_video_frame,
)

logger = logging.getLogger(__name__)

SEEK_SPEEDS = [2.0, 4.0, 8.0]

# Как часто фоново синхронизировать метаданные LazyIndex с диском (см.
# _tick_window_management, п.1). Источник .idx на проде обновляется раз в
# 10-15 сек — 5 сек даёт запас, не нагружая диск чаще необходимого.
_GROWTH_REFRESH_INTERVAL_SEC = 10.0

# Минимальный интервал между повторными попытками сдвига окна, если
# предыдущая попытка ничего не дала (build_slid_window вернул None —
# например, новых данных ещё не подвезли). Без этого кулдауна проверка
# могла бы запускать попытку на каждый кадр рендера (до ~60 раз/сек) в
# ожидании следующего цикла обновления .idx.
_SLIDE_RETRY_COOLDOWN_SEC = 1.0


class PlaybackEngine:
    def __init__(
        self,
        pipeline: ChunkPipeline,
        seek_engine: SeekEngine,
        sync_manager: SyncManager,
        master_clock: Optional[MasterClock],
        video_buffer: FrameRingBuffer,
        start_frame_offset: int = 0,
        total_frames: int = 0,
        fps: float = 25.0,
        lazy_index: Optional[LazyIndex] = None,
    ):
        self._pipeline = pipeline
        self._seek_engine = seek_engine
        self._sync = sync_manager
        self._master_clock = master_clock
        self._video_buffer = video_buffer
        self._lazy_index = lazy_index

        # Тройной буфер
        self._display_buffer = video_buffer
        self._fill_buffer = video_buffer
        self._free_buffer = FrameRingBuffer(max_frames=video_buffer.max_frames)
        self._buffer_lock = threading.Lock()
        self._pending_clear: Optional[FrameRingBuffer] = None

        self.start_frame_offset = start_frame_offset
        self.total_frames = total_frames
        self.fps = fps

        self.playing = False
        self._paused = False
        self._audio_clock = 0
        self._clock_lock = threading.Lock()
        self._current_frame_idx = 0

        # JKL
        self._seek_speed = 1.0
        self._seek_speed_index = -1
        self._seek_direction = 0
        self._last_seek_time = 0.0
        self._seek_accumulator = 0.0
        self._normal_playing_state = False

        self._playback_started = False

        # Защита от повторного seek и поколение запросов
        self._seek_generation = 0
        self._seek_lock = threading.Lock()
        self._seek_in_progress = False

        # Эталонный PTS для фильтрации старых кадров после seek
        self._seek_pts_reference = 0

        # Скользящее окно (см. докстринг модуля) и фоновая синхронизация
        # метаданных LazyIndex с диском
        self._sliding_in_progress = threading.Event()
        self._growth_refresh_in_progress = threading.Event()
        self._last_growth_refresh_ts = 0.0
        self._last_slide_attempt_ts = 0.0

    # ------------------------------------------------------------------
    def set_master_clock(self, master_clock: MasterClock):
        self._master_clock = master_clock

    def set_lazy_index(self, lazy_index: LazyIndex):
        """
        Внедряет LazyIndex после создания движка (если конструктору его не
        передали). Пока не установлен — вся логика скользящего окна и
        фоновой синхронизации метаданных (_tick_window_management) не
        активна, класс ведёт себя ровно как без этой возможности.
        """
        self._lazy_index = lazy_index

    # ------------------------------------------------------------------
    def start_playback(self, global_start_frame: int, window_start_frame: int):
        if self._playback_started:
            logger.warning("start_playback вызван повторно, игнорируем")
            return
        self._playback_started = True

        if self._pipeline:
            self._pipeline.stop()

        local_chunk = (global_start_frame - window_start_frame) // FRAMES_PER_CHUNK
        pts = video_frame_to_pts(global_start_frame)
        with self._clock_lock:
            self._audio_clock = pts
        self._current_frame_idx = global_start_frame

        # Сбрасываем эталонный PTS для обычного воспроизведения
        self._seek_pts_reference = 0
        self._pipeline.set_seek_reference(0)
        self._sync.set_seek_reference(0)

        self._display_buffer = self._fill_buffer
        self._fill_buffer.clear()
        self._free_buffer.clear()

        self._pipeline.start(start_local_chunk=local_chunk)

        waited = 0.0
        while self._fill_buffer.count == 0 and waited < 5.0:
            time.sleep(0.1)
            waited += 0.1

        first = self._fill_buffer.peek_first()
        if first is not None:
            self._display_buffer.update_keep_last(first[1], first[0])

        self.playing = False
        self._paused = True

    def resume(self):
        if not self._paused:
            return
        self._paused = False
        self.playing = True
        if self._master_clock:
            self._master_clock.set_clock(self._audio_clock)
            self._master_clock.start()
        self._sync.reset_drift()

    def pause(self):
        if not self.playing:
            return
        self.playing = False
        self._paused = True
        if self._master_clock:
            self._master_clock.stop()

    def stop(self):
        self.playing = False
        self._paused = False
        self._pipeline.stop()
        if self._master_clock:
            self._master_clock.stop()

    # ------------------------------------------------------------------
    def seek(self, global_frame_idx: int, window: 'IndexWindow',
             on_complete: Optional[callable] = None):
        with self._seek_lock:
            self._seek_generation += 1
            gen = self._seek_generation
            self.pause()

        # Гейт для скользящего окна (_tick_window_management) — сдвиг окна
        # не должен запускаться параллельно с seek. Сбрасывается либо в
        # _on_seek_complete() при успешном завершении АКТУАЛЬНОГО запроса,
        # либо сразу при ошибке ниже.
        self._seek_in_progress = True

        def _on_seek_complete_with_callback(buf, win, g):
            self._on_seek_complete(buf, win, g)
            if on_complete:
                on_complete()

        def _on_seek_error(msg):
            logger.error(f"Seek error: {msg}")
            self._seek_in_progress = False

        self._seek_engine.seek_async(
            global_frame_idx,
            on_complete=lambda buf: _on_seek_complete_with_callback(buf, window, gen),
            on_error=_on_seek_error,
        )

    def _on_seek_complete(self, buffer: FrameRingBuffer, window: 'IndexWindow', gen: int):
        with self._seek_lock:
            if gen != self._seek_generation:
                return  # устаревший запрос (более новый seek уже идёт —
                         # он сам держит _seek_in_progress=True и сам его сбросит)

        # НОВОЕ: устанавливаем эталон ДО всех манипуляций с буферами
        first_entry = buffer.peek_first()
        if first_entry:
            new_reference = first_entry[0]
            self._seek_pts_reference = new_reference
            self._pipeline.set_seek_reference(new_reference)
            self._sync.set_seek_reference(new_reference)

        # 1. Очищаем free_buffer
        self._free_buffer.clear()

        # 2. Переносим кадры из временного буфера в free_buffer
        while True:
            entry = buffer.peek_first()
            if entry is None:
                break
            pts, frame = entry
            if not self._free_buffer.try_push(frame, pts):
                break
            buffer.advance()

        # 3. Атомарно переключаем буферы
        with self._buffer_lock:
            old_display = self._display_buffer

            self._display_buffer = self._free_buffer
            self._fill_buffer = self._free_buffer
            self._free_buffer = old_display

            self._pending_clear = old_display

        # 4. Запускаем фоновую очистку старого буфера
        def _clear_old_buffer():
            if self._pending_clear:
                buf = self._pending_clear
                self._pending_clear = None
                buf.clear()

        threading.Thread(target=_clear_old_buffer, daemon=True).start()

        # 5. Обновляем конвейер и планировщик
        self._pipeline.update_window(window)
        self._pipeline.set_video_buffer(self._fill_buffer)
        try:
            self._pipeline.flush()
        except AttributeError:
            logger.warning("Метод flush() отсутствует в ChunkPipeline, пропускаем очистку очередей")
        # Seek — это полная смена окна, поэтому LazyIndex.window должен
        # указывать на то же окно, что теперь активно в конвейере (при
        # скользящем окне эту синхронизацию делает shift_window()+
        # commit_window(), при seek — pipeline.update_window() выше и
        # commit_window() здесь).
        if self._lazy_index is not None:
            self._lazy_index.commit_window(window)

        first = self._display_buffer.peek_first()
        if first:
            pts, frame = first
            self._display_buffer.update_keep_last(frame, pts)
            with self._clock_lock:
                self._audio_clock = pts
            self._current_frame_idx = pts_to_video_frame(pts)
            if self._master_clock:
                self._master_clock.set_clock(pts)
                self._master_clock.flush_audio()

        local_chunk = (self._current_frame_idx - window.window_start_frame) // FRAMES_PER_CHUNK
        self._pipeline._scheduler.set_normal_mode(local_chunk, window.total_chunks)

        self._paused = True
        self.playing = False
        self._seek_in_progress = False

    # ------------------------------------------------------------------
    # JKL
    # ------------------------------------------------------------------
    def set_speed(self, direction: int):
        if self.total_frames == 0:
            return
        if self._seek_direction != direction:
            self._seek_speed_index = 0
            self._seek_direction = direction
        else:
            self._seek_speed_index = min(self._seek_speed_index + 1, 2)
        self._seek_speed = SEEK_SPEEDS[self._seek_speed_index]
        if self._seek_speed_index == 0:
            self._normal_playing_state = self.playing
            if self._master_clock:
                self._master_clock.set_muted(True)
        self._seek_accumulator = 0.0
        self._last_seek_time = time.monotonic()

    def reset_speed(self):
        if self._seek_speed == 1.0 and self._seek_direction == 0:
            return
        self._seek_speed = 1.0
        self._seek_speed_index = -1
        self._seek_direction = 0
        self._seek_accumulator = 0.0
        if self._master_clock:
            self._master_clock.set_muted(False)
        if self._normal_playing_state and not self.playing:
            self.resume()
        elif not self._normal_playing_state and self.playing:
            self.pause()

    def get_speed_display(self) -> str:
        if self._seek_direction == 0:
            return "▶ x1"
        direction = "<<" if self._seek_direction < 0 else ">>"
        return f"{direction} x{self._seek_speed:.0f}"

    # ------------------------------------------------------------------
    def get_display_frame(self) -> Optional[np.ndarray]:
        if self._master_clock and self.playing:
            with self._clock_lock:
                self._audio_clock = self._master_clock.get_audio_clock()

        self._tick_window_management()

        with self._buffer_lock:
            display_buf = self._display_buffer

        if self._seek_direction != 0 and self._seek_speed > 1.0:
            now = time.monotonic()
            dt = now - self._last_seek_time
            self._last_seek_time = now
            frames_to_skip = self._seek_speed * self.fps * dt
            self._seek_accumulator += frames_to_skip
            if self._seek_accumulator >= 1.0:
                skip = int(self._seek_accumulator)
                self._seek_accumulator -= skip
                target = self._current_frame_idx + skip * self._seek_direction
                target = max(0, min(target, self.total_frames - 1))
                self._fast_seek(target)
            return display_buf.get_keep_last()

        return self._sync.get_display_frame(
            display_buf,
            audio_clock=self._audio_clock,
            playing=self.playing,
        )

    def _fast_seek(self, frame_idx: int):
        with self._seek_lock:
            self._current_frame_idx = frame_idx
            pts = video_frame_to_pts(frame_idx)
            with self._clock_lock:
                self._audio_clock = pts
        self._display_buffer.drop_until(pts)
        first = self._display_buffer.peek_first()
        if first:
            self._display_buffer.update_keep_last(first[1], first[0])

    @property
    def audio_clock(self) -> int:
        with self._clock_lock:
            return self._audio_clock

    # ------------------------------------------------------------------
    # Скользящее окно и фоновая синхронизация метаданных с диском
    # ------------------------------------------------------------------
    def _tick_window_management(self):
        """
        Вызывается на каждый кадр рендера из get_display_frame(). Сама
        проверка дешёвая (сравнения чисел под коротким локом внутри
        LazyIndex); вся тяжёлая работа (mmap remap, построение окна) уходит
        в отдельные daemon-потоки, чтобы не задерживать рендер ни на кадр.
        """
        if self._lazy_index is None or not self._playback_started:
            return

        now = time.monotonic()

        # 1. Держим метаданные свежими — независимо от режима
        # воспроизведения (работает и во время seek/JKL, это безопасно:
        # просто синхронизация с диском, активное окно не трогает).
        if now - self._last_growth_refresh_ts >= _GROWTH_REFRESH_INTERVAL_SEC:
            self._last_growth_refresh_ts = now
            if not self._growth_refresh_in_progress.is_set():
                self._growth_refresh_in_progress.set()
                threading.Thread(
                    target=self._refresh_growth_async, daemon=True
                ).start()

        # 2. Скользящее окно — только вне seek/JKL (см. докстринг модуля).
        if self._seek_in_progress or self._seek_direction != 0:
            return
        if self._sliding_in_progress.is_set():
            return
        if now - self._last_slide_attempt_ts < _SLIDE_RETRY_COOLDOWN_SEC:
            return

        current = pts_to_video_frame(self._audio_clock)
        if self._lazy_index.is_near_window_end(current):
            self._last_slide_attempt_ts = now
            self._sliding_in_progress.set()
            threading.Thread(
                target=self._slide_window_async, args=(current,), daemon=True
            ).start()

    def _refresh_growth_async(self):
        try:
            got_new_data = self._lazy_index.refresh_from_disk()
            if got_new_data:
                self.total_frames = self._lazy_index.total_frames
                logger.debug(
                    "PlaybackEngine: индекс обновлён с диска, total_frames=%d",
                    self.total_frames,
                )
        except Exception:
            logger.exception("PlaybackEngine: ошибка обновления индекса с диска")
        finally:
            self._growth_refresh_in_progress.clear()

    def _slide_window_async(self, current_frame: int):
        try:
            new_window = self._lazy_index.build_slid_window(current_frame)
            if new_window is None:
                logger.debug(
                    "PlaybackEngine: скользящее окно не построено "
                    "(новых данных пока недостаточно)"
                )
                return

            # Порядок обязателен: сначала переключаем конвейер (может
            # упасть — тогда commit_window() ниже не вызовется, и
            # LazyIndex.window останется прежним, согласованным с
            # конвейером; на следующем тике попытка повторится).
            self._pipeline.shift_window(new_window)
            self._lazy_index.commit_window(new_window)
            self.total_frames = self._lazy_index.total_frames

            logger.info(
                "PlaybackEngine: окно сдвинуто, новые границы %d-%d, total_frames=%d",
                new_window.window_start_frame, new_window.window_end_frame,
                self.total_frames,
            )
        except Exception:
            logger.exception("PlaybackEngine: ошибка сдвига окна")
        finally:
            self._sliding_in_progress.clear()

    # ------------------------------------------------------------------
    def get_local_timecode_str(self) -> str:
        idx = pts_to_video_frame(self._audio_clock)
        total_seconds = idx / self.fps
        h = int(total_seconds // 3600)
        m = int((total_seconds % 3600) // 60)
        s = int(total_seconds % 60)
        f = int(round((total_seconds - int(total_seconds)) * self.fps))
        return f"{h:02d}:{m:02d}:{s:02d};{f:02d}"

    def get_real_timecode_str(self) -> str:
        idx = pts_to_video_frame(self._audio_clock) + self.start_frame_offset
        total_seconds = idx / self.fps
        h = int(total_seconds // 3600)
        m = int((total_seconds % 3600) // 60)
        s = int(total_seconds % 60)
        f = int(round((total_seconds - int(total_seconds)) * self.fps))
        return f"{h:02d}:{m:02d}:{s:02d};{f:02d}"

    def close(self):
        self.stop()
        self._seek_engine.cancel_current()
