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

ИЗМЕНЕНИЯ (правки по слайдеру и защите от преждевременного закрытия):
- В get_display_frame() при JKL-перемотке теперь используется SyncManager
  для отображения кадра, соответствующего текущей позиции, а не keep_last.
- В _tick_window_management() текущая позиция берётся из audio_clock
  (pts_to_video_frame(self._audio_clock)), а не из _current_frame_idx,
  чтобы окно сдвигалось при фактическом воспроизведении.
- В close() добавлено ожидание завершения фоновых потоков сдвига и
  обновления метаданных (с таймаутом), чтобы избежать работы с уже
  остановленными компонентами.
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

_GROWTH_REFRESH_INTERVAL_SEC = 5.0
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

        # Скользящее окно и фоновая синхронизация метаданных
        self._sliding_in_progress = threading.Event()
        self._growth_refresh_in_progress = threading.Event()
        self._last_growth_refresh_ts = 0.0
        self._last_slide_attempt_ts = 0.0
        self._window_op_lock = threading.Lock()
        self._window_op_generation = 0
        self._closed = threading.Event()

    # ------------------------------------------------------------------
    def set_master_clock(self, master_clock: MasterClock):
        self._master_clock = master_clock

    def set_lazy_index(self, lazy_index: LazyIndex):
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

        self._seek_in_progress = True

        with self._window_op_lock:
            self._window_op_generation += 1

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
                return

        first_entry = buffer.peek_first()
        if first_entry:
            new_reference = first_entry[0]
            self._seek_pts_reference = new_reference
            self._pipeline.set_seek_reference(new_reference)
            self._sync.set_seek_reference(new_reference)

        self._free_buffer.clear()

        while True:
            entry = buffer.peek_first()
            if entry is None:
                break
            pts, frame = entry
            if not self._free_buffer.try_push(frame, pts):
                break
            buffer.advance()

        with self._buffer_lock:
            old_display = self._display_buffer

            self._display_buffer = self._free_buffer
            self._fill_buffer = self._free_buffer
            self._free_buffer = old_display

            self._pending_clear = old_display

        def _clear_old_buffer():
            if self._pending_clear:
                buf = self._pending_clear
                self._pending_clear = None
                buf.clear()

        threading.Thread(target=_clear_old_buffer, daemon=True).start()

        self._pipeline.update_window(window)
        self._pipeline.set_video_buffer(self._fill_buffer)
        try:
            self._pipeline.flush()
        except AttributeError:
            logger.warning("Метод flush() отсутствует в ChunkPipeline, пропускаем очистку очередей")
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
        if self._master_clock:
            with self._clock_lock:
                self._audio_clock = self._master_clock.get_audio_clock()

        # Обновляем текущую позицию из audio_clock, если не идёт seek/JKL
        if self._seek_direction == 0 and not self._seek_in_progress:
            self._current_frame_idx = pts_to_video_frame(self._audio_clock)

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
            # Показываем кадр, соответствующий текущей позиции при JKL
            return self._sync.get_display_frame(
                display_buf,
                audio_clock=self._audio_clock,
                playing=True,
            )

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
        if self._lazy_index is None or not self._playback_started or self._closed.is_set():
            return

        now = time.monotonic()

        if now - self._last_growth_refresh_ts >= _GROWTH_REFRESH_INTERVAL_SEC:
            self._last_growth_refresh_ts = now
            if not self._growth_refresh_in_progress.is_set():
                self._growth_refresh_in_progress.set()
                threading.Thread(
                    target=self._refresh_growth_async, daemon=True
                ).start()

        if self._seek_in_progress or self._seek_direction != 0:
            return
        if self._sliding_in_progress.is_set():
            return
        if now - self._last_slide_attempt_ts < _SLIDE_RETRY_COOLDOWN_SEC:
            return

        # Используем audio_clock для актуальной позиции
        current = pts_to_video_frame(self._audio_clock)
        if self._lazy_index.is_near_window_end(current):
            self._last_slide_attempt_ts = now
            self._sliding_in_progress.set()
            threading.Thread(
                target=self._slide_window_async, args=(current,), daemon=True
            ).start()

    def _refresh_growth_async(self):
        try:
            if self._closed.is_set():
                return
            got_new_data = self._lazy_index.refresh_from_disk()
            if got_new_data and not self._closed.is_set():
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
        if self._closed.is_set():
            self._sliding_in_progress.clear()
            return

        with self._window_op_lock:
            start_gen = self._window_op_generation

        try:
            new_window = self._lazy_index.build_slid_window(current_frame)
            if new_window is None:
                logger.debug(
                    "PlaybackEngine: скользящее окно не построено "
                    "(новых данных пока недостаточно)"
                )
                return

            with self._window_op_lock:
                if self._closed.is_set() or self._window_op_generation != start_gen:
                    logger.debug(
                        "PlaybackEngine: сдвиг окна отброшен — за время "
                        "построения начался seek/другой сдвиг, либо плеер закрыт"
                    )
                    return

                self._pipeline.shift_window(new_window)
                self._lazy_index.commit_window(new_window)
                self._window_op_generation += 1

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
        self._closed.set()
        self.stop()
        self._seek_engine.cancel_current()
        # Ждём завершения фоновых потоков сдвига и обновления метаданных
        self._sliding_in_progress.wait(timeout=2.0)
        self._growth_refresh_in_progress.wait(timeout=2.0)