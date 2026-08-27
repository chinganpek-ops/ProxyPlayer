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

- seek(): использует единый автомат WindowTransition (см. класс выше) —
  begin_transition(SEEKING, force=True) в начале, end_transition() по
  завершении/ошибке. Гейтит скользящий сдвиг (не может начаться поверх
  идущего seek) и защищает от применения устаревшего сдвига поверх
  только что установленного seek'ом окна (см. _slide_window_async).
- _on_seek_complete(): добавлен вызов lazy_index.commit_window(window) —
  seek тоже полностью меняет активное окно (как и update_window() в
  ChunkPipeline), поэтому LazyIndex.window должен указывать на то же окно,
  что стало активно в конвейере; без этого следующий тик скользящего окна
  строил бы новое окно от устаревшего self._window.
"""

import time
import threading
import logging
from enum import Enum, auto
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
_GROWTH_REFRESH_INTERVAL_SEC = 5.0

# Минимальный интервал между повторными попытками сдвига окна, если
# предыдущая попытка ничего не дала (build_slid_window вернул None —
# например, новых данных ещё не подвезли). Без этого кулдауна проверка
# могла бы запускать попытку на каждый кадр рендера (до ~60 раз/сек) в
# ожидании следующего цикла обновления .idx.
_SLIDE_RETRY_COOLDOWN_SEC = 1.0


class WindowTransition(Enum):
    """
    Явный автомат состояний для управляющей плоскости (кто сейчас имеет
    право менять активное окно ChunkPipeline/позицию StreamScheduler).

    ПРАВКА (продакшен-ревью, "навести порядок в сигналах переключения"):
    раньше это же самое было раскидано по трём отдельным полям —
    _seek_in_progress (bool), _window_op_generation (int) и
    _window_op_lock — которые вместе означали ровно одно и то же: "какое
    из двух действий, меняющих окно (seek или скользящий сдвиг), сейчас
    имеет право быть в процессе, и какое из них последнее по времени".
    Сведено в один явный автомат (_transition_state/_transition_generation
    + begin_transition()/end_transition()), чтобы у каждого действия,
    трогающего окно конвейера, был один явный "сигнал начала" и один
    явный "сигнал завершения", а не несколько флагов, которые нужно
    помнить синхронизировать вручную в каждом месте. Конвейер чтения
    (ReaderStage/DemuxerStage/декодеры) этот автомат НЕ трогает и
    продолжает работать непрерывно — гейтится только то, что реально
    было источником гонок: смена активного окна/позиции планировщика.
    """
    IDLE = auto()
    SEEKING = auto()
    SLIDING = auto()


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

        # Эталонный PTS для фильтрации старых кадров после seek
        self._seek_pts_reference = 0

        # Скользящее окно (см. докстринг модуля) и фоновая синхронизация
        # метаданных LazyIndex с диском
        self._sliding_in_progress = threading.Event()
        self._growth_refresh_in_progress = threading.Event()
        self._last_growth_refresh_ts = 0.0
        self._last_slide_attempt_ts = 0.0

        # Единый автомат состояний управляющей плоскости (см. класс
        # WindowTransition выше) — заменяет собой прежние отдельные поля
        # _seek_in_progress/_window_op_lock/_window_op_generation. Ровно
        # один "переход" (seek ИЛИ сдвиг) может быть в процессе; какой
        # именно — видно явно в self._transition_state (полезно и для
        # логов/диагностики, а не только для внутренней синхронизации).
        self._transition_lock = threading.Lock()
        self._transition_state = WindowTransition.IDLE
        self._transition_generation = 0

        # Флаг, который проверяют _tick_window_management() и фоновые
        # _refresh_growth_async()/_slide_window_async() — и перед началом
        # работы, и перед применением уже готового результата. Без него
        # daemon-поток, запущенный ДО close(), мог продолжать работать и
        # в какой-то момент попытаться применить результат к уже
        # остановленному/закрытому pipeline/LazyIndex (StreamController.close()
        # обнуляет их следом за PlaybackEngine.close()) — не крэш
        # благодаря try/except в этих методах, но лишняя работа и шум в
        # логах после того, как окно уже закрыто пользователем.
        self._closed = threading.Event()

    # ------------------------------------------------------------------
    def _begin_transition(self, kind: WindowTransition, force: bool = False) -> Optional[int]:
        """
        Явно "включает" переход указанного вида. Возвращает поколение
        перехода (нужно передать в _end_transition), либо None, если
        переход не начат (уже идёт другой, а force=False).

        force=True (используется только seek()) — сигнал seek'а обязан
        победить в любом случае: seek имеет приоритет над ещё не
        завершившимся сдвигом окна (но не наоборот — сдвиг никогда не
        может начаться поверх идущего seek, см. force=False в
        _slide_window_async и ранний гейт в _tick_window_management).
        """
        with self._transition_lock:
            if not force and self._transition_state != WindowTransition.IDLE:
                return None
            self._transition_generation += 1
            self._transition_state = kind
            return self._transition_generation

    def _end_transition(self, gen: int) -> bool:
        """
        Явно "выключает" переход. Возвращает True, если это действительно
        завершение ТЕКУЩЕГО (не устаревшего) перехода — тогда вызывающий
        может безопасно применять свой результат. Возвращает False, если
        переход уже устарел (кто-то другой начал новый, пока этот
        выполнялся) — тогда результат нужно отбросить, не применяя.
        """
        with self._transition_lock:
            if gen != self._transition_generation:
                return False
            self._transition_state = WindowTransition.IDLE
            return True

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

        # ПРАВКА (продакшен-ревью, диагностика "пропал звук"): раньше здесь
        # НЕ вызывался self._master_clock.set_clock(pts) — внутренний
        # счётчик MasterClock._samples_played оставался на 0 (или на
        # значении после predыдущего reset()) и НЕ совпадал с реальной
        # стартовой позицией global_start_frame (у растущего файла это
        # может быть очень большое число кадров, если старт — near-live).
        # get_display_frame() каждый тик перезаписывает self._audio_clock
        # значением master_clock.get_audio_clock() — то есть уже на первом
        # тике корректно выставленный выше self._audio_clock тут же
        # затирался рассинхронизированным значением "прошло N сэмплов
        # реального времени с момента start()", никак не привязанным к
        # позиции в файле. Само по себе это не глушит звук (продакшен
        # аудио идёт по своей, независимой цепочке через PTS-окно в
        # AudioDecoderStage), но ломает то, что от audio_clock реально
        # зависит: is_near_window_end()/build_slid_window() в
        # _tick_window_management() и JKL-таргетинг — при большом
        # рассинхроне скользящее окно может никогда не сдвинуться, а JKL
        # считать шаг от неверной точки. Симметрично тому, что уже
        # корректно делает _on_seek_complete().
        if self._master_clock:
            self._master_clock.set_clock(pts)

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
    def seek(self, global_frame_idx: int, window: 'IndexWindow' = None,
             on_complete: Optional[callable] = None,
             on_error: Optional[callable] = None):
        """
        Запускает перемотку.

        window необязателен. SeekEngine всё равно открывает окно сам внутри
        _do_seek() (lazy_index.open_window(frame_idx)), поэтому передавать
        его извне не нужно — и не стоит: окно, открытое ДО запуска seek,
        может устареть, если за время выполнения запроса произойдёт сдвиг
        окна. Если window не передан, актуальное окно берётся у LazyIndex в
        момент завершения — то есть ровно то, которое seek и использовал.

        Аргумент оставлен в сигнатуре для обратной совместимости с кодом,
        который его ещё передаёт.

        on_error(msg) вызывается при любом неуспехе: ошибка чтения/декодирования,
        отмена, таймаут watchdog в SeekEngine, отсутствие активного окна.
        Без него вызывающий (UI) не отличает "seek ещё идёт" от "seek уже
        провалился" и остаётся заблокированным навсегда — именно так
        возникала "вечная заглушка" после неудачной перемотки.
        """
        with self._seek_lock:
            self._seek_generation += 1
            gen = self._seek_generation
            self.pause()

        # Явный сигнал "начат переход SEEKING" — force=True: seek обязан
        # победить в любой ситуации, даже если сейчас выполняется сдвиг
        # окна (см. докстринг _begin_transition). transition_gen передаём
        # в колбэки ниже — по нему _on_seek_complete()/_on_seek_error()
        # узнают, актуален ли ещё именно ИХ переход, когда придёт время
        # его завершать (_end_transition), и не затрут чужой более новый.
        transition_gen = self._begin_transition(WindowTransition.SEEKING, force=True)

        def _on_seek_complete_with_callback(buf, win, g):
            ok = self._on_seek_complete(buf, win, g, transition_gen)
            if ok:
                if on_complete:
                    on_complete()
            elif on_error:
                try:
                    on_error("seek завершился без применения результата")
                except Exception:
                    logger.exception("Ошибка в on_error seek-запроса")

        def _on_seek_error(msg):
            logger.error(f"Seek error: {msg}")
            self._end_transition(transition_gen)
            if on_error:
                try:
                    on_error(msg)
                except Exception:
                    logger.exception("Ошибка в on_error seek-запроса")

        self._seek_engine.seek_async(
            global_frame_idx,
            on_complete=lambda buf: _on_seek_complete_with_callback(buf, window, gen),
            on_error=_on_seek_error,
        )

    def _on_seek_complete(self, buffer: FrameRingBuffer, window: 'IndexWindow',
                          gen: int, transition_gen: int) -> bool:
        """
        Применяет результат seek. Возвращает True, если результат
        действительно применён (буферы переключены, окно обновлено), и
        False, если запрос устарел либо окно получить не удалось — тогда
        вызывающий обязан сообщить об этом через on_error, иначе UI
        останется в состоянии "перемотка выполняется" навсегда.
        """
        with self._seek_lock:
            if gen != self._seek_generation:
                self._end_transition(transition_gen)
                # Устаревший запрос: более новый seek уже идёт и сам держит
                # переход SEEKING. Возвращаем False — вызывающий не должен
                # трактовать это как успешное завершение своего запроса.
                return False

        # НОВОЕ: устанавливаем эталон ДО всех манипуляций с буферами
        # Если окно не передано (обычный путь — см. докстринг seek()),
        # берём актуальное у LazyIndex: SeekEngine уже открыл его внутри
        # _do_seek(), поэтому здесь оно гарантированно соответствует тому,
        # по которому реально выполнялась перемотка.
        if window is None:
            window = getattr(self._lazy_index, "window", None) if self._lazy_index else None
        if window is None:
            logger.error("Seek: не удалось получить активное окно — "
                         "перемотка не применена (кадр %s)",
                         pts_to_video_frame(buffer.peek_first()[0])
                         if buffer.peek_first() else "?")
            self._end_transition(transition_gen)
            return False

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

        # ПРАВКА (продакшен-ревью, гонка при seek): вычисляем pts/
        # current_frame_idx/local_chunk ДО обращения к конвейеру — раньше
        # это делалось ПОСЛЕ pipeline.update_window()/set_video_buffer(),
        # из-за чего update_window() сначала выставлял планировщик на
        # set_normal_mode(0, ...), а верное значение приходило отдельным
        # вызовом set_normal_mode(local_chunk, ...) несколькими строками
        # ниже — в промежутке между ними ReaderStage (работает непрерывно,
        # независимо от seek) мог успеть прочитать чанк 0 нового окна,
        # заведомо не то место, куда метил seek. Теперь оба значения
        # известны заранее и передаются в update_window() одним вызовом —
        # set_normal_mode вызывается ровно один раз, гонка исчезает.
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

        # 5. Обновляем конвейер и планировщик — единым атомарным вызовом,
        # сразу с правильным стартовым чанком (см. правку выше).
        self._pipeline.update_window(window, start_local_chunk=local_chunk)
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

        self._paused = True
        self.playing = False
        self._end_transition(transition_gen)
        return True

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

        # ПРАВКА (продакшен-ревью, аудио/видео синхронизация — риск дропов):
        # _current_frame_idx раньше обновлялся ТОЛЬКО в start_playback()/
        # _on_seek_complete()/_fast_seek() (JKL) и оставался ЗАМОРОЖЕННЫМ
        # во время обычного воспроизведения. Именно на нём основаны
        # is_near_window_end()/build_slid_window() в _tick_window_management()
        # — то есть триггер и точка привязки скользящего окна работали от
        # устаревшей, никогда не растущей позиции. Реальное воспроизведение
        # могло дойти до конца САМОГО ПЕРВОГО открытого окна, пока
        # total_frames в LazyIndex рос через refresh_from_disk() — но
        # активное окно в ChunkPipeline не сдвигалось (сдвиг не триггерился
        # вовремя от замороженной позиции). Планировщик упирался в
        # total_chunks старого окна и переставал отдавать чанки на чтение,
        # буфер MasterClock истощался — это и проявлялось как зависания и
        # провалы звука (AUDIO_UNDERRUN в audio_monitor.log), хотя новые
        # данные уже были в индексе.
        #
        # get_audio_clock() — это надёжный источник текущей позиции при
        # обычном воспроизведении: сэмплы в MasterClock._callback()
        # считаются безусловно на каждый вызов звуковой карты, даже при
        # underrun, то есть это честные "настенные часы". Обновляем
        # _current_frame_idx из него на каждый тик — кроме seek/JKL, где
        # позицией управляют _on_seek_complete()/_fast_seek() (иначе свежий,
        # но ещё не подтверждённый seek'ом audio_clock мог бы затереть
        # позицию, установленную seek'ом/JKL, до их завершения).
        if self._seek_direction == 0 and self._transition_state != WindowTransition.SEEKING:
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
        if self._lazy_index is None or not self._playback_started or self._closed.is_set():
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
        # Проверка self._transition_state здесь — дешёвый, но не строго
        # обязательный ранний выход (реальный гейт — force=False в
        # _begin_transition(SLIDING) внутри _slide_window_async); смысл
        # именно здесь — не тратить поток на попытку, которая заведомо
        # будет отклонена.
        if self._transition_state != WindowTransition.IDLE or self._seek_direction != 0:
            return
        if self._sliding_in_progress.is_set():
            return
        if now - self._last_slide_attempt_ts < _SLIDE_RETRY_COOLDOWN_SEC:
            return

        current = self._current_frame_idx
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
                # Индекс подтвердил появление новых записей — значит,
                # соответствующие байты уже в mdat (индекс пишется ПОСЛЕ
                # данных). Прокидываем сигнал в конвейер, чтобы ReaderStage
                # обновил размер файла у своего ридера: ридер создаётся один
                # раз на сеанс и иначе продолжит клампить чтение по размеру,
                # снятому при открытии, упираясь в стену у live-края.
                # Свои границы (mdat_end) LazyIndex обновляет сам, внутри
                # refresh_from_disk(); здесь только доставка сигнала тому
                # компоненту, о котором LazyIndex знать не должен.
                self._pipeline.notify_file_grew()
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

        # force=False: сдвиг НИКОГДА не может начаться поверх идущего
        # seek (или другого сдвига) — если сейчас что-то уже выполняется,
        # просто тихо отступаем, следующий тик попробует снова.
        transition_gen = self._begin_transition(WindowTransition.SLIDING, force=False)
        if transition_gen is None:
            logger.debug("PlaybackEngine: сдвиг окна отложен — уже идёт другой переход")
            self._sliding_in_progress.clear()
            return

        try:
            new_window = self._lazy_index.build_slid_window(current_frame)
            if new_window is None:
                logger.debug(
                    "PlaybackEngine: скользящее окно не построено "
                    "(новых данных пока недостаточно)"
                )
                return

            # ПРАВКА (продакшен-ревью): если за время построения new_window
            # (build_slid_window выше) успел начаться seek — он уже
            # выполнил _begin_transition(force=True), продвинув поколение
            # перехода. Проверка и применение держатся под ОДНИМ локом
            # (как единая атомарная операция) — иначе между "подтвердили
            # актуальность" и "применили к конвейеру" мог бы успеть
            # начаться новый seek и обе мутации конвейера побежали бы
            # параллельно. Если поколение уже не совпадает — построенное
            # окно отбрасывается: применять его значило бы рассинхронизировать
            # StreamScheduler с окном, которое seek уже установил (особенно
            # заметно при перемотке назад, где новая позиция далеко от той,
            # что использовалась для построения этого окна).
            with self._transition_lock:
                if self._closed.is_set() or transition_gen != self._transition_generation:
                    logger.debug(
                        "PlaybackEngine: сдвиг окна отброшен — за время "
                        "построения начался seek/другой сдвиг, либо плеер закрыт"
                    )
                    return

                # Порядок обязателен: сначала переключаем конвейер (может
                # упасть — тогда commit_window() ниже не вызовется, и
                # LazyIndex.window останется прежним, согласованным с
                # конвейером; на следующем тике попытка повторится).
                self._pipeline.shift_window(new_window)
                self._lazy_index.commit_window(new_window)
                self._transition_state = WindowTransition.IDLE

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
        # ПРАВКА (продакшен-ревью): выставляем ДО stop()/cancel_current(),
        # чтобы _tick_window_management() и уже запущенные
        # _refresh_growth_async()/_slide_window_async() гарантированно
        # увидели закрытие и не начали (и не применили) новую работу поверх
        # компонентов, которые вот-вот остановит/обнулит
        # StreamController.close().
        self._closed.set()
        self.stop()
        self._seek_engine.cancel_current()
