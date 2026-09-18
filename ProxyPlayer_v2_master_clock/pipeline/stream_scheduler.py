"""
stream_scheduler.py – приоритетный планировщик загрузки чанков (локальные индексы).
Добавлено управление заполненностью буфера:
- Загрузка возобновляется при заполнении <= 60%.
- Останавливается при заполнении >= 95%.
- В NORMAL режиме скорость загрузки ограничена 1.15x скорости потребления.

ИЗМЕНЕНИЯ (правки продакшен-ревью):
- extend_total_chunks(): поднимает только верхнюю границу total_chunks, не
  трогая текущую позицию (_current_chunk), уже загруженные чанки
  (_loaded_chunks) и режим (_mode). Изначально был нужен для роста
  live-файла; сейчас, после введения скользящего окна (см. shift_loaded
  ниже), основным механизмом live-роста стал он, но extend_total_chunks()
  оставлен как самостоятельный инструмент — например, если понадобится
  просто продлить границу без сдвига начала (см. LazyIndex.expand_window).

- НОВОЕ: shift_loaded() — пересчитывает локальные индексы планировщика под
  скользящее окно, построенное LazyIndex.build_slid_window(). Когда окно
  сдвигается вперёд (меняются И start, И end), все локальные индексы
  (_current_chunk, _loaded_chunks, _target_chunk) смещены относительно
  нового окна на chunk_shift = new_window_start_chunk - old_window_start_chunk.
  В отличие от set_normal_mode()/set_seek_mode(), НЕ сбрасывает _mode,
  _loading_allowed, _last_chunk_ts — переключение окна не должно прерывать
  то, что планировщик уже знает о текущем воспроизведении (скорость подачи,
  гистерезис буфера). Чанки, ушедшие за пределы нового окна (new_local < 0
  после сдвига — то есть они были в самом начале старого окна, до которого
  новое окно уже не дотягивается), просто выбывают из _loaded_chunks: эти
  данные больше не адресуемы в новой локальной системе координат, и это
  не потеря — соответствующие кадры уже были показаны раньше (сдвиг
  происходит только вперёд, с overlap "с запасом" от текущей позиции).
  Остальная логика планировщика не менялась.
"""

import threading
import logging
import time
from enum import Enum, auto
from typing import Optional, List, Set

from config.timebase import FRAMES_PER_CHUNK, SAMPLES_PER_CHUNK, AUDIO_SAMPLE_RATE
from pipeline.adaptive_chunk import AdaptiveChunkStrategy

logger = logging.getLogger(__name__)


class PlaybackMode(Enum):
    NORMAL = auto()
    SEEK = auto()
    FAST_FORWARD = auto()
    PAUSE = auto()


class StreamScheduler:
    """Планировщик, работающий с локальными индексами чанков внутри окна."""

    def __init__(self, adaptive_strategy: AdaptiveChunkStrategy = None):
        self._adaptive = adaptive_strategy or AdaptiveChunkStrategy()
        self._mode = PlaybackMode.NORMAL
        self._current_chunk = 0
        self._total_chunks = 0

        self._target_chunk = 0
        self._seek_priority_done = False

        self._direction = 0
        self._speed = 1.0

        self._loaded_chunks: Set[int] = set()
        self._lock = threading.Lock()
        self._video_buffer = None

        # --- Управление загрузкой ---
        self._buffer_high_watermark = 0.95  # 95% заполненности
        self._buffer_low_watermark = 0.60   # 60% заполненности
        self._loading_allowed = True
        self._last_chunk_ts = 0.0           # время выдачи последнего чанка
        self._chunk_duration = SAMPLES_PER_CHUNK / AUDIO_SAMPLE_RATE  # 0.48 сек при 25 fps
        self._speed_factor = 1.25           # загрузка со скоростью 1.15x потребления

        logger.info("StreamScheduler создан")

    def set_buffer(self, video_buffer):
        self._video_buffer = video_buffer
        # Пересчитываем абсолютные пороги на основе max_frames буфера
        if video_buffer:
            self._high_threshold = int(video_buffer.max_frames * self._buffer_high_watermark)
            self._low_threshold = int(video_buffer.max_frames * self._buffer_low_watermark)
        else:
            self._high_threshold = 0
            self._low_threshold = 0
        logger.info("StreamScheduler: видеобуфер установлен, пороги: low=%d, high=%d",
                    self._low_threshold, self._high_threshold)

    def set_normal_mode(self, current_local_chunk: int, total_local_chunks: int):
        with self._lock:
            self._mode = PlaybackMode.NORMAL
            self._current_chunk = current_local_chunk
            self._total_chunks = total_local_chunks
            self._loaded_chunks.clear()
            self._loading_allowed = True   # разрешаем загрузку при переходе в NORMAL
            self._last_chunk_ts = 0.0
            logger.info("StreamScheduler: NORMAL mode, current=%d, total=%d",
                        self._current_chunk, self._total_chunks)

    def extend_total_chunks(self, new_total_local_chunks: int):
        """
        Расширяет верхнюю границу доступных чанков БЕЗ сброса текущей
        позиции чтения (_current_chunk), уже загруженных чанков
        (_loaded_chunks) и режима (_mode).

        В отличие от set_normal_mode(), безопасен для вызова во время
        активного воспроизведения.
        """
        with self._lock:
            if new_total_local_chunks > self._total_chunks:
                old = self._total_chunks
                self._total_chunks = new_total_local_chunks
                logger.info("StreamScheduler: total_chunks расширен %d -> %d",
                            old, self._total_chunks)

    def shift_loaded(self, chunk_shift: int, new_total_local_chunks: int):
        """
        Пересчитывает локальные индексы планировщика под новое (сдвинутое)
        окно: new_local = old_local - chunk_shift.

        chunk_shift = new_window.window_start_chunk - old_window.window_start_chunk
        (считает вызывающий — обычно ChunkPipeline.shift_window() — оба
        значения глобальные индексы чанков, так что знак учитывается
        автоматически).

        В отличие от set_normal_mode()/set_seek_mode(), НЕ сбрасывает
        _mode, _loading_allowed, _last_chunk_ts — переключение на
        скользящее окно не должно прерывать то, что планировщик уже знает
        о темпе воспроизведения. Чанки, для которых new_local < 0 или
        new_local >= new_total_local_chunks (вышли за пределы нового окна),
        удаляются из _loaded_chunks — они больше не адресуемы в новой
        локальной системе координат.
        """
        with self._lock:
            if chunk_shift == 0:
                if new_total_local_chunks > self._total_chunks:
                    self._total_chunks = new_total_local_chunks
                logger.debug("[Scheduler] shift_loaded: chunk_shift=0, только total_chunks обновлён")
                return

            old_current = self._current_chunk
            old_target = self._target_chunk
            old_loaded_count = len(self._loaded_chunks)

            shifted_loaded: Set[int] = set()
            for c in self._loaded_chunks:
                nc = c - chunk_shift
                if 0 <= nc < new_total_local_chunks:
                    shifted_loaded.add(nc)
            self._loaded_chunks = shifted_loaded

            self._current_chunk = max(0, old_current - chunk_shift)
            self._target_chunk = max(0, old_target - chunk_shift)
            self._total_chunks = new_total_local_chunks

            logger.info(
                "StreamScheduler: окно сдвинуто (chunk_shift=%d): current %d -> %d, "
                "total_chunks -> %d, loaded_chunks %d -> %d",
                chunk_shift, old_current, self._current_chunk,
                self._total_chunks, old_loaded_count, len(self._loaded_chunks),
            )

    # ------------------------------------------------------------------
    # Публичное состояние (этап 1.2 рефакторинга)
    #
    # Раньше PlaybackEngine и телеметрия читали _mode, _current_chunk,
    # _total_chunks, _loaded_chunks напрямую. Такие связи не проверяются
    # контрактным тестом и ломаются молча при переименовании поля —
    # именно этот класс ошибок оказался самым дорогим в отладке.
    # ------------------------------------------------------------------
    def get_state(self) -> dict:
        """Снимок состояния планировщика. Только чтение, без побочных эффектов."""
        with self._lock:
            return {
                "mode": self._mode.name,
                "current_chunk": self._current_chunk,
                "total_chunks": self._total_chunks,
                "loaded_count": len(self._loaded_chunks),
                "loading_allowed": self._loading_allowed,
                "target_chunk": self._target_chunk,
                "direction": self._direction,
                "speed": self._speed,
            }

    @property
    def mode_name(self) -> str:
        """Имя текущего режима — для гейтов и логирования."""
        return self._mode.name

    @property
    def total_chunks(self) -> int:
        with self._lock:
            return self._total_chunks

    @property
    def current_chunk(self) -> int:
        with self._lock:
            return self._current_chunk

    def set_fast_forward(self, direction: int, speed: float,
                         current_local_chunk: int, total_local_chunks: int):
        """
        Публичный псевдоним set_fast_forward_mode().

        Существует, чтобы ChunkPipeline мог делегировать переключение
        режима, а PlaybackEngine не обращался к планировщику через
        pipeline._scheduler — то есть к приватному полю чужого объекта.
        """
        self.set_fast_forward_mode(direction, speed,
                                   current_local_chunk, total_local_chunks)

    def get_loaded_chunks(self) -> list:
        """
        Отсортированный список уже выданных чанков (копия).

        Нужен там, где важен не счётчик, а сами индексы: проверка
        пересчёта при сдвиге окна (shift_loaded) и диагностика того, какие
        участки окна уже прочитаны. get_state() отдаёт только количество,
        по которому перестановку индексов проверить нельзя.

        Возвращается копия, а не внутреннее множество: иначе вызывающий
        мог бы изменить состояние планировщика, ничего об этом не зная.
        """
        with self._lock:
            return sorted(self._loaded_chunks)

    def reset_rate_limit(self):
        """
        Снимает ограничение темпа выдачи чанков.

        Нужен тестам, которые сейчас пишут в _last_chunk_ts напрямую:
        запись в чужое поле обходит логику объекта и ломается при любом
        изменении механизма ограничения.
        """
        with self._lock:
            self._last_chunk_ts = 0.0

    def set_seek_mode(self, target_local_chunk: int, total_local_chunks: int):
        with self._lock:
            self._mode = PlaybackMode.SEEK
            self._target_chunk = target_local_chunk
            self._current_chunk = target_local_chunk
            self._total_chunks = total_local_chunks
            self._seek_priority_done = False
            self._loaded_chunks.clear()
            logger.info("StreamScheduler: SEEK mode, target=%d, total=%d",
                        self._target_chunk, self._total_chunks)

    def set_fast_forward_mode(self, direction: int, speed: float,
                              current_local_chunk: int, total_local_chunks: int):
        with self._lock:
            self._mode = PlaybackMode.FAST_FORWARD
            self._direction = direction
            self._speed = speed
            self._current_chunk = current_local_chunk
            self._total_chunks = total_local_chunks
            logger.info("StreamScheduler: FAST_FORWARD mode, dir=%d, speed=%.1f, current=%d, total=%d",
                        self._direction, self._speed, self._current_chunk, self._total_chunks)

    def set_pause_mode(self, current_local_chunk: int, total_local_chunks: int):
        with self._lock:
            self._mode = PlaybackMode.PAUSE
            self._current_chunk = current_local_chunk
            self._total_chunks = total_local_chunks
            logger.info("StreamScheduler: PAUSE mode, current=%d, total=%d",
                        self._current_chunk, self._total_chunks)

    # ------------------------------------------------------------------
    def get_next_chunk(self) -> Optional[int]:
        """Возвращает локальный индекс следующего чанка с учётом гистерезиса и скорости."""
        with self._lock:
            # 1. Проверка заполненности буфера (общая для всех режимов)
            # В FAST_FORWARD заполненность видеобуфера не показатель: кадры
            # скраба туда не попадают, буфер остаётся с прежним содержимым и
            # намертво заблокировал бы выдачу чанков.
            if self._video_buffer is not None and self._mode != PlaybackMode.FAST_FORWARD:
                count = self._video_buffer.count
                if count >= self._high_threshold:
                    self._loading_allowed = False
                elif count <= self._low_threshold:
                    self._loading_allowed = True

                if not self._loading_allowed:
                    logger.debug("[Scheduler] Буфер заполнен (%d/%d), загрузка остановлена",
                                 count, self._video_buffer.max_frames)
                    return None
            else:
                # Если буфер не установлен, продолжаем без ограничений
                pass

            # 2. Ограничение темпа выдачи чанков.
            if self._last_chunk_ts > 0:
                elapsed = time.monotonic() - self._last_chunk_ts
                if self._mode == PlaybackMode.NORMAL:
                    min_interval = self._chunk_duration / self._speed_factor
                    if elapsed < min_interval:
                        return None
                elif self._mode == PlaybackMode.FAST_FORWARD:
                    # В FAST_FORWARD видеобуфер не используется как
                    # обратная связь (кадры идут в отдельную ячейку скраба),
                    # поэтому без собственного ограничения чтение ушло бы на
                    # максимальной скорости диска и забило бы сеть.
                    # Темп задаём по скорости перемотки: chunk_duration/speed.
                    speed = max(1.0, self._speed)
                    min_interval = self._chunk_duration / speed
                    if elapsed < min_interval:
                        return None

            # 3. Выбор чанка (приоритет/обычный/перемотка)
            chunk = self._get_next_chunk_locked()

            # 4. Если чанк выбран, фиксируем время
            if chunk is not None:
                self._last_chunk_ts = time.monotonic()
                logger.debug("[Scheduler] mode=%s, returned=%s", self._mode, chunk)

            return chunk

    def _get_next_chunk_locked(self) -> Optional[int]:
        if self._total_chunks == 0:
            logger.debug("[Scheduler] total_chunks=0, no chunks available")
            return None

        if self._mode == PlaybackMode.NORMAL:
            return self._next_normal()
        elif self._mode == PlaybackMode.SEEK:
            return self._next_seek()
        elif self._mode == PlaybackMode.FAST_FORWARD:
            return self._next_fast_forward()
        elif self._mode == PlaybackMode.PAUSE:
            return self._next_pause()
        logger.warning("[Scheduler] unknown mode %s", self._mode)
        return None

    # ------------------------------------------------------------------
    def _next_normal(self) -> Optional[int]:
        chunk = self._current_chunk
        while chunk in self._loaded_chunks and chunk < self._total_chunks:
            chunk += 1
        if chunk >= self._total_chunks:
            logger.debug("[Scheduler] _next_normal: all chunks loaded (current=%d, total=%d)",
                         self._current_chunk, self._total_chunks)
            return None
        self._loaded_chunks.add(chunk)
        self._current_chunk = chunk + 1
        logger.debug("[Scheduler] _next_normal: selected chunk %d, new current=%d", chunk, self._current_chunk)
        return chunk

    def _next_seek(self) -> Optional[int]:
        if not self._seek_priority_done:
            self._seek_priority_done = True
            chunk = self._target_chunk
            if chunk not in self._loaded_chunks and chunk < self._total_chunks:
                self._loaded_chunks.add(chunk)
                logger.debug("[Scheduler] _next_seek: priority chunk %d", chunk)
                return chunk
        for offset in [1, 2]:
            for direction in [-1, 1]:
                chunk = self._target_chunk + offset * direction
                if 0 <= chunk < self._total_chunks and chunk not in self._loaded_chunks:
                    self._loaded_chunks.add(chunk)
                    logger.debug("[Scheduler] _next_seek: neighbour chunk %d", chunk)
                    return chunk
        # переход в нормальный режим
        self._mode = PlaybackMode.NORMAL
        self._current_chunk = max(self._target_chunk + 1, 0)
        logger.info("[Scheduler] _next_seek: switching to NORMAL, current=%d", self._current_chunk)
        return self._next_normal()

    def _next_fast_forward(self) -> Optional[int]:
        stride = self._adaptive.get_fast_forward_stride(self._speed)
        chunk = self._current_chunk
        attempts = 0
        while attempts < self._total_chunks:
            if chunk in self._loaded_chunks:
                chunk += self._direction * stride
                attempts += 1
                continue
            if 0 <= chunk < self._total_chunks:
                self._loaded_chunks.add(chunk)
                self._current_chunk = chunk + self._direction * stride
                logger.debug("[Scheduler] _next_fast_forward: chunk %d (stride=%d)", chunk, stride)
                return chunk
            break
        logger.debug("[Scheduler] _next_fast_forward: no suitable chunk found")
        return None

    def _next_pause(self) -> Optional[int]:
        return self._next_normal()

    # ------------------------------------------------------------------
    def mark_chunk_failed(self, chunk_idx: int):
        with self._lock:
            self._loaded_chunks.discard(chunk_idx)
            logger.debug("[Scheduler] marked chunk %d as failed", chunk_idx)

    def reset(self):
        with self._lock:
            self._loaded_chunks.clear()
            self._mode = PlaybackMode.NORMAL
            self._current_chunk = 0
            self._total_chunks = 0
            self._loading_allowed = True
            self._last_chunk_ts = 0.0
            logger.info("[Scheduler] reset")

    @property
    def loaded_count(self) -> int:
        with self._lock:
            return len(self._loaded_chunks)
