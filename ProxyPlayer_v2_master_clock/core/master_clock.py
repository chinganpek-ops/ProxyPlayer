"""
master_clock.py – единый тактовый генератор на основе sounddevice.
Одна звуковая карта = Master Clock + вывод аудио.
Поддерживает две моно-дорожки (2 → левый канал, 3 → правый канал).
Управление: включение/отключение дорожек, общий mute.
Исправление: независимое микширование левого и правого каналов,
корректный подсчёт underruns.
"""

import threading
import logging
from collections import deque
import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class MasterClock:
    """
    Единый источник времени и аудиовыхода.
    Ведёт подсчёт семплов через callback звуковой карты.
    Раздельно микширует дорожку 2 (левый канал) и дорожку 3 (правый канал).
    """

    def __init__(self, sample_rate: int = 48000, buffer_size: int = 1024):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size

        # Счётчик воспроизведённых семплов (монотонный)
        self._samples_played = 0
        self._clock_lock = threading.Lock()

        # Раздельные очереди для дорожек 2 и 3 (float32)
        self._queue2 = deque()
        self._queue3 = deque()
        self._queue_lock = threading.Lock()

        # Управление дорожками
        self._track2_enabled = True   # дорожка 2 (левый канал)
        self._track3_enabled = True   # дорожка 3 (правый канал)
        self._muted = False           # общий mute

        # Задержка аудио относительно видео (в секундах)
        self.audio_delay = 0.0

        # Поток и управление
        self._stream: sd.OutputStream = None
        self._active = False

        # Статистика
        self._underruns = 0
        self._max_queue_len = 0

        # Флаг доступности устройства
        self._device_available = True

    # ------------------------------------------------------------------
    # Время
    # ------------------------------------------------------------------
    @property
    def samples_played(self) -> int:
        """Общее количество воспроизведённых семплов."""
        with self._clock_lock:
            return self._samples_played

    def get_time(self) -> float:
        """Текущее время в секундах от начала воспроизведения."""
        return self.samples_played / self.sample_rate

    def get_audio_clock(self) -> int:
        """Возвращает текущий audio_clock (в семплах) для синхронизации видео."""
        return self.samples_played + int(self.audio_delay * self.sample_rate)

    def set_clock(self, pts: int):
        """Принудительно устанавливает счётчик (при перемотке)."""
        with self._clock_lock:
            self._samples_played = pts
        logger.debug("MasterClock счётчик установлен на %d", pts)

    def reset(self):
        """Сбрасывает счётчик и очищает очереди (при остановке/перемотке)."""
        self.set_clock(0)
        self._underruns = 0
        logger.debug("MasterClock сброшен")

    # ------------------------------------------------------------------
    # Управление дорожками
    # ------------------------------------------------------------------
    def set_track_enabled(self, track_id: int, enabled: bool):
        """Включает или отключает дорожку (2 или 3)."""
        with self._queue_lock:
            if track_id == 2:
                self._track2_enabled = enabled
                if not enabled:
                    self._queue2.clear()
            elif track_id == 3:
                self._track3_enabled = enabled
                if not enabled:
                    self._queue3.clear()
        logger.debug("Дорожка %d %s", track_id, "включена" if enabled else "отключена")

    def set_muted(self, muted: bool):
        """Общий mute для всех дорожек."""
        self._muted = muted
        logger.debug("Mute: %s", "включен" if muted else "выключен")

    def is_track_enabled(self, track_id: int) -> bool:
        """Возвращает True, если дорожка включена."""
        if track_id == 2:
            return self._track2_enabled
        elif track_id == 3:
            return self._track3_enabled
        return False

    # ------------------------------------------------------------------
    # Аудиовыход
    # ------------------------------------------------------------------
    def push_audio(self, track_id: int, samples: np.ndarray):
        """
        Добавляет декодированные аудиосемплы (float64, моно) в очередь дорожки.
        Вызывается из AudioDecoderStage.
        """
        if not self._active:
            return
        with self._queue_lock:
            if track_id == 2 and self._track2_enabled:
                self._queue2.append(samples.astype(np.float32))
            elif track_id == 3 and self._track3_enabled:
                self._queue3.append(samples.astype(np.float32))
            self._max_queue_len = max(self._max_queue_len,
                                      len(self._queue2) + len(self._queue3))

    def get_queue_size(self) -> int:
        """Возвращает суммарный размер аудиоочередей."""
        with self._queue_lock:
            return len(self._queue2) + len(self._queue3)

    # ------------------------------------------------------------------
    # Callback звуковой карты
    # ------------------------------------------------------------------
    def _callback(self, outdata: np.ndarray, frames: int, time_info, status):
        if status:
            logger.debug("Sounddevice status: %s", status)

        with self._clock_lock:
            self._samples_played += frames

        outdata.fill(0.0)
        if self._muted:
            return

        # Микшируем каждый канал независимо
        if self._track2_enabled:
            self._mix_channel(outdata, 0, self._queue2, frames)
        if self._track3_enabled:
            self._mix_channel(outdata, 1, self._queue3, frames)

    def _mix_channel(self, outdata, channel_idx, queue, frames):
        """
        Заполняет канал channel_idx данными из очереди.
        Неиспользованный остаток возвращается в начало очереди.
        """
        written = 0
        while written < frames:
            with self._queue_lock:
                if queue:
                    chunk = queue.popleft()
                else:
                    break  # данных больше нет, остальное останется тишиной
            take = min(len(chunk), frames - written)
            outdata[written:written+take, channel_idx] = chunk[:take]
            if take < len(chunk):
                # Возвращаем оставшуюся часть обратно в начало очереди
                with self._queue_lock:
                    queue.appendleft(chunk[take:])
            written += take
        if written < frames:
            self._underruns += 1
        return written

    # ------------------------------------------------------------------
    # Управление потоком
    # ------------------------------------------------------------------
    def start(self):
        """Запускает звуковой поток."""
        if self._active:
            return
        try:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=2,          # стерео
                callback=self._callback,
                blocksize=self.buffer_size,
                latency='low',
                dtype='float32'
            )
            self._stream.start()
            self._active = True
            self._device_available = True
            logger.info("MasterClock запущен: %d Hz, stereo, buffer=%d",
                        self.sample_rate, self.buffer_size)
        except Exception as e:
            logger.error("Не удалось запустить аудиоустройство: %s", e)
            self._active = False
            self._device_available = False

    def stop(self):
        """Останавливает звуковой поток."""
        self._active = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        logger.info("MasterClock остановлен")

    def close(self):
        """Останавливает и очищает ресурсы."""
        self.stop()
        with self._queue_lock:
            self._queue2.clear()
            self._queue3.clear()
        logger.info("MasterClock закрыт")

    # ------------------------------------------------------------------
    # Диагностика
    # ------------------------------------------------------------------
    def get_stats(self) -> dict:
        return {
            'samples_played': self.samples_played,
            'queue_size': self.get_queue_size(),
            'max_queue_len': self._max_queue_len,
            'underruns': self._underruns,
            'audio_delay': self.audio_delay,
            'track2': self._track2_enabled,
            'track3': self._track3_enabled,
            'muted': self._muted,
            'device_available': self._device_available,
        }