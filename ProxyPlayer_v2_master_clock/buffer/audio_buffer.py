"""
audio_buffer.py – многодорожечный кольцевой буфер аудиосэмплов (float64).
Поддерживает независимые дорожки 2..17. Потокобезопасен.
Добавлен неблокирующий метод try_write с контролем максимального опережения.
Исправление: при пустом буфере (write_pos==0) разрешается запись любых данных
для инициализации позиций.
"""

import threading
import logging
import numpy as np

logger = logging.getLogger(__name__)


class AudioRingBuffer:
    """Один кольцевой буфер для одной моно-дорожки (1 канал)."""

    def __init__(self, capacity_samples: int = 480000, max_ahead_samples: int = 48000):
        if capacity_samples < 1024:
            raise ValueError("capacity_samples должен быть >= 1024")
        self.capacity = capacity_samples
        self.max_ahead_samples = max_ahead_samples
        self._buffer = np.zeros(capacity_samples, dtype=np.float64)
        self._read_pos = 0
        self._write_pos = 0
        self._total_written = 0
        self._lock = threading.Lock()
        self._data_available = threading.Condition(self._lock)
        self._debug_cb = None

    def set_debug_callback(self, cb):
        self._debug_cb = cb

    def _send_debug(self, event_type, details=""):
        if self._debug_cb:
            try:
                self._debug_cb(event_type, details)
            except Exception:
                pass

    def write_block(self, data: np.ndarray, start_sample: int):
        """
        Записывает блок 1D float64 в буфер (блокирующий режим).
        Сохраняет обратную совместимость.
        """
        if data.ndim != 1:
            raise ValueError("data должна быть одномерным массивом")
        n_samples = len(data)
        if n_samples == 0:
            return

        with self._lock:
            read_pos = self._read_pos
            if start_sample < read_pos:
                overlap = read_pos - start_sample
                if overlap >= n_samples:
                    self._send_debug("AUDIO_DISCARD", f"overlap: start={start_sample} read={read_pos}")
                    return
                data = data[overlap:]
                start_sample = read_pos
                n_samples = len(data)

            if n_samples == 0:
                return

            while start_sample + n_samples > self._read_pos + self.capacity:
                self._send_debug("AUDIO_BUFFER_FULL",
                                 f"start={start_sample} n={n_samples} read={self._read_pos}")
                if not self._data_available.wait(timeout=1.0):
                    continue

            self._write_data(data, start_sample, n_samples)

    def try_write(self, data: np.ndarray, start_sample: int) -> bool:
        """
        Пытается записать блок без бесконечной блокировки.
        Возвращает True, если запись выполнена, False – если буфер переполнен
        и превышено max_ahead_samples.
        """
        if data.ndim != 1:
            raise ValueError("data должна быть одномерным массивом")
        n_samples = len(data)
        if n_samples == 0:
            return True

        with self._lock:
            read_pos = self._read_pos

            # Если буфер полностью пуст (write_pos == 0), разрешаем запись любых данных,
            # чтобы инициализировать позицию записи и сократить расхождение.
            if self._write_pos == 0:
                # Устаревшие данные (start_sample < read_pos) не записываем,
                # но подтягиваем write_pos к read_pos для предотвращения вечного отставания.
                if start_sample < read_pos:
                    overlap = read_pos - start_sample
                    if overlap >= n_samples:
                        # Полностью устарело – просто сдвигаем write_pos
                        self._write_pos = max(self._write_pos, read_pos)
                        return False
                    # Частично устарело – обрезаем
                    data = data[overlap:]
                    start_sample = read_pos
                    n_samples = len(data)
                if n_samples > 0:
                    self._write_data(data, start_sample, n_samples)
                    return True
                else:
                    return False

            # Стандартная логика для непустого буфера
            if start_sample < read_pos:
                overlap = read_pos - start_sample
                if overlap >= n_samples:
                    # Полностью устарело – подтягиваем write_pos к read_pos
                    self._write_pos = max(self._write_pos, read_pos)
                    return False
                data = data[overlap:]
                start_sample = read_pos
                n_samples = len(data)

            if n_samples == 0:
                return True

            max_allowed = read_pos + self.capacity + self.max_ahead_samples
            if start_sample + n_samples > max_allowed:
                self._send_debug("AUDIO_DROPPED",
                                 f"start={start_sample} n={n_samples} read={read_pos} max={max_allowed}")
                return False

            if start_sample + n_samples > read_pos + self.capacity:
                if not self._data_available.wait(timeout=0.1):
                    if start_sample + n_samples > self._read_pos + self.capacity:
                        self._send_debug("AUDIO_TRY_WRITE_FULL",
                                         f"start={start_sample} n={n_samples} read={self._read_pos}")
                        return False

            self._write_data(data, start_sample, n_samples)
            return True

    def _write_data(self, data: np.ndarray, start_sample: int, n_samples: int):
        """Внутренняя операция копирования данных в кольцевой массив."""
        if self._debug_cb and np.max(np.abs(data)) < 1e-10:
            self._send_debug("AUDIO_SILENCE_WRITE", f"start={start_sample} samples={n_samples}")

        idx_start = start_sample % self.capacity
        if idx_start + n_samples <= self.capacity:
            self._buffer[idx_start:idx_start + n_samples] = data
        else:
            first_part = self.capacity - idx_start
            self._buffer[idx_start:] = data[:first_part]
            second_part = n_samples - first_part
            self._buffer[:second_part] = data[first_part:]

        self._write_pos = max(self._write_pos, start_sample + n_samples)
        self._total_written += n_samples
        self._data_available.notify_all()

    def read(self, num_samples: int, timeout: float = 0.1) -> np.ndarray:
        """Читает num_samples сэмплов. Ждёт до timeout сек, если данных мало."""
        if num_samples <= 0:
            return np.empty(0, dtype=np.float64)

        with self._lock:
            while self._write_pos - self._read_pos < num_samples:
                if not self._data_available.wait(timeout):
                    break

            available = max(0, self._write_pos - self._read_pos)
            to_read = min(num_samples, available)
            if to_read == 0:
                return np.empty(0, dtype=np.float64)

            out = np.empty(to_read, dtype=np.float64)
            idx_start = self._read_pos % self.capacity
            if idx_start + to_read <= self.capacity:
                out[:] = self._buffer[idx_start:idx_start + to_read]
            else:
                first_part = self.capacity - idx_start
                out[:first_part] = self._buffer[idx_start:]
                second_part = to_read - first_part
                out[first_part:to_read] = self._buffer[:second_part]
            self._read_pos += to_read

            if self._debug_cb and np.max(np.abs(out)) < 1e-10:
                self._send_debug("AUDIO_SILENCE_READ", f"pos={self._read_pos - to_read} samples={to_read}")
            return out

    def reset_read_to(self, position: int):
        with self._lock:
            self._read_pos = position
            self._data_available.notify_all()

    def clear(self):
        with self._lock:
            self._read_pos = 0
            self._write_pos = 0
            self._total_written = 0
            self._buffer.fill(0.0)
            self._data_available.notify_all()

    @property
    def available_read(self) -> int:
        with self._lock:
            return max(0, self._write_pos - self._read_pos)

    @property
    def read_pos(self) -> int:
        with self._lock:
            return self._read_pos

    @property
    def write_pos(self) -> int:
        with self._lock:
            return self._write_pos

    def get_stats(self) -> dict:
        with self._lock:
            return {
                'read_pos': self._read_pos,
                'write_pos': self._write_pos,
                'available': max(0, self._write_pos - self._read_pos),
                'total_written': self._total_written,
                'capacity': self.capacity,
            }


class MultiTrackAudioBuffer:
    """Управляет независимыми буферами для каждой дорожки (2..17)."""

    def __init__(self, capacity_samples: int = 480000):
        self.capacity = capacity_samples
        self.buffers = {track_id: AudioRingBuffer(capacity_samples)
                        for track_id in range(2, 18)}
        logger.info(f"Созданы аудиобуферы для дорожек 2..17 с ёмкостью {capacity_samples} сэмплов")

    def set_debug_callback(self, cb):
        for buf in self.buffers.values():
            buf.set_debug_callback(cb)

    def write(self, track_id: int, data: np.ndarray, start_sample: int):
        if track_id not in self.buffers:
            logger.warning(f"Попытка записи в несуществующую дорожку {track_id}")
            return
        self.buffers[track_id].write_block(data, start_sample)

    def try_write(self, track_id: int, data: np.ndarray, start_sample: int) -> bool:
        """Неблокирующая запись в буфер дорожки. Возвращает True при успехе."""
        if track_id not in self.buffers:
            logger.warning(f"Попытка записи в несуществующую дорожку {track_id}")
            return False
        return self.buffers[track_id].try_write(data, start_sample)

    def read(self, track_id: int, num_samples: int, timeout: float = 0.1) -> np.ndarray:
        if track_id not in self.buffers:
            logger.warning(f"Попытка чтения из несуществующей дорожки {track_id}")
            return np.empty(0, dtype=np.float64)
        return self.buffers[track_id].read(num_samples, timeout)

    def clear_all(self):
        for buf in self.buffers.values():
            buf.clear()
        logger.info("Все аудиобуферы очищены")

    def reset_all_read_to(self, position: int):
        for buf in self.buffers.values():
            buf.reset_read_to(position)
        logger.debug(f"Указатели чтения всех аудиобуферов сброшены на {position}")

    def reset_track_read_to(self, track_id: int, position: int):
        if track_id in self.buffers:
            self.buffers[track_id].reset_read_to(position)
            logger.debug(f"Указатель чтения дорожки {track_id} сброшен на {position}")

    def get_stats(self, track_id: int = None) -> dict:
        if track_id is not None:
            if track_id in self.buffers:
                return self.buffers[track_id].get_stats()
            else:
                return {}
        return {tid: buf.get_stats() for tid, buf in self.buffers.items()}