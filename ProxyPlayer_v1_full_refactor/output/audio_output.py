"""
audio_output.py – стерео аудиовыход с опциональным debug_callback для мониторинга.
Дорожка 2 → левый канал (ведущая для audio_clock).
Дорожка 3 → правый канал.

Изменения:
- AudioOutput не запускается автоматически при создании.
- readData всегда возвращает полный буфер max_len, дополняя тишиной при нехватке данных.
  Это гарантирует непрерывный рост audio_clock.
- audio_clock (_samples_written) увеличивается на полный размер запроса samples_to_read.
- Добавлены защитные try/except для предотвращения падений при работе с аудио.
- Проверка доступности аудиоустройства перед запуском.
"""

import logging
import numpy as np
from PyQt5.QtCore import QIODevice, QObject, pyqtSignal
from PyQt5.QtMultimedia import QAudioOutput, QAudioFormat, QAudioDeviceInfo, QAudio

logger = logging.getLogger(__name__)

SAMPLE_RATE = 48000
CHANNELS = 2
SMOOTHING_WINDOW = 3


class _AudioIODevice(QIODevice):
    def __init__(self, buffers, parent=None):
        super().__init__(parent)
        self._buffers = buffers
        self._track_volumes = {2: 1.0, 3: 1.0}
        self._samples_written = 0
        self._open = False
        self._muted = False
        self._smoothing = True
        self._first_read = True
        self._debug_cb = None

    def set_debug_callback(self, cb):
        self._debug_cb = cb

    def open(self, mode=QIODevice.ReadOnly):
        try:
            if super().open(mode):
                self._open = True
                return True
        except Exception as e:
            logger.error(f"Ошибка открытия аудио-устройства: {e}")
        return False

    def close(self):
        self._open = False
        super().close()

    def isSequential(self) -> bool:
        return True

    def set_track_volumes(self, volumes: dict):
        self._track_volumes = volumes.copy()

    def set_muted(self, muted: bool):
        self._muted = muted

    def set_smoothing(self, enabled: bool):
        self._smoothing = enabled

    def reset(self):
        self._samples_written = 0

    def seek(self, pos):
        return False

    def bytesAvailable(self) -> int:
        if not self._open:
            return 0
        try:
            avail = min(
                self._buffers.buffers[t].available_read if t in self._buffers.buffers else 48000
                for t in (2, 3)
            )
            return avail * CHANNELS * 4
        except Exception as e:
            logger.debug(f"Ошибка в bytesAvailable: {e}")
            return 0

    def readData(self, max_len: int) -> bytes:
        if not self._open:
            return b''

        bytes_per_sample = 4
        samples_to_read = max_len // (CHANNELS * bytes_per_sample)
        if samples_to_read == 0:
            return b''

        try:
            # Читаем столько, сколько запрошено, из обоих каналов
            data_left = self._buffers.read(2, samples_to_read)
            data_right = self._buffers.read(3, samples_to_read)
        except Exception as e:
            logger.warning(f"Ошибка чтения аудиоданных: {e}")
            data_left = np.zeros(samples_to_read, dtype=np.float64)
            data_right = np.zeros(samples_to_read, dtype=np.float64)

        # Отладка: фактический PTS начала прочитанного аудиоблока
        if self._debug_cb:
            try:
                pts = self._buffers.buffers[2].read_pos - len(data_left)
                self._debug_cb("AUDIO_OUT", f"pts={pts} aclock={self._samples_written}")
            except Exception:
                pass

        # Всегда дополняем до полного размера запроса тишиной
        if len(data_left) < samples_to_read:
            data_left = np.pad(data_left, (0, samples_to_read - len(data_left)), mode='constant')
        if len(data_right) < samples_to_read:
            data_right = np.pad(data_right, (0, samples_to_read - len(data_right)), mode='constant')

        out = np.zeros((samples_to_read, CHANNELS), dtype=np.float32)
        gain_left = self._track_volumes.get(2, 0.0)
        gain_right = self._track_volumes.get(3, 0.0)

        if not self._muted:
            out[:, 0] = (data_left * gain_left).astype(np.float32)
            out[:, 1] = (data_right * gain_right).astype(np.float32)

        if self._smoothing and not self._muted and samples_to_read > 0:
            try:
                kernel = np.ones(SMOOTHING_WINDOW, dtype=np.float32) / SMOOTHING_WINDOW
                for ch in range(CHANNELS):
                    out[:, ch] = np.convolve(out[:, ch], kernel, mode='same')
            except Exception as e:
                logger.debug(f"Ошибка сглаживания: {e}")

        if self._first_read and self._debug_cb:
            try:
                self._debug_cb("AUDIO_FIRST_READ", f"samples={samples_to_read}")
                self._first_read = False
            except Exception:
                pass

        interleaved = np.ascontiguousarray(out).tobytes()

        # audio_clock увеличивается на полный размер запроса – непрерывный ход
        self._samples_written += samples_to_read
        return interleaved


class AudioOutput(QObject):
    audio_clock_changed = pyqtSignal(int)

    def __init__(self, audio_buffers, parent=None, debug_callback=None):
        super().__init__(parent)
        self._buffers = audio_buffers
        self._device = _AudioIODevice(audio_buffers, self)
        if debug_callback:
            self._device.set_debug_callback(debug_callback)
        self._output = None
        self._active = False
        self._track_volumes = {2: 1.0, 3: 1.0}
        self._debug_cb = debug_callback
        self._device_open_failed = False

    def _start_output(self):
        """Создаёт и запускает QAudioOutput с защитой от ошибок."""
        if self._device_open_failed:
            logger.warning("Аудиоустройство недоступно, повторный запуск невозможен")
            return False

        fmt = QAudioFormat()
        fmt.setSampleRate(SAMPLE_RATE)
        fmt.setChannelCount(CHANNELS)
        fmt.setSampleSize(32)
        fmt.setCodec("audio/pcm")
        fmt.setByteOrder(QAudioFormat.LittleEndian)
        fmt.setSampleType(QAudioFormat.Float)

        info = QAudioDeviceInfo.defaultOutputDevice()
        if info.isNull():
            logger.warning("Аудиоустройство по умолчанию не найдено")
            self._device_open_failed = True
            return False

        if not info.isFormatSupported(fmt):
            logger.warning("Формат не поддерживается, использую ближайший.")
            fmt = info.nearestFormat(fmt)

        try:
            if self._output:
                self._output.stop()
                self._output.disconnect()
                self._output.deleteLater()
                self._output = None

            self._device.open(QIODevice.ReadOnly)
            self._output = QAudioOutput(fmt, self)
            self._output.setNotifyInterval(10)
            self._output.stateChanged.connect(self._on_state_changed)
            self._output.notify.connect(self._on_notify)
            self._output.start(self._device)
            self._active = True
            self._device_open_failed = False

            if self._debug_cb:
                self._debug_cb("AUDIO_OUTPUT_START", "AudioOutput started successfully")
            logger.info("Аудиовыход запущен")
            return True
        except Exception as e:
            logger.error(f"Ошибка при запуске аудиовыхода: {e}", exc_info=True)
            self._device_open_failed = True
            self._active = False
            if self._output:
                try:
                    self._output.stop()
                    self._output.disconnect()
                except Exception:
                    pass
                self._output = None
            return False

    def set_active_tracks(self, tracks: list):
        volumes = {2: 0.0, 3: 0.0}
        if 2 in tracks:
            volumes[2] = 1.0
        if 3 in tracks:
            volumes[3] = 1.0
        self._track_volumes = volumes
        if self._device:
            self._device.set_track_volumes(volumes)

    def set_muted(self, muted: bool):
        if self._device:
            self._device.set_muted(muted)

    def set_smoothing(self, enabled: bool):
        if self._device:
            self._device.set_smoothing(enabled)

    def start(self):
        if self._active:
            return
        if self._device_open_failed:
            logger.warning("Аудиоустройство недоступно, пропускаем start")
            return
        self._start_output()

    def stop(self):
        try:
            if self._output:
                self._output.stop()
                self._output.disconnect()
                self._output.deleteLater()
                self._output = None
            self._device.close()
            self._active = False
        except Exception as e:
            logger.warning(f"Ошибка при остановке аудиовыхода: {e}")

    def set_volume(self, volume: float):
        if self._output:
            try:
                self._output.setVolume(max(0.0, min(1.0, volume)))
            except Exception as e:
                logger.debug(f"Ошибка установки громкости: {e}")

    def current_clock(self) -> int:
        return self._device._samples_written if self._device else 0

    def reset_clock(self, position: int = 0):
        if self._device:
            self._device._samples_written = position
        if self._buffers:
            try:
                self._buffers.reset_all_read_to(position)
            except Exception as e:
                logger.debug(f"Ошибка сброса буферов: {e}")

    def _on_state_changed(self, state):
        try:
            if state == QAudio.IdleState and self._active:
                logger.debug("Аудиовыход в Idle")
        except Exception as e:
            logger.debug(f"Ошибка в _on_state_changed: {e}")

    def _on_notify(self):
        try:
            if self._output and self._active:
                self.audio_clock_changed.emit(self._device._samples_written)
        except Exception as e:
            logger.debug(f"Ошибка в _on_notify: {e}")