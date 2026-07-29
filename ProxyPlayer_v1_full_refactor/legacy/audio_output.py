"""
audio_output.py – стерео аудиовыход с опциональным debug_callback для мониторинга.
Дорожка 2 → левый канал (ведущая для audio_clock).
Дорожка 3 → правый канал.

Изменения:
- AudioOutput не запускается автоматически при создании.
- readData всегда возвращает полный буфер max_len, дополняя тишиной при нехватке данных.
  Это гарантирует непрерывный рост audio_clock.
- audio_clock (_samples_written) увеличивается на полный размер запроса samples_to_read.
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

    def open(self, mode=QIODevice.ReadOnly):
        if super().open(mode):
            self._open = True
            return True
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
        avail = min(
            self._buffers.buffers[t].available_read if t in self._buffers.buffers else 48000
            for t in (2, 3)
        )
        return avail * CHANNELS * 4

    def readData(self, max_len: int) -> bytes:
        if not self._open:
            return b''

        bytes_per_sample = 4
        samples_to_read = max_len // (CHANNELS * bytes_per_sample)
        if samples_to_read == 0:
            return b''

        # Читаем столько, сколько запрошено, из обоих каналов
        data_left = self._buffers.read(2, samples_to_read)
        data_right = self._buffers.read(3, samples_to_read)

        # Отладка: фактический PTS начала прочитанного аудиоблока
        if self.parent() and hasattr(self.parent(), '_debug_cb'):
            pts = self._buffers.buffers[2].read_pos - len(data_left)
            self.parent()._debug_cb("AUDIO_OUT", f"pts={pts} aclock={self._samples_written}")

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

        if self._smoothing and not self._muted:
            kernel = np.ones(SMOOTHING_WINDOW, dtype=np.float32) / SMOOTHING_WINDOW
            for ch in range(CHANNELS):
                out[:, ch] = np.convolve(out[:, ch], kernel, mode='same')

        if self._first_read and self.parent() and hasattr(self.parent(), '_debug_cb'):
            self.parent()._debug_cb("AUDIO_FIRST_READ", f"samples={samples_to_read}")
            self._first_read = False

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
        self._output = None
        self._active = False
        self._track_volumes = {2: 1.0, 3: 1.0}
        self._debug_cb = debug_callback
        # ИЗМЕНЕНИЕ: не запускаем аудиовыход автоматически

    def _start_output(self):
        fmt = QAudioFormat()
        fmt.setSampleRate(SAMPLE_RATE)
        fmt.setChannelCount(CHANNELS)
        fmt.setSampleSize(32)
        fmt.setCodec("audio/pcm")
        fmt.setByteOrder(QAudioFormat.LittleEndian)
        fmt.setSampleType(QAudioFormat.Float)

        info = QAudioDeviceInfo.defaultOutputDevice()
        if not info.isFormatSupported(fmt):
            logger.warning("Формат не поддерживается, использую ближайший.")
            fmt = info.nearestFormat(fmt)

        if self._output:
            self._output.stop()
            self._output.disconnect()
            self._output = None

        self._device.open(QIODevice.ReadOnly)
        self._output = QAudioOutput(fmt, self)
        self._output.setNotifyInterval(10)
        self._output.stateChanged.connect(self._on_state_changed)
        self._output.notify.connect(self._on_notify)
        self._output.start(self._device)
        self._active = True

        if self._debug_cb:
            self._debug_cb("AUDIO_OUTPUT_START", "AudioOutput started")

    def set_active_tracks(self, tracks: list):
        volumes = {2: 0.0, 3: 0.0}
        if 2 in tracks:
            volumes[2] = 1.0
        if 3 in tracks:
            volumes[3] = 1.0
        self._track_volumes = volumes
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
        self._start_output()

    def stop(self):
        if self._output:
            self._output.stop()
            self._output.disconnect()
            self._output = None
        self._device.close()
        self._active = False

    def set_volume(self, volume: float):
        if self._output:
            self._output.setVolume(volume)

    def current_clock(self) -> int:
        return self._device._samples_written if self._device else 0

    def reset_clock(self, position: int = 0):
        if self._device:
            self._device._samples_written = position
        if self._buffers:
            self._buffers.reset_all_read_to(position)

    def _on_state_changed(self, state):
        if state == QAudio.IdleState and self._active:
            logger.debug("Аудиовыход в Idle")

    def _on_notify(self):
        if self._output and self._active:
            self.audio_clock_changed.emit(self._device._samples_written)