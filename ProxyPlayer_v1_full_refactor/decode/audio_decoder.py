"""
audio_decoder.py – декодер AAC (моно) с праймингом для устранения ошибок холодного старта.
После создания отправляет минимальный валидный ADTS-фрейм для прогрева контекста.
"""

import logging
import numpy as np
import av

logger = logging.getLogger(__name__)

SAMPLES_PER_AAC_FRAME = 1024
DEFAULT_ASC = bytes([0x11, 0x88])
ADTS_SAMPLERATE_INDEX = 3   # 48000 Гц


def _create_adts_header(aac_frame_len: int) -> bytes:
    """Создаёт 7-байтный ADTS-заголовок для фрейма указанной длины."""
    length = aac_frame_len + 7
    header = bytes([
        0xFF, 0xF1,                                      # syncword + MPEG-4, layer=0, protection_absent=1
        (ADTS_SAMPLERATE_INDEX << 2) | 0x00,             # samplerate + no CRC
        ((1 << 6) | (length >> 11) & 0x3),               # channel config + high bits of length
        (length >> 3) & 0xFF,
        ((length & 0x7) << 5) | 0x1F,
        0xFC,
    ])
    return header


def _add_adts_header(raw_frame: bytes) -> bytes:
    """Добавляет ADTS-заголовок к raw AAC-фрейму."""
    if not raw_frame:
        return raw_frame
    return _create_adts_header(len(raw_frame)) + raw_frame


class AudioDecoder:
    """Декодер одного монофонического AAC-потока с защитой от холодного старта."""

    def __init__(self, asc: bytes = DEFAULT_ASC):
        if not asc or len(asc) < 2:
            raise ValueError(f"Некорректный AudioSpecificConfig: {asc.hex() if asc else 'None'}")
        self.codec = av.CodecContext.create('aac', 'r')
        self.codec.extradata = asc
        self.codec.sample_rate = 48000
        self._last_good_frame: np.ndarray | None = None
        self._error_count = 0

        # Прайминг: отправляем минимальный валидный ADTS-фрейм (1 байт тишины)
        self._prime_decoder()

        logger.debug(f"AudioDecoder создан с ASC {asc.hex()}")

    def _prime_decoder(self):
        """Прогревает декодер, подавая 4 кадра тишины (4096 сэмплов)."""
        priming_frame = _create_adts_header(1) + b'\x00'
        for _ in range(4):
            packet = av.Packet(priming_frame)
            try:
                self.codec.decode(packet)
            except Exception:
                pass

    def decode(self, aac_data: bytes) -> np.ndarray:
        if not aac_data:
            return self._fallback()

        packet = av.Packet(_add_adts_header(aac_data))
        try:
            frames = self.codec.decode(packet)
            if frames:
                frame = frames[0]
                arr = frame.to_ndarray().astype(np.float64)
                if frame.format.name == 's16':
                    arr /= 32768.0
                flat = arr.flatten()[:SAMPLES_PER_AAC_FRAME]
                if len(flat) < SAMPLES_PER_AAC_FRAME:
                    out = np.zeros(SAMPLES_PER_AAC_FRAME, dtype=np.float64)
                    out[:len(flat)] = flat
                    flat = out
                self._last_good_frame = flat.copy()
                self._error_count = 0
                return flat
        except Exception:
            # Редкие ошибки после прогрева – просто пропускаем
            pass

        return self._fallback()

    def _fallback(self) -> np.ndarray:
        self._error_count += 1
        if self._last_good_frame is not None:
            fade = max(0.0, 1.0 - self._error_count / 10.0)
            return self._last_good_frame * fade
        return np.zeros(SAMPLES_PER_AAC_FRAME, dtype=np.float64)

    def close(self):
        if self.codec:
            try:
                # Флешируем внутренние буферы FFmpeg
                self.codec.decode(None)
            except Exception:
                pass
            self.codec = None