"""
decoder.py – программный и аппаратный (NVIDIA CUDA/CUVID) декодер H.264.
Поддерживает многопоточность (CPU), zero-copy, опциональный пропуск B-кадров
и автоматический выбор между CPU и GPU. Кэширует проверку доступности GPU.
Выдаёт кадры в формате RGB24. Основной метод декодирования – decode_sample (AVCC).
Версия production: детальное логирование, явные ошибки вместо скрытых fallback-путей.
Добавлены защитные проверки и обработка исключений для предотвращения падений.
"""

import logging
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import av

logger = logging.getLogger(__name__)

NAL_TYPE_SEI = 6
NAL_TYPE_AUD = 9

# Глобальный кэш доступности GPU
_gpu_available_cache: Optional[Tuple[bool, str]] = None


class Decoder:
    """Декодер H.264, инициализируемый avcC (SPS/PPS)."""

    def __init__(
        self,
        avcc_data: bytes,
        mp4_path: Optional[Path] = None,
        thread_type: str = "AUTO",
        thread_count: int = 0,
        skip_frame: bool = False,
        gpu_mode: str = "off",
    ):
        if not avcc_data:
            raise ValueError("avcc_data не может быть пустым")
        self._avcc = avcc_data
        self.gpu_mode = gpu_mode
        self._codec_name = "h264"
        self._hw_device_ctx = None
        self.codec = None  # будет инициализирован позже
        self._closed = False

        # --- Попытка использовать GPU ---
        if gpu_mode in ("on", "auto"):
            gpu_available, reason = self._is_gpu_available_cached()
            if gpu_available:
                try:
                    self._init_gpu_decoder()
                    self._codec_name = "h264_cuvid"
                    logger.info("GPU-декодер h264_cuvid успешно инициализирован")
                except Exception as e:
                    logger.error(f"Не удалось инициализировать GPU-декодер: {e}. Причина: {reason}")
                    if gpu_mode == "on":
                        raise RuntimeError(
                            f"GPU-декодер запрошен, но не может быть инициализирован: {reason}"
                        )
                    # fallback to CPU для 'auto' – только после явного предупреждения
                    logger.warning("Переключаюсь на программный декодер (auto mode)")
            elif gpu_mode == "on":
                raise RuntimeError(f"GPU-декодер запрошен, но h264_cuvid не найден: {reason}")
            else:
                logger.info(f"GPU-декодер не найден ({reason}), fallback на программный")

        # --- Создание контекста кодека ---
        try:
            self.codec = av.CodecContext.create(self._codec_name, "r")
        except Exception as e:
            raise RuntimeError(f"Не удалось создать кодек {self._codec_name}: {e}")

        # Передача SPS/PPS
        self.codec.extradata = avcc_data

        # --- Применение параметров (только для CPU) ---
        if self._codec_name == "h264":
            if thread_type:
                self.codec.thread_type = thread_type
            self.codec.thread_count = thread_count
            logger.debug(f"CPU-декодер: thread_type={thread_type}, thread_count={thread_count}")
        if skip_frame:
            self.codec.skip_frame = "BIDIR"
            logger.info("Пропуск B-кадров включён")

    # ------------------------------------------------------------------
    def _init_gpu_decoder(self):
        """Инициализация аппаратного контекста CUDA."""
        try:
            from av.util import HWDeviceType
            device_type = HWDeviceType('cuda')
            self._hw_device_ctx = av.util.HWDeviceContext.create(device_type)
            self.codec.hw_device_ctx = self._hw_device_ctx
            logger.debug("HWDeviceContext для CUDA создан")
        except ImportError:
            raise RuntimeError("Модуль av.util не поддерживает HWDeviceType")
        except Exception as e:
            raise RuntimeError(f"Ошибка создания HWDeviceContext: {e}")

    @staticmethod
    def _is_gpu_available_cached() -> Tuple[bool, str]:
        """Проверка доступности GPU-декодера с кэшированием результата."""
        global _gpu_available_cache
        if _gpu_available_cache is not None:
            return _gpu_available_cache
        try:
            codec = av.codec.Codec('h264_cuvid', 'r')
            try:
                subprocess.run(['nvidia-smi'], capture_output=True, check=True, timeout=10)
                _gpu_available_cache = (True, "Доступен")
                logger.debug("GPU NVIDIA доступен")
            except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
                _gpu_available_cache = (False, "Драйвер NVIDIA не обнаружен")
                logger.debug("nvidia-smi не обнаружен")
        except av.codec.UnknownCodecError:
            _gpu_available_cache = (False, "Кодек h264_cuvid не найден в сборке FFmpeg")
            logger.debug("h264_cuvid отсутствует в FFmpeg")
        except Exception as e:
            _gpu_available_cache = (False, str(e))
            logger.error(f"Неожиданная ошибка при проверке GPU: {e}")
        return _gpu_available_cache

    # ------------------------------------------------------------------
    def decode_sample(self, data: bytes) -> List[np.ndarray]:
        """
        Основной метод декодирования. Принимает AVCC-данные (после filter_avcc).
        Возвращает список кадров RGB24.
        При ошибке возвращает пустой список и логирует предупреждение.
        """
        if self._closed or self.codec is None:
            logger.debug("decode_sample: декодер закрыт или не инициализирован")
            return []
        if not data:
            logger.debug("decode_sample: пустые данные")
            return []
        try:
            packet = av.Packet(memoryview(data))
            return self._decode_packet_frames(packet)
        except Exception as e:
            logger.error(f"Ошибка при создании пакета или декодировании: {e}", exc_info=True)
            return []

    def decode_sample_with_pts(self, data: bytes, base_pts: int = 0) -> List[Tuple[np.ndarray, int]]:
        """
        Декодирует AVCC-данные и возвращает кадры с реальными PTS.
        
        Args:
            data: AVCC-данные (после filter_avcc)
            base_pts: базовый PTS для случая, если frame.pts отсутствует
        
        Returns:
            Список кортежей (кадр RGB24, pts в аудиосэмплах 48 кГц)
        """
        if self._closed or self.codec is None:
            logger.debug("decode_sample_with_pts: декодер закрыт или не инициализирован")
            return []
        if not data:
            logger.debug("decode_sample_with_pts: пустые данные")
            return []
        try:
            packet = av.Packet(memoryview(data))
            packet.pts = base_pts
            frames = self.codec.decode(packet)
            result = []
            for i, frame in enumerate(frames):
                try:
                    img = np.ascontiguousarray(frame.to_ndarray(format="rgb24"))
                    # Пытаемся получить реальный PTS из кадра
                    if frame.pts is not None:
                        # Пересчитываем PTS в сэмплы 48 кГц
                        time_base = float(self.codec.time_base.numerator) / self.codec.time_base.denominator
                        pts = int(frame.pts * time_base * 48000)
                    else:
                        pts = base_pts + i * 1920  # fallback: последовательные PTS
                    result.append((img, pts))
                except Exception as e:
                    logger.debug(f"Ошибка преобразования кадра {i}: {e}")
                    continue
            return result
        except Exception as e:
            logger.error(f"Ошибка декодирования с PTS: {e}", exc_info=True)
            return []

    def _decode_packet_frames(self, packet: av.Packet) -> List[np.ndarray]:
        if self.codec is None:
            return []
        frames = []
        try:
            for frame in self.codec.decode(packet):
                try:
                    img = np.ascontiguousarray(frame.to_ndarray(format="rgb24"))
                    frames.append(img)
                except ValueError as e:
                    logger.debug(f"Ошибка преобразования кадра: {e}")
                except Exception as e:
                    logger.debug(f"Неожиданная ошибка при преобразовании кадра: {e}")
        except ValueError as e:
            logger.debug(f"Ошибка декодирования кадра: {e}")
        except Exception as e:
            logger.error(f"Неожиданная ошибка декодирования: {e}", exc_info=True)
        return frames

    def filter_avcc(self, data: bytes) -> bytes:
        """
        Убирает NAL-юниты SEI и AUD из AVCC-потока.
        Возвращает очищенные данные.
        При ошибке возвращает исходные данные (fallback) и логирует предупреждение.
        """
        if self._closed:
            logger.debug("filter_avcc: декодер закрыт")
            return b''
        if not data:
            return b''
        # Для коротких блоков используем упрощённый, но надёжный метод
        if len(data) < 1024:
            return self._filter_avcc_fast(data)
        try:
            arr = np.frombuffer(data, dtype=np.uint8)
            pos = 0
            parts = []
            while pos + 4 <= len(data):
                size = int.from_bytes(data[pos:pos+4], 'big')
                if size == 0 or pos + 4 + size > len(data):
                    logger.debug(f"Некорректный размер NAL-юнита: size={size}, pos={pos}")
                    break
                nal_type = arr[pos + 4] & 0x1F
                if nal_type not in (NAL_TYPE_SEI, NAL_TYPE_AUD):
                    parts.append(data[pos:pos+4+size])
                pos += 4 + size
            return b''.join(parts)
        except Exception as e:
            logger.warning(f"Ошибка при фильтрации AVCC через NumPy, использую резервный метод: {e}")
            return self._filter_avcc_fast(data)

    @staticmethod
    def _filter_avcc_fast(data: bytes) -> bytes:
        """Побайтовый фильтр AVCC без NumPy, всегда корректен."""
        result = bytearray()
        pos = 0
        while pos + 4 <= len(data):
            size = int.from_bytes(data[pos:pos+4], 'big')
            if pos + 4 + size > len(data):
                break
            if size > 0:
                nal_type = data[pos + 4] & 0x1F
                if nal_type not in (NAL_TYPE_SEI, NAL_TYPE_AUD):
                    result.extend(data[pos:pos+4+size])
            pos += 4 + size
        return bytes(result)

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.codec:
            try:
                # Извлекаем оставшиеся кадры из внутренних буферов FFmpeg
                remaining = self.codec.decode(None)
                if remaining:
                    logger.debug(f"Флешировано {len(remaining)} кадров при закрытии декодера")
            except Exception as e:
                logger.debug(f"Ошибка при флеше декодера: {e}")
            finally:
                try:
                    self.codec = None
                except Exception:
                    pass
        self._hw_device_ctx = None
        logger.debug("Декодер закрыт")