"""
sync_manager.py – менеджер синхронизации аудио/видео для ProxyPlayer v2.
Работает напрямую с audio_clock от MasterClock.
Дрейф-коррекция и зависимость от аудиобуферов удалены.
"""

import time
import logging
from typing import Optional

import numpy as np

from buffer.frame_buffer import FrameRingBuffer
from config.timebase import (
    SAMPLES_PER_VIDEO_FRAME,
)

logger = logging.getLogger(__name__)

MAX_VIDEO_LAG = SAMPLES_PER_VIDEO_FRAME * 10      # 19200 сэмплов (400 мс)
FUTURE_HORIZON = SAMPLES_PER_VIDEO_FRAME // 2      # 960 сэмплов


class SyncManager:
    """
    Управляет синхронизацией видео с аудио.
    Видео ведомое, аудио ведущее (через MasterClock).
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Основной метод выбора кадра
    # ------------------------------------------------------------------
    def get_display_frame(
        self,
        video_buffer: FrameRingBuffer,
        audio_clock: int,
        playing: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Возвращает кадр для отображения на основе текущего audio_clock.
        Автоматически отбрасывает устаревшие кадры.
        """
        if not playing:
            return video_buffer.get_keep_last()

        current_clock = audio_clock

        # Удаляем кадры, отстающие более чем на MAX_VIDEO_LAG
        video_buffer.drop_until(int(current_clock) - MAX_VIDEO_LAG)

        first = video_buffer.peek_first()
        if first is not None:
            pts, frame = first
            delta = pts - current_clock

            if delta < -MAX_VIDEO_LAG:
                # Безнадёжно устарел – пропускаем
                video_buffer.advance()
                return self.get_display_frame(
                    video_buffer, audio_clock, playing
                )
            else:
                # Показываем кадр
                video_buffer.update_keep_last(frame, pts)
                video_buffer.advance()
                return frame
        else:
            return video_buffer.get_keep_last()

    # ------------------------------------------------------------------
    # Заглушка для совместимости со старым кодом
    # ------------------------------------------------------------------
    def reset_drift(self):
        """Больше не используется, оставлен для совместимости."""
        pass