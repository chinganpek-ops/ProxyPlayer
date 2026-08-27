"""
sync_manager.py – менеджер синхронизации аудио/видео для ProxyPlayer v2.
Работает напрямую с audio_clock от MasterClock.
Дрейф-коррекция удалена.

Добавлено логирование синхронизации в sync_monitor.log через SyncMonitor.
Исправлено: добавлена проверка FUTURE_HORIZON для предотвращения забегания кадров вперёд.
"""

import logging
from typing import Optional

import numpy as np

from buffer.frame_buffer import FrameRingBuffer
from config.timebase import SAMPLES_PER_VIDEO_FRAME
from utils.sync_logger import sync_monitor_logger

logger = logging.getLogger(__name__)

MAX_VIDEO_LAG = SAMPLES_PER_VIDEO_FRAME * 10      # 19200 сэмплов (400 мс)
FUTURE_HORIZON = SAMPLES_PER_VIDEO_FRAME // 2      # 960 сэмплов


class SyncManager:
    """
    Управляет синхронизацией видео с аудио.
    Видео ведомое, аудио ведущее (через MasterClock).
    """

    def __init__(self):
        self._seek_pts_reference = 0

    def set_seek_reference(self, pts: int):
        """Устанавливает минимальный допустимый PTS для выдачи кадров."""
        self._seek_pts_reference = pts
        sync_monitor_logger.debug(f"SYNC_SEEK_REF pts={pts}")

    def get_display_frame(
        self,
        video_buffer: FrameRingBuffer,
        audio_clock: int,
        playing: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Возвращает кадр для отображения на основе текущего audio_clock.
        Автоматически отбрасывает устаревшие кадры и кадры, не соответствующие
        новому месту после seek.
        """
        if not playing:
            return video_buffer.get_keep_last()

        current_clock = audio_clock

        # Удаляем кадры, отстающие более чем на MAX_VIDEO_LAG
        video_buffer.drop_until(int(current_clock) - MAX_VIDEO_LAG)

        # Отбрасываем кадры, оставшиеся до новой точки отсчёта после seek
        while True:
            first_check = video_buffer.peek_first()
            if first_check is None or first_check[0] >= self._seek_pts_reference:
                break
            sync_monitor_logger.debug(f"SYNC_VIDEO_SEEK_FILTER pts={first_check[0]}")
            video_buffer.advance()

        first = video_buffer.peek_first()
        if first is not None:
            pts, frame = first
            delta = pts - current_clock

            if delta < -MAX_VIDEO_LAG:
                # Безнадёжно устарел – пропускаем
                video_buffer.advance()
                sync_monitor_logger.debug(
                    f"SYNC_VIDEO_DROP pts={pts} audio_clock={audio_clock} delta={delta}"
                )
                return self.get_display_frame(
                    video_buffer, audio_clock, playing
                )
            elif delta > FUTURE_HORIZON:
                # Кадр ещё слишком рано показывать – ждём, пока audio_clock догонит
                sync_monitor_logger.debug(
                    f"SYNC_VIDEO_FUTURE pts={pts} audio_clock={audio_clock} delta={delta}"
                )
                return video_buffer.get_keep_last()
            else:
                # Показываем кадр
                video_buffer.update_keep_last(frame, pts)
                video_buffer.advance()
                sync_monitor_logger.debug(
                    f"SYNC_VIDEO pts={pts} audio_clock={audio_clock} delta={delta} buffer_count={video_buffer.count}"
                )
                return frame
        else:
            # Буфер пуст – показываем последний сохранённый кадр
            sync_monitor_logger.debug(f"SYNC_VIDEO_EMPTY audio_clock={audio_clock}")
            return video_buffer.get_keep_last()

    def reset_drift(self):
        """Больше не используется, оставлен для совместимости."""
        pass