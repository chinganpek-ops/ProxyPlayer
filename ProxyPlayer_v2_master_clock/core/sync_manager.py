"""
sync_manager.py – менеджер синхронизации аудио/видео для ProxyPlayer v2.
Работает напрямую с audio_clock от MasterClock.
Дрейф-коррекция и зависимость от аудиобуферов удалены.

Добавлена фильтрация по эталонному PTS после seek: кадры с PTS меньше
эталонного отбрасываются на уровне выдачи, чтобы исключить подмешивание
старых кадров, даже если они попали в буфер до обновления окна.
"""

import logging
from typing import Optional

import numpy as np

from buffer.frame_buffer import FrameRingBuffer
from config.timebase import SAMPLES_PER_VIDEO_FRAME

logger = logging.getLogger(__name__)

# --- Логгер для мониторинга seek (используется совместно с playback_engine и chunk_pipeline) ---
monitor_logger = logging.getLogger("SeekMonitor")
monitor_logger.setLevel(logging.DEBUG)
if not monitor_logger.handlers:
    _mon_handler = logging.FileHandler("seek_monitor.log", encoding="utf-8")
    _mon_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    monitor_logger.addHandler(_mon_handler)
monitor_logger.propagate = False

MAX_VIDEO_LAG = SAMPLES_PER_VIDEO_FRAME * 10      # 19200 сэмплов (400 мс)
FUTURE_HORIZON = SAMPLES_PER_VIDEO_FRAME // 2      # 960 сэмплов


class SyncManager:
    """
    Управляет синхронизацией видео с аудио.
    Видео ведомое, аудио ведущее (через MasterClock).
    """

    def __init__(self):
        # Эталонный PTS после seek: кадры с меньшим PTS отбрасываются
        self._seek_pts_reference = 0

    def set_seek_reference(self, pts: int):
        """Устанавливает минимальный допустимый PTS для выдачи кадров."""
        self._seek_pts_reference = pts
        monitor_logger.debug(f"SYNC_SEEK_REF pts={pts}")

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
            monitor_logger.debug(f"DROP_BEFORE_SEEK_REF pts={first_check[0]}")
            video_buffer.advance()

        first = video_buffer.peek_first()
        if first is not None:
            pts, frame = first
            delta = pts - current_clock

            if delta < -MAX_VIDEO_LAG:
                # Безнадёжно устарел – пропускаем
                video_buffer.advance()
                monitor_logger.debug(f"DROP_STALE pts={pts} audio_clock={audio_clock}")
                return self.get_display_frame(
                    video_buffer, audio_clock, playing
                )
            else:
                # Показываем кадр
                video_buffer.update_keep_last(frame, pts)
                video_buffer.advance()
                monitor_logger.debug(f"DISPLAY_FRAME pts={pts} audio_clock={audio_clock} buffer_count={video_buffer.count}")
                return frame
        else:
            # Буфер пуст – показываем последний сохранённый кадр
            monitor_logger.debug(f"BUFFER_EMPTY audio_clock={audio_clock}")
            return video_buffer.get_keep_last()

    # ------------------------------------------------------------------
    # Заглушка для совместимости со старым кодом
    # ------------------------------------------------------------------
    def reset_drift(self):
        """Больше не используется, оставлен для совместимости."""
        pass