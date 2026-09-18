"""
sync_manager.py – менеджер синхронизации аудио/видео для ProxyPlayer v2.
Работает напрямую с audio_clock от MasterClock.
Дрейф-коррекция удалена.

Добавлено логирование синхронизации в sync_monitor.log через SyncMonitor.

ИЗМЕНЕНИЯ (правки продакшен-ревью):
- get_display_frame(): отбрасывание устаревших кадров раньше делалось
  через хвостовую рекурсию (return self.get_display_frame(...)). Python не
  оптимизирует хвостовую рекурсию, а метод вызывается на каждый тик
  рендера — при накоплении в буфере множества устаревших кадров (пауза,
  лаги декодера) это могло упереться в RecursionError. Логика и пороги
  (MAX_VIDEO_LAG, тексты и порядок логов) не изменились, только цикл
  вместо self-вызова.
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
        # Эталонный PTS после seek: кадры с меньшим PTS отбрасываются
        self._seek_pts_reference = 0

    def set_seek_reference(self, pts: int):
        """Устанавливает минимальный допустимый PTS для выдачи кадров."""
        self._seek_pts_reference = pts
        sync_monitor_logger.debug(f"SYNC_SEEK_REF pts={pts}")

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
            sync_monitor_logger.debug(f"SYNC_VIDEO_SEEK_FILTER pts={first_check[0]}")
            video_buffer.advance()

        # Выбираем кадр ПО ЧАСАМ, а не «по одному за вызов».
        #
        # Раньше здесь показывался первый кадр буфера и делался ровно один
        # advance(). Это привязывало скорость видео к частоте вызовов
        # рендера, а не к звуку: если тик приходит реже 25 раз в секунду —
        # на Windows QTimer с интервалом 40 мс из-за гранулярности
        # системного таймера реально срабатывает раз в 46-62 мс — видео
        # получало 16-21 кадр в секунду вместо 25 и накапливало отставание.
        # Порог сброса MAX_VIDEO_LAG (400 мс) при этом не достигался, и
        # дрейф просто копился, не корректируясь.
        #
        # Теперь берётся самый свежий кадр, который УЖЕ должен был быть
        # показан (pts <= audio_clock), а всё, что он обогнал,
        # отбрасывается. Видео следует за звуком независимо от того, с
        # какой частотой приходят тики рендера.
        chosen = None
        dropped = 0
        while True:
            first = video_buffer.peek_first()
            if first is None:
                break

            pts, frame = first
            if pts > current_clock:
                # Кадр из будущего — его время ещё не пришло.
                break

            if chosen is not None:
                dropped += 1
            chosen = (pts, frame)
            video_buffer.advance()

        if chosen is None:
            # Подходящего кадра нет: либо буфер пуст, либо все кадры ещё
            # впереди по времени. Показываем последний сохранённый.
            sync_monitor_logger.debug(
                f"SYNC_VIDEO_WAIT audio_clock={audio_clock} "
                f"buffer_count={video_buffer.count}")
            return video_buffer.get_keep_last()

        pts, frame = chosen
        video_buffer.update_keep_last(frame, pts)
        delta = pts - current_clock
        if dropped:
            sync_monitor_logger.debug(
                f"SYNC_VIDEO_CATCHUP dropped={dropped} pts={pts} "
                f"audio_clock={audio_clock} delta={delta}")
        sync_monitor_logger.debug(
            f"SYNC_VIDEO pts={pts} audio_clock={audio_clock} delta={delta} "
            f"buffer_count={video_buffer.count}")
        return frame

    # ------------------------------------------------------------------
    # Заглушка для совместимости со старым кодом
    # ------------------------------------------------------------------
    def reset_drift(self):
        """Больше не используется, оставлен для совместимости."""
        pass
