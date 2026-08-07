"""
sync_manager.py – менеджер синхронизации аудио/видео для ProxyPlayer v1.
Выделен из PlayerController. Отвечает за:
- отображение правильного кадра по audio_clock
- коррекцию дрейфа аудиовыхода
- отбрасывание устаревших кадров
"""

import time
import logging
from typing import Optional

import numpy as np

from buffer.frame_buffer import FrameRingBuffer
from buffer.audio_buffer import MultiTrackAudioBuffer
from config.timebase import (
    AUDIO_SAMPLE_RATE,
    SAMPLES_PER_VIDEO_FRAME,
    video_frame_to_pts,
    pts_to_video_frame,
)

logger = logging.getLogger(__name__)

MAX_VIDEO_LAG = SAMPLES_PER_VIDEO_FRAME * 10      # 19200 сэмплов (400 мс)
FUTURE_HORIZON = SAMPLES_PER_VIDEO_FRAME // 2      # 960 сэмплов


class SyncManager:
    """
    Управляет синхронизацией видео с аудио.
    Видео ведомое, аудио ведущее.
    """

    def __init__(self):
        # Коррекция дрейфа
        self._drift = {
            'enabled': True,
            'nominal_rate': AUDIO_SAMPLE_RATE,
            'measure_interval': 2.0,
            'last_sys_time': None,
            'last_audio_clock': None,
            'measured_rate': None,
            'start_sys_time': None,
            'start_audio_clock': None,
        }

    # ------------------------------------------------------------------
    # Основной метод выбора кадра
    # ------------------------------------------------------------------
    def get_display_frame(
        self,
        video_buffer: FrameRingBuffer,
        audio_buffers: MultiTrackAudioBuffer,
        audio_clock: int,
        audio_delay_samples: int = 0,
        playing: bool = True,
        active_tracks: list = None,
    ) -> Optional[np.ndarray]:
        """
        Возвращает кадр для отображения на основе текущего audio_clock.
        Автоматически отбрасывает устаревшие кадры.
        """
        if not playing:
            return video_buffer.get_keep_last()

        # Обновляем коррекцию дрейфа
        self._update_drift_correction(audio_clock)
        adjusted_clock = self._get_adjusted_audio_clock(audio_clock)
        current_clock = adjusted_clock - audio_delay_samples

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
                    video_buffer, audio_buffers, audio_clock,
                    audio_delay_samples, playing, active_tracks
                )
            else:
                # Показываем кадр
                video_buffer.update_keep_last(frame, pts)
                video_buffer.advance()
                return frame
        else:
            return video_buffer.get_keep_last()

    # ------------------------------------------------------------------
    # Коррекция дрейфа аудиовыхода
    # ------------------------------------------------------------------
    def _update_drift_correction(self, audio_clock: int):
        """Измеряет реальную скорость audio_clock и обновляет модель дрейфа."""
        if not self._drift['enabled']:
            return

        now = time.monotonic()
        dc = self._drift

        if dc['start_sys_time'] is None:
            dc['start_sys_time'] = now
            dc['start_audio_clock'] = audio_clock
            dc['last_sys_time'] = now
            dc['last_audio_clock'] = audio_clock
            return

        if now - dc['last_sys_time'] >= dc['measure_interval']:
            dt = now - dc['last_sys_time']
            da = audio_clock - dc['last_audio_clock']
            if dt > 0.1 and da > 0:
                measured_rate = da / dt
                if dc['measured_rate'] is None:
                    dc['measured_rate'] = measured_rate
                else:
                    alpha = 0.3
                    dc['measured_rate'] = (1 - alpha) * dc['measured_rate'] + alpha * measured_rate
                dc['start_sys_time'] = now
                dc['start_audio_clock'] = audio_clock
            dc['last_sys_time'] = now
            dc['last_audio_clock'] = audio_clock

    def _get_adjusted_audio_clock(self, audio_clock: int) -> float:
        """Возвращает скорректированный audio_clock с учётом дрейфа."""
        dc = self._drift
        if not dc['enabled'] or dc['measured_rate'] is None:
            return float(audio_clock)

        if abs(dc['measured_rate'] - dc['nominal_rate']) / dc['nominal_rate'] < 0.0001:
            return float(audio_clock)

        now = time.monotonic()
        adjusted = dc['start_audio_clock'] + (now - dc['start_sys_time']) * dc['nominal_rate']

        max_deviation = int(0.5 * AUDIO_SAMPLE_RATE)
        if abs(adjusted - audio_clock) > max_deviation:
            logger.warning("Слишком большая коррекция дрейфа, сброс")
            dc['measured_rate'] = None
            return float(audio_clock)

        return adjusted

    def reset_drift(self):
        """Сброс накопленной коррекции (при переходе на новую позицию)."""
        self._drift['measured_rate'] = None
        self._drift['start_sys_time'] = None
        self._drift['last_sys_time'] = None
        self._drift['last_audio_clock'] = None