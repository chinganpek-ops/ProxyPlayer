"""
Тестирование SyncManager с мок-буферами.
"""

import numpy as np
from buffer.frame_buffer import FrameRingBuffer
from buffer.audio_buffer import MultiTrackAudioBuffer
from core.sync_manager import SyncManager


class TestGetDisplayFrame:
    def setup_method(self):
        self.sync = SyncManager()
        # Отключаем коррекцию дрейфа для детерминированности
        self.sync._drift['enabled'] = False

        self.video_buf = FrameRingBuffer(max_frames=10)
        self.audio_buf = MultiTrackAudioBuffer(capacity_samples=48000)

    def _push_frame(self, pts: int):
        """Вспомогательный метод: добавляет фейковый кадр."""
        frame = np.zeros((576, 720, 3), dtype=np.uint8)
        self.video_buf.try_push(frame, pts)

    def test_returns_keep_last_when_not_playing(self):
        self._push_frame(0)
        self._push_frame(1920)
        # playing=False
        frame = self.sync.get_display_frame(
            self.video_buf, self.audio_buf, audio_clock=0, playing=False
        )
        # Буфер не должен опустошаться, но keep_last ещё не установлен.
        # get_display_frame при not playing возвращает get_keep_last(),
        # который изначально None. После ручной установки keep_last должен вернуться.
        self.video_buf.update_keep_last(np.zeros((1,1,3), dtype=np.uint8), 0)
        frame = self.sync.get_display_frame(
            self.video_buf, self.audio_buf, audio_clock=0, playing=False
        )
        assert frame is not None

    def test_advances_when_frame_ready(self):
        # Кадры с pts=0, 1920, 3840
        self._push_frame(0)
        self._push_frame(1920)
        self._push_frame(3840)

        # audio_clock = 1000, pts=0 отстаёт меньше чем на MAX_VIDEO_LAG (19200), должен показаться
        frame = self.sync.get_display_frame(
            self.video_buf, self.audio_buf, audio_clock=1000, playing=True
        )
        assert frame is not None
        # Первый кадр удалён, в буфере осталось 2
        assert self.video_buf.count == 2

    def test_drops_old_frame(self):
        # Кадр 0, аудио ушло далеко вперёд (20000 > 19200)
        self._push_frame(0)
        frame = self.sync.get_display_frame(
            self.video_buf, self.audio_buf, audio_clock=20000, playing=True
        )
        # Кадр должен быть пропущен, буфер пуст
        assert self.video_buf.count == 0
        # Возвращается keep_last (None, если не установлен)
        assert frame is None

    def test_returns_keep_last_when_buffer_empty(self):
        # Буфер пуст, playing=True -> get_keep_last
        frame = self.sync.get_display_frame(
            self.video_buf, self.audio_buf, audio_clock=0, playing=True
        )
        assert frame is None  # keep_last не установлен

    def test_drift_correction_disabled(self):
        # При выключенной коррекции _get_adjusted_audio_clock возвращает исходный
        adj = self.sync._get_adjusted_audio_clock(48000)
        assert adj == 48000.0