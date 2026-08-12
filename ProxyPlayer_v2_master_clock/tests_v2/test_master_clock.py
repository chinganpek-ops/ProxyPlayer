"""
test_master_clock.py – модульные тесты для MasterClock (v2 исправленный).
Полностью соответствует обновлённому master_clock.py с независимым микшированием каналов.
"""

import pytest
import numpy as np
import time
from unittest.mock import MagicMock, patch

from core.master_clock import MasterClock


@pytest.fixture
def master_clock():
    """Создаёт MasterClock с _active=True для тестов очередей и коллбэка."""
    mc = MasterClock(sample_rate=48000, buffer_size=1024)
    mc._active = True
    return mc


# ------------------------------------------------------------------
# Тесты счётчика и времени
# ------------------------------------------------------------------
class TestTiming:
    def test_samples_played_starts_zero(self, master_clock):
        assert master_clock.samples_played == 0

    def test_set_clock(self, master_clock):
        master_clock.set_clock(48000)
        assert master_clock.samples_played == 48000

    def test_reset(self, master_clock):
        master_clock.set_clock(100000)
        master_clock._queue2.append(np.array([1.0], dtype=np.float32))
        master_clock.reset()
        assert master_clock.samples_played == 0
        assert len(master_clock._queue2) == 0

    def test_get_audio_clock_without_delay(self, master_clock):
        master_clock.set_clock(1920)
        master_clock.audio_delay = 0.0
        assert master_clock.get_audio_clock() == 1920

    def test_get_audio_clock_with_delay(self, master_clock):
        master_clock.set_clock(1920)
        master_clock.audio_delay = 0.01
        expected = 1920 + int(0.01 * 48000)
        assert master_clock.get_audio_clock() == expected


# ------------------------------------------------------------------
# Тесты push_audio и очередей
# ------------------------------------------------------------------
class TestPushAudio:
    def test_push_audio_to_empty_queue(self, master_clock):
        samples = np.ones(1024, dtype=np.float64)
        master_clock.push_audio(2, samples)
        assert len(master_clock._queue2) == 1
        np.testing.assert_array_equal(master_clock._queue2[0], samples.astype(np.float32))

    def test_push_audio_when_inactive(self):
        mc = MasterClock()
        mc._active = False
        mc.push_audio(2, np.ones(1024))
        assert len(mc._queue2) == 0

    def test_push_audio_ignores_disabled_track(self, master_clock):
        master_clock.set_track_enabled(2, False)
        master_clock.push_audio(2, np.ones(1024))
        assert len(master_clock._queue2) == 0

    def test_push_audio_updates_max_queue_len(self, master_clock):
        master_clock.push_audio(2, np.ones(1024))
        master_clock.push_audio(3, np.ones(512))
        assert master_clock._max_queue_len == 2

    def test_push_audio_invalid_track(self, master_clock):
        master_clock.push_audio(1, np.ones(1024))
        assert len(master_clock._queue2) == 0
        assert len(master_clock._queue3) == 0


# ------------------------------------------------------------------
# Тесты управления дорожками и mute
# ------------------------------------------------------------------
class TestTrackControl:
    def test_mute(self, master_clock):
        master_clock.set_muted(True)
        assert master_clock._muted is True
        master_clock.set_muted(False)
        assert master_clock._muted is False

    def test_set_track_enabled(self, master_clock):
        master_clock.set_track_enabled(2, False)
        assert not master_clock.is_track_enabled(2)
        # После отключения дорожка очищается
        master_clock._queue2.append(np.ones(100, dtype=np.float32))
        master_clock.set_track_enabled(2, False)
        assert len(master_clock._queue2) == 0
        master_clock.set_track_enabled(2, True)
        assert master_clock.is_track_enabled(2)

    def test_set_track_enabled_invalid_id(self, master_clock):
        # Никаких исключений
        master_clock.set_track_enabled(5, True)
        assert not master_clock.is_track_enabled(5)


# ------------------------------------------------------------------
# Тесты коллбэка
# ------------------------------------------------------------------
class TestCallback:
    def test_callback_increments_counter(self, master_clock):
        outdata = np.zeros((1024, 2), dtype=np.float32)
        initial = master_clock.samples_played
        master_clock._callback(outdata, 1024, None, None)
        assert master_clock.samples_played == initial + 1024

    def test_callback_muted_fills_silence(self, master_clock):
        master_clock.set_muted(True)
        outdata = np.ones((1024, 2), dtype=np.float32)
        master_clock._callback(outdata, 1024, None, None)
        assert np.all(outdata == 0.0)

    def test_callback_plays_from_queue(self, master_clock):
        chunk = np.array([0.5] * 500, dtype=np.float64)
        master_clock.push_audio(2, chunk)
        outdata = np.zeros((1024, 2), dtype=np.float32)
        master_clock._callback(outdata, 1024, None, None)

        expected_left = np.zeros(1024, dtype=np.float32)
        expected_left[:500] = 0.5
        np.testing.assert_array_almost_equal(outdata[:, 0], expected_left)
        assert np.all(outdata[:, 1] == 0.0)

    def test_callback_mixes_both_channels_independently(self, master_clock):
        master_clock.push_audio(2, np.array([0.8] * 1024, dtype=np.float64))
        master_clock.push_audio(3, np.array([0.3] * 1024, dtype=np.float64))
        outdata = np.zeros((1024, 2), dtype=np.float32)
        master_clock._callback(outdata, 1024, None, None)

        np.testing.assert_array_almost_equal(outdata[:, 0], 0.8)
        np.testing.assert_array_almost_equal(outdata[:, 1], 0.3)

    def test_callback_partial_chunk(self, master_clock):
        master_clock.push_audio(2, np.ones(1500, dtype=np.float64))
        outdata = np.zeros((1024, 2), dtype=np.float32)
        master_clock._callback(outdata, 1024, None, None)

        np.testing.assert_array_almost_equal(outdata[:, 0], 1.0)
        remaining = master_clock._queue2[0]
        assert len(remaining) == 1500 - 1024

    def test_callback_underrun_counted(self, master_clock):
        outdata = np.zeros((1024, 2), dtype=np.float32)
        master_clock._callback(outdata, 1024, None, None)
        assert master_clock._underruns == 2  # underrun по каждому каналу

    def test_callback_disabled_track_not_played(self, master_clock):
        # Включаем дорожку, добавляем данные
        master_clock.set_track_enabled(2, True)
        master_clock.push_audio(2, np.ones(1024, dtype=np.float64))
        # Отключаем дорожку (это очищает очередь)
        master_clock.set_track_enabled(2, False)
        outdata = np.ones((1024, 2), dtype=np.float32)
        master_clock._callback(outdata, 1024, None, None)
        # Левый канал должен быть тишиной
        assert np.all(outdata[:, 0] == 0.0)
        # Очередь пуста
        assert len(master_clock._queue2) == 0


# ------------------------------------------------------------------
# Тесты жизненного цикла (с моками sounddevice)
# ------------------------------------------------------------------
class TestLifecycle:
    @patch('sounddevice.OutputStream')
    def test_start_creates_stream(self, MockStream, master_clock):
        master_clock._active = False
        master_clock.start()
        MockStream.assert_called_once()
        MockStream.return_value.start.assert_called_once()
        assert master_clock._active is True

    def test_stop_closes_stream(self, master_clock):
        mock_stream = MagicMock()
        master_clock._stream = mock_stream
        master_clock._active = True
        master_clock.stop()
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
        assert master_clock._active is False

    def test_close(self, master_clock):
        master_clock._stream = MagicMock()
        master_clock.close()
        assert master_clock._active is False
        assert len(master_clock._queue2) == 0
        assert len(master_clock._queue3) == 0