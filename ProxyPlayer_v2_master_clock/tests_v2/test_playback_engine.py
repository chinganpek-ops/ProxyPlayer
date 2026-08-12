"""
test_playback_engine.py – тесты для PlaybackEngine.
Проверяет seek, поколения запросов, паузу/возобновление, JKL,
fast_seek и синхронизацию с MasterClock.
"""

import pytest
import numpy as np
import threading
import time
from unittest.mock import MagicMock

from core.playback_engine import PlaybackEngine
from buffer.frame_buffer import FrameRingBuffer


@pytest.fixture
def mock_pipeline():
    return MagicMock()


@pytest.fixture
def mock_seek_engine():
    return MagicMock()


@pytest.fixture
def mock_master_clock():
    mc = MagicMock()
    mc.get_audio_clock.return_value = 0
    return mc


@pytest.fixture
def video_buffer():
    return FrameRingBuffer(max_frames=100)


@pytest.fixture
def engine(mock_pipeline, mock_seek_engine, mock_master_clock, video_buffer):
    sync_mgr = MagicMock()
    sync_mgr.get_display_frame.return_value = np.zeros((10,10,3), dtype=np.uint8)
    eng = PlaybackEngine(
        pipeline=mock_pipeline,
        seek_engine=mock_seek_engine,
        sync_manager=sync_mgr,
        master_clock=mock_master_clock,
        video_buffer=video_buffer,
        total_frames=1000,
        fps=25.0,
    )
    return eng


class TestSeek:
    def test_seek_calls_seek_engine(self, engine, mock_seek_engine):
        window = MagicMock()
        engine.seek(500, window)
        mock_seek_engine.seek_async.assert_called_once()
        # Проверяем, что был передан on_complete и on_error
        args, kwargs = mock_seek_engine.seek_async.call_args
        assert 'on_complete' in kwargs

    def test_seek_increments_generation(self, engine):
        window = MagicMock()
        gen_before = engine._seek_generation
        engine.seek(100, window)
        assert engine._seek_generation == gen_before + 1

    def test_on_seek_complete_ignores_old_generation(self, engine, mock_pipeline):
        window = MagicMock()
        window.window_start_frame = 0
        buf = FrameRingBuffer(max_frames=10)
        frame = np.zeros((10,10,3), dtype=np.uint8)
        buf.try_push(frame, 1920)
        # Симулируем вызов с устаревшим поколением
        engine._seek_generation = 5
        engine._on_seek_complete(buf, window, gen=3)  # gen старый
        # Конвейер не должен перезапускаться
        mock_pipeline.stop.assert_not_called()

    def test_on_seek_complete_updates_state(self, engine, mock_pipeline, video_buffer):
        window = MagicMock()
        window.window_start_frame = 0
        buf = FrameRingBuffer(max_frames=10)
        frame = np.ones((10,10,3), dtype=np.uint8)
        buf.try_push(frame, 1920)
        engine._seek_generation = 1
        engine._on_seek_complete(buf, window, gen=1)
        # Проверяем новый буфер движка (buf), а не старый video_buffer
        assert engine._video_buffer.count == 1
        mock_pipeline.stop.assert_called_once()
        mock_pipeline.update_window.assert_called_once_with(window)
        mock_pipeline.start.assert_called_once()


class TestPlayPause:
    def test_resume_starts_master_clock(self, engine, mock_master_clock):
        engine._paused = True
        engine.resume()
        assert engine.playing
        mock_master_clock.start.assert_called_once()

    def test_pause_stops_master_clock(self, engine, mock_master_clock):
        engine.playing = True
        engine.pause()
        assert not engine.playing
        mock_master_clock.stop.assert_called_once()

    def test_stop(self, engine, mock_pipeline, mock_master_clock):
        engine.playing = True
        engine._paused = False
        engine.stop()
        assert not engine.playing
        assert not engine._paused
        mock_pipeline.stop.assert_called_once()
        mock_master_clock.stop.assert_called_once()


class TestJKL:
    def test_set_speed_mutes_at_first(self, engine, mock_master_clock):
        engine.set_speed(1)
        assert engine._seek_direction == 1
        assert engine._seek_speed == 2.0
        mock_master_clock.set_muted.assert_called_with(True)

    def test_reset_speed_unmutes(self, engine, mock_master_clock):
        engine.set_speed(1)
        engine.reset_speed()
        mock_master_clock.set_muted.assert_called_with(False)
        assert engine._seek_speed == 1.0

    def test_fast_seek_updates_frame_idx(self, engine, video_buffer):
        frame = np.ones((10,10,3), dtype=np.uint8)
        video_buffer.try_push(frame, 5000)
        video_buffer.try_push(frame, 6000)
        engine._current_frame_idx = 100
        engine._fast_seek(200)
        assert engine._current_frame_idx == 200
        # Устаревшие кадры удалены
        assert video_buffer.count == 0  # потому что drop_until 200*1920


class TestGetDisplayFrame:
    def test_playing_uses_sync_manager(self, engine):
        engine.playing = True
        frame = engine.get_display_frame()
        engine._sync.get_display_frame.assert_called_once()

    def test_paused_returns_keep_last(self, engine, video_buffer):
        frame = np.ones((10,10,3), dtype=np.uint8)
        video_buffer.update_keep_last(frame, 5000)
        engine.playing = False
        # Заставляем sync_manager вернуть frame
        engine._sync.get_display_frame.return_value = frame
        result = engine.get_display_frame()
        assert result is frame

    def test_jkl_does_not_call_sync(self, engine):
        engine.set_speed(1)
        engine.playing = False
        engine.get_display_frame()
        engine._sync.get_display_frame.assert_not_called()