"""
test_sync_manager.py – модульные тесты для SyncManager.
Покрывает: выбор кадра при играющем и остановленном плеере,
удаление устаревших кадров, возврат keep_last при пустом буфере.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from core.sync_manager import SyncManager, MAX_VIDEO_LAG
from config.timebase import SAMPLES_PER_VIDEO_FRAME


@pytest.fixture
def mock_video_buffer():
    """FrameRingBuffer с контролируемым поведением."""
    buf = MagicMock()
    # Методы, которые нам понадобятся
    buf.peek_first.return_value = None
    buf.get_keep_last.return_value = None
    buf.drop_until = MagicMock()
    buf.advance = MagicMock()
    buf.update_keep_last = MagicMock()
    return buf


# ------------------------------------------------------------------
# Тесты
# ------------------------------------------------------------------

def test_paused_returns_keep_last(mock_video_buffer):
    """Если плеер на паузе, возвращается keep_last, буфер не трогается."""
    mgr = SyncManager()
    mock_video_buffer.get_keep_last.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
    result = mgr.get_display_frame(mock_video_buffer, audio_clock=10000, playing=False)
    assert result is mock_video_buffer.get_keep_last.return_value
    mock_video_buffer.drop_until.assert_not_called()
    mock_video_buffer.peek_first.assert_not_called()


def test_playing_drops_old_frames(mock_video_buffer):
    """При воспроизведении удаляются кадры старше audio_clock - MAX_VIDEO_LAG."""
    mgr = SyncManager()
    audio_clock = 50000
    # Настроим буфер: первый кадр возвращается, он не устарел
    frame = np.ones((10, 10, 3), dtype=np.uint8)
    mock_video_buffer.peek_first.return_value = (48000, frame)  # pts < audio_clock, но не устарел
    result = mgr.get_display_frame(mock_video_buffer, audio_clock=audio_clock, playing=True)
    # Должен быть вызван drop_until с audio_clock - MAX_VIDEO_LAG
    mock_video_buffer.drop_until.assert_called_once_with(audio_clock - MAX_VIDEO_LAG)
    # Кадр должен быть показан и обновлён keep_last
    assert result is frame
    mock_video_buffer.update_keep_last.assert_called_once_with(frame, 48000)
    mock_video_buffer.advance.assert_called_once()


def test_playing_skip_outdated_frame(mock_video_buffer):
    mgr = SyncManager()
    audio_clock = 50000
    outdated_pts = audio_clock - MAX_VIDEO_LAG - 1
    frame_outdated = np.zeros((10, 10, 3), dtype=np.uint8)
    frame_good = np.ones((10, 10, 3), dtype=np.uint8)

    mock_video_buffer.peek_first.side_effect = [
        (outdated_pts, frame_outdated),
        (48000, frame_good)
    ]
    result = mgr.get_display_frame(mock_video_buffer, audio_clock=audio_clock, playing=True)

    # Ожидаем два вызова advance: один для устаревшего кадра, второй — для показанного хорошего
    assert mock_video_buffer.advance.call_count == 2
    assert result is frame_good
    # keep_last обновлён только для хорошего кадра
    mock_video_buffer.update_keep_last.assert_called_once_with(frame_good, 48000)


def test_playing_empty_buffer_returns_keep_last(mock_video_buffer):
    """Если буфер пуст, возвращается keep_last."""
    mgr = SyncManager()
    keep = np.ones((10, 10, 3), dtype=np.uint8)
    mock_video_buffer.peek_first.return_value = None
    mock_video_buffer.get_keep_last.return_value = keep
    result = mgr.get_display_frame(mock_video_buffer, audio_clock=1000, playing=True)
    assert result is keep
    # drop_until был вызван
    mock_video_buffer.drop_until.assert_called_once()


def test_playing_no_keep_last_fallback(mock_video_buffer):
    """Если буфер пуст и keep_last тоже None, возвращается None."""
    mgr = SyncManager()
    mock_video_buffer.peek_first.return_value = None
    mock_video_buffer.get_keep_last.return_value = None
    result = mgr.get_display_frame(mock_video_buffer, audio_clock=1000, playing=True)
    assert result is None


def test_reset_drift_is_noop():
    """reset_drift существует и не вызывает ошибок."""
    mgr = SyncManager()
    mgr.reset_drift()  # не должно быть исключений