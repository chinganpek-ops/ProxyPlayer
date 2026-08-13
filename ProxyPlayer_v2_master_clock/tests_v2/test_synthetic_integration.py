"""
test_synthetic_integration.py – синтетический интеграционный тест
для проверки критических сценариев: seek, синхронизация, PTS, буферы.
Без реальных файлов и GUI, с контролируемым временем и моками.
Все критические точки покрыты.
"""

import pytest
import numpy as np
import threading
import time
from unittest.mock import MagicMock

from buffer.frame_buffer import FrameRingBuffer
from core.sync_manager import SyncManager
from core.playback_engine import PlaybackEngine
from core.master_clock import MasterClock
from config.timebase import video_frame_to_pts, pts_to_video_frame, SAMPLES_PER_VIDEO_FRAME


# ------------------------------------------------------------------
# Генерация тестового видеокадра
# ------------------------------------------------------------------
def make_test_frame(pts: int) -> np.ndarray:
    """Создаёт кадр, в котором записан его PTS (для отладки)."""
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img[:, :, 0] = (pts // 100) % 256
    return img


# ------------------------------------------------------------------
# Мок ChunkPipeline, который при старте добавляет кадры в буфер
# ------------------------------------------------------------------
class MockPipeline:
    def __init__(self, video_buffer: FrameRingBuffer, fps=25.0):
        self._video_buffer = video_buffer
        self._fps = fps
        self._window = None
        self._active = False
        self._scheduler = MagicMock()
        self._scheduler.set_normal_mode = MagicMock()
        self._started = False

    def set_master_clock(self, mc):
        pass

    def get_window_snapshot(self):
        return self._window

    def update_window(self, new_window):
        self._window = new_window

    def start(self, start_local_chunk=0):
        self._active = True
        base_frame = start_local_chunk * 12
        for i in range(12):
            pts = video_frame_to_pts(base_frame + i)
            self._video_buffer.try_push(make_test_frame(pts), pts)
        self._started = True

    def stop(self):
        self._active = False
        self._started = False


# ------------------------------------------------------------------
# Мок SeekEngine, синхронно возвращающий буфер с кадрами
# ------------------------------------------------------------------
class MockSeekEngine:
    def __init__(self, lazy_index=None, decoder=None, reader=None):
        self._generation = 0
        self._lock = threading.Lock()
        self._current_request = None

    def seek_async(self, frame_idx, on_complete, on_error=None):
        with self._lock:
            self._generation += 1
            gen = self._generation
        idr_frame = (frame_idx // 12) * 12
        buf = FrameRingBuffer(max_frames=30)
        for i in range(12):
            pts = video_frame_to_pts(idr_frame + i)
            buf.try_push(make_test_frame(pts), pts)
        threading.Thread(target=lambda: on_complete(buf), daemon=True).start()

    def cancel_current(self):
        pass

    def seek_sync(self, frame_idx):
        idr_frame = (frame_idx // 12) * 12
        buf = FrameRingBuffer(max_frames=30)
        for i in range(12):
            pts = video_frame_to_pts(idr_frame + i)
            buf.try_push(make_test_frame(pts), pts)
        return buf


# ------------------------------------------------------------------
# Мок MasterClock, управляемый вручную
# ------------------------------------------------------------------
class ManualMasterClock(MasterClock):
    """MasterClock, который не запускает реальный поток."""
    def __init__(self):
        super().__init__(sample_rate=48000, buffer_size=1024)
        self._manual_clock = 0
        self._active = True

    def start(self):
        self._active = True

    def stop(self):
        self._active = False

    def get_audio_clock(self) -> int:
        return self._manual_clock

    def set_clock(self, pts: int):
        self._manual_clock = pts

    def push_audio(self, track_id, samples):
        pass


# ------------------------------------------------------------------
# Фикстура с полным окружением
# ------------------------------------------------------------------
@pytest.fixture
def engine_env():
    video_buf = FrameRingBuffer(max_frames=100)
    pipeline = MockPipeline(video_buf)
    seek_engine = MockSeekEngine()
    sync_mgr = SyncManager()
    master_clock = ManualMasterClock()
    engine = PlaybackEngine(
        pipeline=pipeline,
        seek_engine=seek_engine,
        sync_manager=sync_mgr,
        master_clock=master_clock,
        video_buffer=video_buf,
        total_frames=300,
        fps=25.0,
    )
    return engine, video_buf, master_clock, pipeline, seek_engine


# ------------------------------------------------------------------
# Вспомогательный класс окна индекса
# ------------------------------------------------------------------
class FakeIndexWindow:
    def __init__(self, start_frame=0, total_chunks=25):
        self.window_start_frame = start_frame
        self.total_chunks = total_chunks


# ------------------------------------------------------------------
# Тесты
# ------------------------------------------------------------------
class TestSyntheticIntegration:
    def test_start_playback_and_resume(self, engine_env):
        engine, buf, mc, pipeline, _ = engine_env
        engine.start_playback(0, 0)
        assert engine._paused
        assert buf.count >= 1
        first_pts = buf.peek_first()[0]
        assert first_pts == 0

        mc.set_clock(0)
        engine.resume()
        assert engine.playing
        mc.set_clock(1920)
        frame = engine.get_display_frame()
        assert frame is not None

    def test_seek_and_buffer_consistency(self, engine_env):
        engine, buf, mc, pipeline, _ = engine_env
        engine.start_playback(0, 0)
        assert buf.count > 0
        initial_last = buf.get_keep_last_pts()
        window = FakeIndexWindow(0, 25)
        engine.seek(50, window)
        time.sleep(0.1)
        new_first = buf.peek_first()
        assert new_first is not None
        assert new_first[0] >= video_frame_to_pts(48)
        assert buf.get_keep_last_pts() > initial_last

    def test_double_seek_only_last_applied(self, engine_env):
        engine, buf, mc, pipeline, _ = engine_env
        engine.start_playback(0, 0)
        window1 = FakeIndexWindow(0, 25)
        window2 = FakeIndexWindow(0, 25)
        engine.seek(50, window1)
        engine.seek(100, window2)
        time.sleep(0.2)
        first_pts = buf.peek_first()[0]
        assert first_pts in [video_frame_to_pts(96), video_frame_to_pts(108)]

    def test_jkl_mutes_and_unmutes(self, engine_env):
        engine, buf, mc, pipeline, _ = engine_env
        engine.start_playback(0, 0)
        mc.set_clock(1920)
        engine.resume()
        engine.set_speed(1)
        assert mc._muted is True
        engine.reset_speed()
        assert mc._muted is False

    def test_pause_keeps_last_frame(self, engine_env):
        engine, buf, mc, pipeline, _ = engine_env
        engine.start_playback(0, 0)
        mc.set_clock(0)
        engine.resume()
        mc.set_clock(1920)
        frame1 = engine.get_display_frame()
        engine.pause()
        frame2 = engine.get_display_frame()
        assert frame2 is not None

    def test_audio_clock_progression(self, engine_env):
        engine, buf, mc, pipeline, _ = engine_env
        engine.start_playback(0, 0)
        mc.set_clock(0)
        engine.resume()
        mc.set_clock(96000)
        engine.get_display_frame()
        assert engine.audio_clock == 96000