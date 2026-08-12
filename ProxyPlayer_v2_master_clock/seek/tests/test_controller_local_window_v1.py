#!/usr/bin/env python3
"""
Комплексный интеграционный тест StreamController с новыми модулями.
Проверяет: инициализацию, старт, seek, JKL, таймкод.
Требует TEST_MP4_PATH.
"""

import os
import sys
import time
from pathlib import Path

import pytest
import numpy as np
from PyQt5.QtWidgets import QApplication

# Добавляем корень проекта в sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.stream_controller import StreamController
from config.timebase import timecode_to_frame

TEST_MP4 = os.environ.get('TEST_MP4_PATH')
if not TEST_MP4:
    pytest.skip("TEST_MP4_PATH не задан", allow_module_level=True)

_app = QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def controller():
    mp4_path = Path(TEST_MP4)
    stem = mp4_path.stem
    parent = mp4_path.parent
    idx_path = parent / "idx" / "mp4" / f"{stem}.idx"
    if not idx_path.exists():
        idx_path = parent / f"{stem}.idx"
    ref_path = parent / f"{stem}.mp4.ref"
    if not ref_path.exists():
        ref_path = parent / f"{stem}.ref"

    ctrl = StreamController(
        ref_path=ref_path if ref_path.exists() else None,
        idx_path=idx_path,
        mp4_path=mp4_path,
        start_from_live=False,
    )
    assert ctrl._ready.wait(timeout=120), "StreamController не инициализировался"
    yield ctrl
    ctrl.close()


class TestComprehensive:
    def test_initialization(self, controller):
        assert controller.total_frames > 0
        assert controller._lazy_index is not None
        assert controller._playback is not None
        assert controller._pipeline is not None
        assert controller._seek_engine is not None

    def test_start_and_get_frame(self, controller):
        controller.start_playback()
        frame = None
        waited = 0
        while waited < 15:
            frame = controller.get_display_frame()
            if frame is not None:
                break
            time.sleep(0.5)
            waited += 0.5
        assert frame is not None, "Не получен кадр после старта"
        assert frame.shape[2] == 3  # RGB

    def test_seek(self, controller):
        controller.seek_absolute(1000)
        time.sleep(2)
        frame = controller.get_display_frame()
        assert frame is not None, "Нет кадра после seek"

    def test_timecode_input(self, controller):
        target_frame = timecode_to_frame("00:01:00;00", controller.fps)
        controller.seek_absolute(target_frame)
        time.sleep(2)
        frame = controller.get_display_frame()
        assert frame is not None, "Нет кадра после ввода таймкода"

    def test_jkl_speed(self, controller):
        controller.set_seek_speed(1)  # L
        display = controller.get_seek_speed_display()
        assert "x2" in display or "x4" in display or "x8" in display
        controller.reset_seek_speed()
        assert "x1" in controller.get_seek_speed_display()

    def test_toggle_pause(self, controller):
        controller.resume()
        assert controller.playing
        controller.pause()
        assert not controller.playing
        controller.resume()
        assert controller.playing

    def test_get_timecodes(self, controller):
        local = controller.get_local_timecode_str()
        real = controller.get_real_timecode_str()
        assert ":" in local
        assert ":" in real

    def test_seek_to_live(self, controller):
        controller.seek_absolute(max(0, controller.total_frames - 1600))
        time.sleep(2)
        frame = controller.get_display_frame()
        assert frame is not None, "Нет кадра после live seek"