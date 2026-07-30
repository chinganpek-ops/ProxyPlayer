"""
Интеграционный тест StreamController на реальном файле.
Требует TEST_MP4_PATH.
"""

import os
import time
from pathlib import Path

import pytest
import numpy as np
from PyQt5.QtWidgets import QApplication

from core.stream_controller import StreamController

TEST_MP4 = os.environ.get('TEST_MP4_PATH')
if not TEST_MP4:
    pytest.skip("TEST_MP4_PATH не задан", allow_module_level=True)

# Создаём QApplication один раз для всех тестов
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


class TestStreamController:
    def test_initialization(self, controller):
        assert controller.total_frames > 0
        assert controller._lazy_index is not None
        assert controller._playback is not None

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

    def test_seek(self, controller):
        controller.seek_absolute(1000)
        time.sleep(2)
        frame = controller.get_display_frame()
        assert frame is not None, "Нет кадра после seek"