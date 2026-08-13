"""
test_gui_seek.py – GUI-тест кнопок Live и слайдера (реальный файл).
"""

import os, sys, time
from pathlib import Path
import pytest
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from index.idx_cache import prepare_mirror
from player_window import PlayerWidget

@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app
    app.quit()

def _get_dalet_files():
    mp4_str = os.environ.get("DALET_MP4")
    mp4_path = Path(mp4_str) if mp4_str else Path(
        "//10.20.49.171/DaletProxy/LSU/LR_2689/1923198_2026-08-12T15-45-25264.mp4"
    )
    if not mp4_path.exists():
        return None
    idx = mp4_path.parent / "idx" / "mp4" / f"{mp4_path.stem}.idx"
    if not idx.exists():
        idx = mp4_path.parent / f"{mp4_path.stem}.idx"
    ref = mp4_path.parent / f"{mp4_path.stem}.mp4.ref"
    if not ref.exists():
        ref = mp4_path.parent / f"{mp4_path.stem}.ref"
    if idx.exists() and ref.exists():
        return {"mp4": mp4_path, "idx": idx, "ref": ref}
    return None

@pytest.fixture(scope="module")
def dalet_files():
    files = _get_dalet_files()
    if files is None:
        pytest.skip("Файлы Dalet не найдены – тест пропущен")
    mirror_path = prepare_mirror(files["idx"])
    files["mirror"] = str(mirror_path)
    return files

@pytest.fixture
def player_widget(qtbot, dalet_files, qapp):
    config = {
        "fps": 25.0,
        "buffer_size": 600,
        "free_slots_required": 25,
        "group_chunks": 30,
        "max_retries": 3,
        "thread_type": "AUTO",
        "thread_count": 0,
        "skip_frame": False,
        "use_gpu_decoder": "off",
        "audio_delay_ms": 0,
        "active_tracks": [2, 3],
    }
    widget = PlayerWidget(
        mp4_path=dalet_files["mp4"],
        config=config,
        use_moov=False,
        mirror_path=dalet_files["mirror"],
    )
    qtbot.addWidget(widget)
    widget.show()
    # Ждём готовности с таймаутом 60 секунд
    if not widget.player._ready.wait(timeout=60):
        widget.close()
        pytest.fail("StreamController не инициализировался за 60 секунд")
    widget.start_playback()
    time.sleep(1.5)
    yield widget
    widget.player.close()
    widget.close()

class TestGUISeek:
    def test_live_button_fills_buffer(self, player_widget, qtbot):
        assert player_widget.isVisible()
        QTest.mouseClick(player_widget.live_btn, Qt.LeftButton)
        time.sleep(2.0)
        assert player_widget.player.buffer_main.count > 0, "Буфер пуст после Live"
        frame = player_widget._active_player.get_display_frame()
        assert frame is not None and frame.size > 0, "Не получен кадр после Live"