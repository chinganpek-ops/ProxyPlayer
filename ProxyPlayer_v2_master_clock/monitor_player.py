#!/usr/bin/env python3
"""
monitor_player.py – мониторинг нажатий клавиш и состояния плеера.
Запускает плеер с указанным MP4, выводит в консоль все изменения.
"""

import sys
import time
import logging
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QTimer

# Добавляем корень проекта
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.logger import setup_logging
from config.config import load_config
from core.stream_controller import StreamController
from output.video_widget import GLVideoWidget
from index.idx_cache import prepare_mirror, get_mirror_path
from player_window import PlayerWidget

# Настраиваем логгер для мониторинга
mon_log = logging.getLogger("Monitor")
mon_log.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
mon_log.addHandler(handler)

class MonitorPlayer(PlayerWidget):
    """Расширенная версия PlayerWidget с мониторингом событий."""

    def __init__(self, mp4_path, config, use_moov=False, mirror_path=None):
        super().__init__(mp4_path, config, use_moov, mirror_path)
        self._monitor_timer = QTimer(self)
        self._monitor_timer.timeout.connect(self._dump_state)
        self._monitor_timer.start(100)  # каждые 100 мс

    def keyPressEvent(self, event):
        """Логируем нажатия клавиш."""
        key_name = self._key_to_name(event.key())
        mon_log.info(f"KEY: {key_name}")
        super().keyPressEvent(event)

    def _key_to_name(self, key):
        mapping = {
            Qt.Key_Space: "SPACE",
            Qt.Key_Left: "LEFT",
            Qt.Key_Right: "RIGHT",
            Qt.Key_Up: "UP",
            Qt.Key_Down: "DOWN",
            Qt.Key_J: "J",
            Qt.Key_K: "K",
            Qt.Key_L: "L",
            Qt.Key_Escape: "ESC",
        }
        return mapping.get(key, f"0x{key:X}")

    def _dump_state(self):
        if not self.player:
            return
        playing = self.player.playing
        paused = self.player._paused
        aclock = self.player.audio_clock
        slider_val = self.slider.value()
        btn_text = self.play_pause_btn.text()
        mon_log.debug(f"STATE: playing={playing}, paused={paused}, "
                      f"audio_clock={aclock}, slider={slider_val}, btn='{btn_text}'")

def main():
    if len(sys.argv) < 2:
        print("Usage: python monitor_player.py <mp4_path>")
        sys.exit(1)

    mp4_path = Path(sys.argv[1])
    if not mp4_path.exists():
        print(f"File not found: {mp4_path}")
        sys.exit(1)

    config = load_config()
    app = QApplication(sys.argv)

    # Подготовка зеркала
    idx_path, ref_path = _find_related(mp4_path)
    try:
        mirror_path = prepare_mirror(idx_path)
    except Exception:
        mirror_path = get_mirror_path(idx_path)
        if not mirror_path.exists():
            print("Зеркало не найдено")
            sys.exit(1)

    window = MonitorPlayer(mp4_path, config, mirror_path=str(mirror_path))
    window.setWindowTitle(f"Monitor: {mp4_path.name}")
    window.show()
    window.start_playback()
    exit_code = app.exec_()
    sys.exit(exit_code)

def _find_related(mp4):
    stem = mp4.stem
    parent = mp4.parent
    ref = parent / f"{stem}.mp4.ref"
    if not ref.exists():
        ref = parent / f"{stem}.ref"
    idx = parent / "idx" / "mp4" / f"{stem}.idx"
    if not idx.exists():
        idx = parent / f"{stem}.idx"
    return idx, ref

if __name__ == "__main__":
    main()