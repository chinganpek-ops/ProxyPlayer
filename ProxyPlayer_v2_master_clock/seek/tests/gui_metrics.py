#!/usr/bin/env python3
"""
gui_metrics.py – запускает GUI-плеер, собирает метрики и пишет их в metrics.log.
Окно остаётся открытым, метрики собираются в фоне.
"""

import sys
import time
import logging
from pathlib import Path

# Добавляем корень проекта в путь
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtCore import QTimer, Qt

from config.logger import setup_logging
from config.config import load_config
from core.stream_controller import StreamController
from output.video_widget import GLVideoWidget
from index.idx_cache import prepare_mirror, get_mirror_path

# Настройка логирования в файл (только для метрик)
metrics_logger = logging.getLogger("Metrics")
metrics_logger.setLevel(logging.INFO)
fh = logging.FileHandler("metrics.log", mode='w')
fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
metrics_logger.addHandler(fh)

class MetricsPlayer(QMainWindow):
    def __init__(self, mp4_path: Path, config: dict):
        super().__init__()
        self.setWindowTitle("GUI Metrics Player")
        self.resize(800, 600)

        # Центральный виджет с layout'ом
        central = QWidget()
        layout = QVBoxLayout(central)
        self.setCentralWidget(central)

        # Видео-виджет
        self.video_widget = GLVideoWidget()
        layout.addWidget(self.video_widget, stretch=1)

        # Кнопка Play/Pause
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        layout.addWidget(self.play_btn)

        # Контроллер
        idx_path, ref_path = self._find_related(mp4_path)
        mirror_path = self._prepare_mirror(idx_path)
        self.controller = StreamController(
            ref_path=ref_path,
            idx_path=idx_path,
            mp4_path=mp4_path,
            fps=config.get('fps', 25.0),
            buffer_size=800,
            start_from_live=True,
            mirror_path=str(mirror_path)
        )
        if not self.controller._ready.wait(timeout=60):
            raise RuntimeError("Инициализация не завершена")

        # Запуск воспроизведения
        self.controller.start_playback()
        # Рендер-таймер
        self.render_timer = QTimer(self)
        self.render_timer.timeout.connect(self._update_frame)
        self.render_timer.start(40)  # 25 fps

        # Таймер сбора метрик
        self.metrics_timer = QTimer(self)
        self.metrics_timer.timeout.connect(self._collect_metrics)
        self.metrics_timer.start(500)  # каждые 0.5 сек

        self.playing = False

    def _find_related(self, mp4):
        stem = mp4.stem
        parent = mp4.parent
        ref = parent / f"{stem}.mp4.ref"
        if not ref.exists(): ref = parent / f"{stem}.ref"
        idx = parent / "idx" / "mp4" / f"{stem}.idx"
        if not idx.exists(): idx = parent / f"{stem}.idx"
        return idx, ref

    def _prepare_mirror(self, idx):
        try:
            return prepare_mirror(idx)
        except Exception:
            return get_mirror_path(idx)

    def toggle_play(self):
        if self.playing:
            self.controller.pause()
            self.play_btn.setText("Play")
            self.playing = False
        else:
            self.controller.resume()
            self.play_btn.setText("Pause")
            self.playing = True

    def _update_frame(self):
        frame = self.controller.get_display_frame()
        if frame is not None and frame.size > 0:
            self.video_widget.set_frame(frame)

    def _collect_metrics(self):
        if not self.controller:
            return
        aclock = self.controller.audio_clock
        vb_cnt = self.controller.buffer_main.count
        ab2 = self.controller.audio_buffers.buffers[2]
        ab3 = self.controller.audio_buffers.buffers[3]
        raw_q = vid_q = aud_q = -1
        if self.controller._pipeline:
            raw_q = self.controller._pipeline._raw_queue.qsize()
            vid_q = self.controller._pipeline._video_queue.qsize()
            aud_q = self.controller._pipeline._audio_queue.qsize()

        log_line = (f"clock={aclock} | vb={vb_cnt} | "
                    f"ab2_read={ab2.read_pos} ab2_avail={ab2.available_read} ab2_write={ab2.write_pos} | "
                    f"ab3_read={ab3.read_pos} ab3_avail={ab3.available_read} ab3_write={ab3.write_pos} | "
                    f"raw_q={raw_q} vid_q={vid_q} aud_q={aud_q}")
        metrics_logger.info(log_line)

    def closeEvent(self, event):
        self.metrics_timer.stop()
        self.render_timer.stop()
        self.controller.stop()
        self.controller.close()
        event.accept()

def main():
    if len(sys.argv) < 2:
        print("Usage: python gui_metrics.py <mp4_path>")
        sys.exit(1)

    mp4_path = Path(sys.argv[1])
    if not mp4_path.exists():
        print(f"File not found: {mp4_path}")
        sys.exit(1)

    # Минимальный конфиг
    config = load_config()

    app = QApplication(sys.argv)
    window = MetricsPlayer(mp4_path, config)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()