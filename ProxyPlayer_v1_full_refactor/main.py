#!/usr/bin/env python3
"""
main.py – точка входа ProxyPlayer v1.
Запуск: python main.py [путь_к_mp4]
"""

import sys
import os
import time
from pathlib import Path
import logging

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QStandardPaths

from config.logger import setup_logging
from config.config import load_config
from player_window import ManagerWindow


def main():
    # Настройка логирования
    config = load_config()
    log_dir = Path(config.get('log_directory', '')) or Path(
        QStandardPaths.writableLocation(QStandardPaths.AppConfigLocation)
    ) / "ProxyPlayer" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) > 1:
        mp4_path = Path(sys.argv[1])
        safe_stem = "".join(c if c.isalnum() or c in "._- " else "_" for c in mp4_path.stem) or "empty"
        log_file = log_dir / f"{safe_stem}_{os.getpid()}.log"
    else:
        log_file = log_dir / "player.log"

    setup_logging(level=logging.DEBUG, log_file=str(log_file), mode='w')
    logging.getLogger("av").setLevel(logging.WARNING)

    app = QApplication(sys.argv)
    app.setApplicationName("ProxyPlayer")
    app.setApplicationVersion("1.0.0")

    # Определяем, открывать ли файл или пустой менеджер
    mp4_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("")
    window = ManagerWindow(mp4_path, config)
    if mp4_path.exists():
        window.setWindowTitle(f"ProxyPlayer - {mp4_path.name}")
        window.show()
    else:
        window.setWindowTitle("ProxyPlayer – Manager")
        window.show()
        window.hide()  # сворачиваем в трей

    exit_code = app.exec_()
    logging.getLogger("ProxyPlayer").info(f"Завершено с кодом {exit_code}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()