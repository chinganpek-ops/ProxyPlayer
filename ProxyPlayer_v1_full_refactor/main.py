#!/usr/bin/env python3
"""
main.py – точка входа ProxyPlayer v1.
Запуск: python main.py [путь_к_mp4] [--managed]
"""

import sys
import os
from pathlib import Path
import logging

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QStandardPaths
from PyQt5.QtNetwork import QLocalSocket

from config.logger import setup_logging
from config.config import load_config
from player_window import ManagerWindow, PlayerWidget  # важно: PlayerWidget


def send_to_existing(path_str: str) -> bool:
    """Отправляет путь существующему менеджеру и возвращает True, если удалось."""
    socket = QLocalSocket()
    socket.connectToServer("ProxyPlayerManager")
    if socket.waitForConnected(500):
        msg = f"FILE:{path_str}".encode()
        socket.write(msg)
        socket.flush()
        socket.disconnectFromServer()
        socket.close()
        return True
    return False


def main():
    config = load_config()
    log_dir = Path(config.get('log_directory', '') or
                   QStandardPaths.writableLocation(QStandardPaths.AppConfigLocation)) / "ProxyPlayer" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    managed = '--managed' in sys.argv
    mp4_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("")

    if len(sys.argv) > 1:
        safe_stem = "".join(c if c.isalnum() or c in "._- " else "_" for c in mp4_path.stem) or "empty"
        log_file = log_dir / f"{safe_stem}_{os.getpid()}.log"
    else:
        log_file = log_dir / "player.log"

    setup_logging(level=logging.DEBUG, log_file=str(log_file), mode='w')
    logging.getLogger("av").setLevel(logging.WARNING)

    app = QApplication(sys.argv)
    app.setApplicationName("ProxyPlayer")
    app.setApplicationVersion("1.0.0")

    # Если это не управляемый плеер и файл существует, пробуем отправить существующему менеджеру
    if not managed and mp4_path.exists() and send_to_existing(str(mp4_path)):
        logging.getLogger("ProxyPlayer").info("Файл отправлен существующему менеджеру")
        sys.exit(0)

    if managed:
        # Управляемый плеер – создаём только PlayerWidget
        if not mp4_path.exists():
            logging.error(f"Файл {mp4_path} не найден, управляемый плеер не запущен")
            sys.exit(1)
        widget = PlayerWidget(mp4_path, config)
        widget.setWindowTitle(f"Player - {mp4_path.name}")
        widget.show()
        widget.start_playback()
    else:
        # Обычный менеджер или пустой запуск
        window = ManagerWindow(mp4_path, config)
        if mp4_path.exists():
            window.setWindowTitle(f"ProxyPlayer - {mp4_path.name}")
            window.show()
        else:
            window.setWindowTitle("ProxyPlayer – Manager")
            window.show()
            window.hide()

    exit_code = app.exec_()
    logging.getLogger("ProxyPlayer").info(f"Завершено с кодом {exit_code}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()