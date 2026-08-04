#!/usr/bin/env python3
"""
index_service.py – сервис индексирования для ProxyPlayer v1.
Запускается как отдельный процесс менеджером.
Единолично владеет зеркалом .idx, инкрементально обновляет его
и уведомляет плееры о готовности новых данных через IPC.
"""

import sys
import os
import time
import threading
import logging
from pathlib import Path

from PyQt5.QtCore import QObject, pyqtSignal, QTimer
from PyQt5.QtNetwork import QLocalServer, QLocalSocket

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger("IndexService")


class IndexService(QObject):
    """
    Сервис, управляющий зеркалом .idx.
    Работает в отдельном процессе, запускается менеджером.
    """

    # Сигналы для внутреннего использования
    mirror_ready = pyqtSignal(str)       # путь к зеркалу
    mirror_updated = pyqtSignal(str, int) # путь, новый размер

    def __init__(self, idx_path: str, poll_interval: float = 5.0):
        super().__init__()
        self.idx_path = Path(idx_path)
        self.poll_interval = poll_interval
        self._mirror_path: Path = None
        self._stop_event = threading.Event()
        self._server: QLocalServer = None
        self._clients: list = []  # список активных QLocalSocket клиентов

        # Таймер для периодической проверки роста
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._check_growth)

        # Подготавливаем зеркало при старте
        QTimer.singleShot(0, self._initial_sync)

    # ------------------------------------------------------------------
    def _initial_sync(self):
        """Первичная синхронизация зеркала и запуск IPC-сервера."""
        try:
            from index.idx_cache import prepare_mirror
            self._mirror_path = prepare_mirror(self.idx_path)
            logger.info(f"Зеркало готово: {self._mirror_path}")
            self.mirror_ready.emit(str(self._mirror_path))
            self._start_ipc_server()
            self._poll_timer.start(int(self.poll_interval * 1000))
        except Exception as e:
            logger.error(f"Ошибка первичной синхронизации: {e}")
            sys.exit(1)

    def _check_growth(self):
        """Проверяет, не вырос ли удалённый .idx, и докачивает зеркало."""
        try:
            from index.idx_cache import prepare_mirror
            new_path = prepare_mirror(self.idx_path)
            new_size = new_path.stat().st_size
            if new_size > self._mirror_path.stat().st_size:
                self._mirror_path = new_path
                logger.info(f"Зеркало обновлено до {new_size} байт")
                self.mirror_updated.emit(str(self._mirror_path), new_size)
                self._notify_clients(str(self._mirror_path))
        except Exception as e:
            logger.error(f"Ошибка проверки роста: {e}")

    # ------------------------------------------------------------------
    # IPC для уведомления плееров
    # ------------------------------------------------------------------
    def _start_ipc_server(self):
        """Запускает локальный сервер для связи с плеерами."""
        self._server = QLocalServer(self)
        self._server.newConnection.connect(self._on_new_ipc_connection)
        server_name = f"ProxyPlayerIndexService_{os.getpid()}"
        self._server.listen(server_name)
        logger.info(f"IPC-сервер запущен: {server_name}")

    def _on_new_ipc_connection(self):
        client = self._server.nextPendingConnection()
        if client:
            self._clients.append(client)
            client.disconnected.connect(lambda: self._cleanup_client(client))

    def _notify_clients(self, mirror_path: str):
        """Отправляет путь к обновлённому зеркалу всем подключённым клиентам."""
        dead = []
        for client in self._clients:
            if client.state() == QLocalSocket.ConnectedState:
                client.write(mirror_path.encode())
                client.flush()
            else:
                dead.append(client)
        for client in dead:
            self._cleanup_client(client)

    def _cleanup_client(self, client):
        if client in self._clients:
            self._clients.remove(client)
            client.deleteLater()

    # ------------------------------------------------------------------
    def stop(self):
        """Останавливает сервис."""
        self._stop_event.set()
        self._poll_timer.stop()
        if self._server:
            self._server.close()
        for client in self._clients:
            client.close()
        self._clients.clear()
        logger.info("IndexService остановлен")
        sys.exit(0)


# ------------------------------------------------------------------
# Точка входа для запуска как отдельный процесс
# ------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: index_service.py <idx_path> [poll_interval]")
        sys.exit(1)

    idx_path = sys.argv[1]
    poll_interval = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0

    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("ProxyPlayerIndexService")

    service = IndexService(idx_path, poll_interval)

    # Обработка Ctrl+C
    import signal
    signal.signal(signal.SIGINT, lambda sig, frame: service.stop())
    signal.signal(signal.SIGTERM, lambda sig, frame: service.stop())

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()