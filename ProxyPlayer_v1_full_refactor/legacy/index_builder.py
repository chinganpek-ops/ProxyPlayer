#!/usr/bin/env python3
"""
index_builder.py – фоновый процесс синхронизации локального зеркала idx.
Только докачивает удалённый .idx в локальное зеркало через prepare_mirror.
Никакого построения цепочек или кэша – это задача плеера.
"""

import sys
import time
import threading
from pathlib import Path

# Добавляем корень проекта в sys.path (для запуска как отдельный процесс)
if getattr(sys, 'frozen', False):
    base_dir = sys._MEIPASS
else:
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from idx_cache import prepare_mirror

import logging
logger = logging.getLogger(__name__)


class IndexBuilder:
    """
    Фоновый синхронизатор зеркала idx.
    Следит за ростом удалённого .idx и докачивает его в локальное зеркало.
    """

    def __init__(self, idx_path: Path, poll_interval: float = 1.0):
        """
        Args:
            idx_path: путь к удалённому .idx файлу
            poll_interval: интервал проверки в секундах
        """
        self.idx_path = idx_path
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        """Запускает фоновую синхронизацию."""
        # Первичная синхронизация
        logger.info("IndexBuilder: первичная синхронизация %s", self.idx_path)
        try:
            prepare_mirror(self.idx_path)
        except Exception as e:
            logger.error("Ошибка первичной синхронизации: %s", e)

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("IndexBuilder запущен (интервал %.1f с)", self.poll_interval)

    def stop(self):
        """Останавливает фоновую синхронизацию."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("IndexBuilder остановлен")

    def _monitor_loop(self):
        """Основной цикл: периодически докачивает зеркало."""
        while not self._stop_event.is_set():
            try:
                prepare_mirror(self.idx_path)
            except Exception as e:
                logger.error("Ошибка синхронизации: %s", e)
            # Ждём с учётом остановки
            self._stop_event.wait(self.poll_interval)


def main():
    """Точка входа для запуска как отдельный процесс."""
    if len(sys.argv) < 2:
        print("Usage: index_builder.py <idx_path> [poll_interval]")
        sys.exit(1)

    idx_path = Path(sys.argv[1])
    poll_interval = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    builder = IndexBuilder(idx_path, poll_interval)
    builder.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        builder.stop()


if __name__ == "__main__":
    main()