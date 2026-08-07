"""
logger.py – настройка логирования для Dalet Proxy Player v4 (production).
Поддерживает перезапись лог-файла при каждом запуске (mode='w').
Уровень логирования по умолчанию INFO.
"""

import sys
import logging
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: str = "dalet_player.log",
    capture_stdout: bool = True,
    console_output: bool = False,
    mode: str = 'a'      # 'a' – дописывать, 'w' – перезаписывать
) -> None:
    """
    Настраивает корневой логгер.
    - Все сообщения пишутся в файл (уровень level).
    - Опционально вывод в консоль (уровень INFO).
    - Опциональный перехват stdout (печатает как DEBUG).
    - Параметр mode позволяет перезаписывать лог при каждом запуске.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Удаляем существующие обработчики, чтобы избежать дублирования
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Единый формат для всех обработчиков
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(threadName)-20s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # Обработчик записи в файл
    try:
        fh = logging.FileHandler(log_file, encoding="utf-8", mode=mode)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)
        logging.getLogger(__name__).info(f"Логирование в файл: {log_file} (mode={mode})")
    except OSError as e:
        sys.stderr.write(f"Не удалось создать файл лога {log_file}: {e}\n")

    # Консольный обработчик (если запрошен)
    if console_output:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        root_logger.addHandler(ch)

    # Перехват stdout (для отладки print'ов из библиотек)
    if capture_stdout:
        sys.stdout = _StreamToLogger(logging.getLogger("stdout"), logging.DEBUG)

    logging.getLogger("DaletPlayer").info("=" * 60)
    logging.getLogger("DaletPlayer").info("Логирование настроено. Старт сессии.")


def add_gui_handler(handler: logging.Handler) -> None:
    """Добавляет GUI-обработчик в корневой логгер."""
    logging.getLogger().addHandler(handler)
    logging.getLogger(__name__).debug("GUI-обработчик логов добавлен")


def remove_gui_handler(handler: logging.Handler) -> None:
    """Удаляет GUI-обработчик из корневого логгера."""
    logging.getLogger().removeHandler(handler)
    logging.getLogger(__name__).debug("GUI-обработчик логов удалён")


class _StreamToLogger:
    """Перенаправляет stdout в логгер (для перехвата print'ов)."""

    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.level = level
        self._buffer = ""

    def write(self, buf: str):
        for line in buf.rstrip().splitlines():
            if line:
                self.logger.log(self.level, line)

    def flush(self):
        pass