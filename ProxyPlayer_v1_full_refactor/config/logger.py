"""
logger.py – настройка логирования для ProxyPlayer v1.
"""

import sys
import logging


def setup_logging(
    level: int = logging.INFO,
    log_file: str = "player.log",
    capture_stdout: bool = True,
    console_output: bool = False,
    mode: str = 'w'
) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(threadName)-20s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    try:
        fh = logging.FileHandler(log_file, encoding="utf-8", mode=mode)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)
    except OSError as e:
        sys.stderr.write(f"Не удалось создать файл лога {log_file}: {e}\n")

    if console_output:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        root_logger.addHandler(ch)

    if capture_stdout:
        sys.stdout = _StreamToLogger(logging.getLogger("stdout"), logging.DEBUG)

    logging.getLogger("ProxyPlayer").info("=" * 60)
    logging.getLogger("ProxyPlayer").info("Логирование настроено. Старт сессии.")


class _StreamToLogger:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            if line:
                self.logger.log(self.level, line)

    def flush(self):
        pass