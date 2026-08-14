#!/usr/bin/env python3
"""
seek_monitor.py – монитор событий перемотки и буферов.
Запускается параллельно с плеером, читает лог-файл seek_monitor.log
и выводит события в реальном времени с подсветкой.

Использование:
    python seek_monitor.py [путь_к_лог_файлу]

По умолчанию лог-файл ищется в текущей директории (seek_monitor.log).
"""

import sys
import time
import os
import argparse
from datetime import datetime
from pathlib import Path

# ANSI-цвета для подсветки
COLORS = {
    "SEEK_START": "\033[93m",    # жёлтый
    "SEEK_COMPLETE": "\033[92m", # зелёный
    "FAST_SEEK": "\033[95m",     # пурпурный
    "FRAME_ADDED": "\033[94m",   # синий
    "PUSH_FRAME": "\033[96m",    # голубой
    "DISPLAY_FRAME": "\033[97m", # белый
    "WARNING": "\033[91m",       # красный
    "RESET": "\033[0m",
}

def colorize(event_type: str, text: str) -> str:
    color = COLORS.get(event_type, COLORS["RESET"])
    return f"{color}{text}{COLORS['RESET']}"

def follow(file):
    """Аналог tail -f, но с учётом ротации файла."""
    with open(file, "r", encoding="utf-8") as f:
        # Переходим в конец файла
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if line:
                yield line.rstrip()
            else:
                time.sleep(0.1)
                # Проверяем, не был ли файл пересоздан (ротация)
                current_size = os.path.getsize(file)
                if current_size < f.tell():
                    f.seek(0)
                else:
                    f.seek(f.tell())

def parse_line(line: str):
    """Извлекает тип события и сообщение."""
    parts = line.split("|")
    if len(parts) < 3:
        return None, line
    # Формат: asctime | message (мы логировали без уровня)
    # Поэтому message = всё после второго '|'
    message = "|".join(parts[2:]).strip()
    event_type = None
    for et in COLORS:
        if message.startswith(et):
            event_type = et
            break
    return event_type, message

def main():
    parser = argparse.ArgumentParser(description="Seek monitor")
    parser.add_argument("logfile", nargs="?", default="seek_monitor.log",
                        help="Путь к лог-файлу мониторинга")
    args = parser.parse_args()

    log_path = Path(args.logfile)
    if not log_path.exists():
        print(f"Файл {log_path} не найден. Убедитесь, что плеер запущен с включённым мониторингом.")
        sys.exit(1)

    print(f"Мониторинг файла: {log_path}")
    print("=" * 80)

    for raw_line in follow(str(log_path)):
        event_type, message = parse_line(raw_line)
        if event_type:
            print(colorize(event_type, message))
        else:
            print(message)

if __name__ == "__main__":
    main()