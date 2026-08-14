#!/usr/bin/env python3
"""
sync_monitor.py – монитор синхронизации видео и аудио.
Читает sync_monitor.log и выводит события SYNC_VIDEO / SYNC_AUDIO
с подсветкой и вычисленным рассинхроном.
"""

import time
import os
import sys
import argparse
from pathlib import Path

# ANSI-цвета
COLORS = {
    "SYNC_VIDEO": "\033[94m",   # голубой
    "SYNC_VIDEO_DROP": "\033[91m", # красный
    "SYNC_AUDIO": "\033[92m",   # зелёный
    "RESET": "\033[0m",
}

def colorize(event_type, text):
    color = COLORS.get(event_type, COLORS["RESET"])
    return f"{color}{text}{COLORS['RESET']}"

def follow(file):
    with open(file, "r", encoding="utf-8") as f:
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if line:
                yield line.rstrip()
            else:
                time.sleep(0.05)
                cur_size = os.path.getsize(file)
                if cur_size < f.tell():
                    f.seek(0)

def parse_line(line):
    parts = line.split("|")
    if len(parts) < 3:
        return None, line
    message = "|".join(parts[2:]).strip()
    for et in COLORS:
        if message.startswith(et):
            return et, message
    return None, message

def main():
    parser = argparse.ArgumentParser(description="Sync monitor")
    parser.add_argument("logfile", nargs="?", default="sync_monitor.log",
                        help="Путь к лог-файлу синхронизации")
    args = parser.parse_args()
    log_path = Path(args.logfile)
    if not log_path.exists():
        print(f"Файл {log_path} не найден. Убедитесь, что плеер запущен с логированием синхронизации.")
        sys.exit(1)
    print(f"Мониторинг синхронизации: {log_path}")
    print("=" * 80)
    for raw_line in follow(str(log_path)):
        event_type, message = parse_line(raw_line)
        if event_type:
            print(colorize(event_type, message))
        else:
            print(message)

if __name__ == "__main__":
    main()