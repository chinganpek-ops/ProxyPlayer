#!/usr/bin/env python3
"""
audio_monitor.py – монитор аудио-событий.
Запускается параллельно с плеером, читает audio_monitor.log и выводит события.
"""

import sys
import time
import os
import argparse
from pathlib import Path

# ANSI-цвета
COLORS = {
    "AUDIO_PUSH": "\033[94m",
    "AUDIO_PUSHED": "\033[96m",
    "AUDIO_QUEUE_HIGH": "\033[91m",
    "AUDIO_QUEUE_LOW": "\033[92m",
    "AUDIO_FLUSH": "\033[95m",
    "AUDIO_UNDERRUN": "\033[93m",
    "AUDIO_DECODER_WINDOW_SET": "\033[97m",
    "AUDIO_PACKET_OUTSIDE_WINDOW": "\033[90m",
    "RESET": "\033[0m",
}

def colorize(event_type: str, text: str) -> str:
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
    parser = argparse.ArgumentParser(description="Audio monitor")
    parser.add_argument("logfile", nargs="?", default="audio_monitor.log",
                        help="Путь к лог-файлу аудио")
    args = parser.parse_args()
    log_path = Path(args.logfile)
    if not log_path.exists():
        print(f"Файл {log_path} не найден. Убедитесь, что плеер запущен с включённым аудио-мониторингом.")
        sys.exit(1)
    print(f"Мониторинг аудио: {log_path}")
    print("=" * 80)
    for raw_line in follow(str(log_path)):
        event_type, message = parse_line(raw_line)
        if event_type:
            print(colorize(event_type, message))
        else:
            print(message)

if __name__ == "__main__":
    main()