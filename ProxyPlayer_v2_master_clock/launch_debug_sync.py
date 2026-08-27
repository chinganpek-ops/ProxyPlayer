#!/usr/bin/env python3
"""
launch_debug_sync.py – запускает плеер в debug-режиме с перехватом sync-сообщений.
Использование:
    python launch_debug_sync.py <mp4_path> [доп. аргументы...]
    или
    python launch_debug_sync.py -- <аргументы плеера>
"""

import sys
import os
import subprocess
import re
import threading
import time
from pathlib import Path

# Файл, куда пишем sync-сообщения
SYNC_LOG = "sync_debug.log"

def filter_output(stream, logfile):
    """Читает поток, ищет sync-события и пишет их в logfile, остальное выводит на экран."""
    with open(logfile, "a", encoding="utf-8") as f:
        for line in iter(stream.readline, b''):
            line = line.decode('utf-8', errors='replace').rstrip()
            if any(key in line for key in ["SYNC_VIDEO", "SYNC_AUDIO", "SYNC_SEEK_REF"]):
                f.write(line + "\n")
                f.flush()
            # Также можно выводить все сообщения на консоль
            print(line)

def main():
    # Очищаем предыдущий sync-лог
    Path(SYNC_LOG).unlink(missing_ok=True)

    # Аргументы: если первый аргумент "--", то остальное передаётся как есть,
    # иначе первый аргумент это mp4, остальные добавляются.
    if len(sys.argv) < 2:
        print("Usage: python launch_debug_sync.py <mp4_path> [доп. аргументы]")
        sys.exit(1)

    args = sys.argv[1:]
    if args[0] == "--":
        args = args[1:]

    # Включаем debug-режим для SyncManager через переменную окружения (если main.py его поддерживает)
    env = os.environ.copy()
    env["PYPLAYER_DEBUG_SYNC"] = "1"

    # Запускаем плеер
    cmd = [sys.executable, "main.py"] + args
    print(f"Запуск: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        bufsize=1,
    )

    # Запускаем потоки для чтения stdout и stderr (объединены)
    # (в Popen мы объединили stderr в stdout)
    threading.Thread(target=filter_output, args=(proc.stdout, SYNC_LOG), daemon=True).start()

    # Ожидаем завершения
    proc.wait()
    print("Плеер завершился.")

if __name__ == "__main__":
    main()