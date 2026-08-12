#!/usr/bin/env python3
"""
test_master_clock.py – проверка MasterClock: 3 сек левый канал, 3 сек правый.
"""

import sys
import time
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.master_clock import MasterClock

def main():
    print("Создание MasterClock...")
    clock = MasterClock(sample_rate=48000, buffer_size=1024)

    # Генерируем 0.1 с тона 440 Гц (моно)
    tone_duration = 0.1
    t = np.linspace(0, tone_duration, int(48000 * tone_duration), endpoint=False)
    tone = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

    print("Запуск воспроизведения...")
    clock.start()

    # --- Левый канал (дорожка 2) 3 секунды ---
    print("Воспроизведение 3 секунды в ЛЕВЫЙ канал (дорожка 2)...")
    start_time = time.monotonic()
    while time.monotonic() - start_time < 3.0:
        clock.push_audio(2, tone.copy())
        time.sleep(0.1)
        stats = clock.get_stats()
        print(f"\r[LEFT]  Сыграно: {stats['samples_played']:>8d} семплов "
              f"({stats['samples_played']/48000:.2f} с), "
              f"очередь: {stats['queue_size']:>4d}, "
              f"underruns: {stats['underruns']:>3d}", end='')
    print()

    # --- Правый канал (дорожка 3) 3 секунды ---
    print("Воспроизведение 3 секунды в ПРАВЫЙ канал (дорожка 3)...")
    start_time = time.monotonic()
    while time.monotonic() - start_time < 3.0:
        clock.push_audio(3, tone.copy())
        time.sleep(0.1)
        stats = clock.get_stats()
        print(f"\r[RIGHT] Сыграно: {stats['samples_played']:>8d} семплов "
              f"({stats['samples_played']/48000:.2f} с), "
              f"очередь: {stats['queue_size']:>4d}, "
              f"underruns: {stats['underruns']:>3d}", end='')
    print()

    clock.stop()
    clock.close()
    print("Тест завершён. Вы должны были услышать 3 сек слева, затем 3 сек справа.")

if __name__ == "__main__":
    main()