#!/usr/bin/env python3
"""
drift_monitor.py – расширенный мониторинг синхронизации.
Анализирует BUFFER_STATS, PLAY_BUFFER_STATE, AUDIO_FIRST_READ, AUDIO_OUTPUT_START.
Показывает тренд рассинхрона и содержимое буферов на момент Play.
"""

import socket
import time
import csv
from datetime import datetime
from collections import deque
import numpy as np

UDP_IP = "127.0.0.1"
UDP_PORT = 18081
CSV_FILE = "drift_log.csv"

ANALYSIS_WINDOW = 30.0
PRINT_INTERVAL = 1.0
DRIFT_ALARM_THRESHOLD = 20.0  # мс/с

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"Drift Monitor (расширенный) запущен на {UDP_IP}:{UDP_PORT}")
    print(f"Лог: {CSV_FILE}")

    history = deque(maxlen=1000)
    last_print_time = time.time()
    audio_output_start = None
    first_read = False

    with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Event', 'Details', 'Desync_ms', 'Drift_ms_per_sec'])

        try:
            while True:
                data, _ = sock.recvfrom(4096)
                msg = data.decode('utf-8').strip()
                if '|' not in msg:
                    continue

                event, details = msg.split('|', 1)
                ts = time.time()

                # --- Обработка PLAY_BUFFER_STATE ---
                if event == "PLAY_BUFFER_STATE":
                    print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] "
                          f"PLAY_BUFFER_STATE: {details}")
                    writer.writerow([datetime.now().strftime('%H:%M:%S.%f'), event, details, "", ""])

                # --- Обработка AUDIO_OUTPUT_START ---
                elif event == "AUDIO_OUTPUT_START":
                    audio_output_start = ts
                    print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] "
                          f"AUDIO_OUTPUT_START: {details}")
                    writer.writerow([datetime.now().strftime('%H:%M:%S.%f'), event, details, "", ""])

                # --- Обработка AUDIO_FIRST_READ ---
                elif event == "AUDIO_FIRST_READ":
                    delay = ""
                    if audio_output_start is not None and not first_read:
                        delay = f" задержка {(ts - audio_output_start)*1000:.1f} мс"
                        first_read = True
                    print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] "
                          f"AUDIO_FIRST_READ: {details}{delay}")
                    writer.writerow([datetime.now().strftime('%H:%M:%S.%f'), event, details, delay, ""])

                # --- Обработка BUFFER_STATS (стандартная) ---
                elif event == "BUFFER_STATS":
                    parts = details.split()
                    try:
                        vpts = int(parts[5].split('=')[1])
                        aclock = int(parts[6].split('=')[1])
                    except (IndexError, ValueError):
                        continue

                    if vpts == -1:
                        continue

                    desync_samples = aclock - vpts
                    desync_ms = desync_samples * 1000.0 / 48000.0

                    history.append((ts, desync_ms))

                    # Периодический вывод тренда
                    if ts - last_print_time >= PRINT_INTERVAL:
                        last_print_time = ts
                        cutoff = ts - ANALYSIS_WINDOW
                        while history and history[0][0] < cutoff:
                            history.popleft()

                        drift_str = "N/A"
                        if len(history) >= 2:
                            times = np.array([t for t, _ in history])
                            desyncs = np.array([d for _, d in history])
                            a, _ = np.polyfit(times, desyncs, 1)
                            drift_rate = a * 1000.0
                            drift_str = f"{drift_rate:+.2f}"
                            if abs(drift_rate) > DRIFT_ALARM_THRESHOLD:
                                print(f"[!] ТРЕВОГА: сильный дрейф {drift_rate:+.2f} мс/с")

                        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                              f"Рассинхрон: {desync_ms:.1f} мс, "
                              f"Тренд: {drift_str} мс/с "
                              f"(aclock={aclock}, vpts={vpts})")

                        writer.writerow([
                            datetime.now().strftime('%H:%M:%S.%f'),
                            event, details,
                            f"{desync_ms:.1f}",
                            drift_str
                        ])

        except KeyboardInterrupt:
            print("\nМониторинг остановлен.")
            if len(history) > 1:
                times = np.array([t for t, _ in history])
                desyncs = np.array([d for _, d in history])
                a, _ = np.polyfit(times, desyncs, 1)
                print(f"Финальный тренд: {a*1000:+.2f} мс/с")
                print(f"Средний рассинхрон: {np.mean(desyncs):.1f} мс")

if __name__ == '__main__':
    main()