#!/usr/bin/env python3
"""
main_debug_sync.py – запускает плеер в debug-режиме и анализирует синхронизацию.

Плеер запускается с переменной окружения PYPLAYER_DEBUG_SYNC=1, которая
включает запись sync-событий в файл sync_debug.log рядом с main.py.
После завершения плеера скрипт анализирует этот файл.

ВАЖНО: плеер запускается БЕЗ --managed, чтобы работал ManagerWindow,
который сам создаст IndexService и дочерний процесс PlayerWidget.
Дочерний процесс унаследует переменную окружения и создаст sync_debug.log.
"""

import sys
import os
import subprocess
import time
from pathlib import Path
import re
from collections import defaultdict

SYNC_LOG = "sync_debug.log"

SEEK_REF_RE = re.compile(r'SYNC_SEEK_REF pts=(\d+)')
VIDEO_DROP_RE = re.compile(r'SYNC_VIDEO_DROP pts=(\d+) audio_clock=(\d+) delta=(-?\d+)')
VIDEO_FUTURE_RE = re.compile(r'SYNC_VIDEO_FUTURE pts=(\d+) audio_clock=(\d+) delta=(-?\d+)')
VIDEO_RE = re.compile(r'SYNC_VIDEO pts=(\d+) audio_clock=(\d+) delta=(-?\d+) buffer_count=(\d+)')
VIDEO_EMPTY_RE = re.compile(r'SYNC_VIDEO_EMPTY audio_clock=(\d+)')


def analyze_sync_log(log_path: Path):
    print(f"\nАнализ {log_path}...\n")

    sessions = []
    current_session = None

    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            m_seek = SEEK_REF_RE.search(line)
            if m_seek:
                if current_session:
                    sessions.append(current_session)
                current_session = {
                    'seek_pts': int(m_seek.group(1)),
                    'events': [],
                }
                continue

            if current_session is None:
                continue

            m_drop = VIDEO_DROP_RE.search(line)
            if m_drop:
                current_session['events'].append({
                    'type': 'DROP',
                    'pts': int(m_drop.group(1)),
                    'audio_clock': int(m_drop.group(2)),
                    'delta': int(m_drop.group(3)),
                })
                continue

            m_future = VIDEO_FUTURE_RE.search(line)
            if m_future:
                current_session['events'].append({
                    'type': 'FUTURE',
                    'pts': int(m_future.group(1)),
                    'audio_clock': int(m_future.group(2)),
                    'delta': int(m_future.group(3)),
                })
                continue

            m_video = VIDEO_RE.search(line)
            if m_video:
                current_session['events'].append({
                    'type': 'VIDEO',
                    'pts': int(m_video.group(1)),
                    'audio_clock': int(m_video.group(2)),
                    'delta': int(m_video.group(3)),
                    'buffer_count': int(m_video.group(4)),
                })
                continue

            m_empty = VIDEO_EMPTY_RE.search(line)
            if m_empty:
                current_session['events'].append({
                    'type': 'EMPTY',
                    'audio_clock': int(m_empty.group(1)),
                })
                continue

    if current_session:
        sessions.append(current_session)

    if not sessions:
        print("В логе не найдено событий SYNC_SEEK_REF (перемоток).")
        print("Убедитесь, что плеер был запущен с debug-режимом и производились перемотки.")
        return

    print(f"Найдено перемоток: {len(sessions)}")

    for i, session in enumerate(sessions):
        seek_pts = session['seek_pts']
        events = session['events']

        print(f"\n--- Сессия {i+1}: после перемотки к PTS {seek_pts} ---")

        counts = defaultdict(int)
        first_video_idx = None
        last_video_idx = None
        stall_point = None

        for idx, ev in enumerate(events):
            counts[ev['type']] += 1
            if ev['type'] == 'VIDEO':
                if first_video_idx is None:
                    first_video_idx = idx
                last_video_idx = idx

        consecutive_non_video = 0
        for idx, ev in enumerate(events):
            if ev['type'] != 'VIDEO':
                consecutive_non_video += 1
            else:
                consecutive_non_video = 0
            if consecutive_non_video >= 3:
                stall_point = idx
                break

        print(f"  Показано кадров (VIDEO):      {counts['VIDEO']}")
        print(f"  Отброшено как устаревшие (DROP): {counts['DROP']}")
        print(f"  Слишком рано (FUTURE):         {counts['FUTURE']}")
        print(f"  Пустой буфер (EMPTY):          {counts['EMPTY']}")

        if first_video_idx is not None and last_video_idx is not None:
            first_ev = events[first_video_idx]
            last_ev = events[last_video_idx]
            print(f"  Первый показанный кадр: pts={first_ev['pts']} delta={first_ev['delta']}")
            print(f"  Последний показанный кадр: pts={last_ev['pts']} delta={last_ev['delta']}")

            if stall_point is not None:
                stall_ev = events[stall_point]
                print(f"  ⚠ Остановка обнаружена после события #{stall_point}: "
                      f"тип={stall_ev['type']} audio_clock={stall_ev.get('audio_clock', '?')}")
                if stall_ev['type'] == 'FUTURE':
                    print("  Причина: кадры опережают audio_clock более чем на FUTURE_HORIZON.")
                    print("  Возможно, audio_clock не был сброшен при seek.")
                elif stall_ev['type'] == 'EMPTY':
                    print("  Причина: буфер кадров пуст. Декодер не успевает наполнять буфер.")
                elif stall_ev['type'] == 'DROP':
                    print("  Причина: все кадры отброшены как устаревшие.")
            else:
                print("  Остановка не обнаружена.")
        else:
            print("  Видео кадры не были показаны вообще.")

    print("\n" + "=" * 60)
    print("ВЫВОДЫ И РЕКОМЕНДАЦИИ")
    print("=" * 60)
    print("Если много SYNC_VIDEO_FUTURE — проверьте сброс audio_clock после seek.")
    print("Если много SYNC_VIDEO_EMPTY — декодер не успевает, увеличьте буфер.")
    print("Если много SYNC_VIDEO_DROP — возможно, seek_reference слишком велик.")
    print("=" * 60)


def main():
    # Удаляем старый лог рядом с main.py
    main_dir = Path(__file__).resolve().parent
    sync_log_path = main_dir / SYNC_LOG
    if sync_log_path.exists():
        sync_log_path.unlink()

    if len(sys.argv) < 2:
        print("Usage: python main_debug_sync.py <mp4_path> [доп. аргументы]")
        sys.exit(1)

    args = sys.argv[1:]
    if args[0] == "--":
        args = args[1:]

    # НЕ добавляем --managed — плеер должен запускаться в обычном режиме,
    # чтобы ManagerWindow поднял IndexService.
    env = os.environ.copy()
    env["PYPLAYER_DEBUG_SYNC"] = "1"

    cmd = [sys.executable, "main.py"] + args
    print(f"Запуск плеера: {' '.join(cmd)}")

    proc = subprocess.Popen(cmd, env=env)
    proc.wait()

    print("Плеер завершился.")

    if sync_log_path.exists() and sync_log_path.stat().st_size > 0:
        analyze_sync_log(sync_log_path)
    else:
        print(f"Файл {sync_log_path} пуст или не создан. Проверьте, что в main.py включён FileHandler для SyncMonitor.")


if __name__ == "__main__":
    main()