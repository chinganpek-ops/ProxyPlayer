#!/usr/bin/env python3
"""
sync_stall_analyzer.py – анализ причин остановки видео при бегущем таймкоде.

Читает sync_monitor.log (создаётся SyncManager'ом) и для каждой перемотки
(событие SYNC_SEEK_REF) анализирует последующие события синхронизации:
- сколько кадров было показано (SYNC_VIDEO)
- сколько отброшено как устаревшие (SYNC_VIDEO_DROP)
- сколько было "будущих" кадров (SYNC_VIDEO_FUTURE)
- сколько раз буфер оказывался пустым (SYNC_VIDEO_EMPTY)
Выявляет паттерн, при котором после перемотки видео начинает показывать
кадры, но вскоре останавливается на keep_last (таймкод продолжает идти).

Запуск:
    python sync_stall_analyzer.py [путь_к_sync_monitor.log]

Если путь не указан, используется sync_monitor.log в текущей директории.
"""

import re
import sys
from pathlib import Path
from collections import defaultdict

# Регулярки для событий
SEEK_REF_RE = re.compile(r'SYNC_SEEK_REF pts=(\d+)')
VIDEO_DROP_RE = re.compile(r'SYNC_VIDEO_DROP pts=(\d+) audio_clock=(\d+) delta=(-?\d+)')
VIDEO_FUTURE_RE = re.compile(r'SYNC_VIDEO_FUTURE pts=(\d+) audio_clock=(\d+) delta=(-?\d+)')
VIDEO_RE = re.compile(r'SYNC_VIDEO pts=(\d+) audio_clock=(\d+) delta=(-?\d+) buffer_count=(\d+)')
VIDEO_EMPTY_RE = re.compile(r'SYNC_VIDEO_EMPTY audio_clock=(\d+)')
VIDEO_SEEK_FILTER_RE = re.compile(r'SYNC_VIDEO_SEEK_FILTER pts=(\d+)')


def analyze_log(log_path: Path):
    """Анализирует лог и возвращает список сессий после каждой перемотки."""
    print(f"Анализ {log_path}...\n")

    sessions = []  # каждая сессия: {'seek_pts': int, 'events': list}
    current_session = None

    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Начало новой перемотки
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

            # Пропускаем фильтрацию по seek_reference (это внутреннее событие)
            if VIDEO_SEEK_FILTER_RE.search(line):
                # не добавляем, но можно учесть отдельно
                continue

            # Обрабатываем основные события
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

    return sessions


def print_session_summary(session, idx):
    """Выводит сводку по одной сессии (после одной перемотки)."""
    seek_pts = session['seek_pts']
    events = session['events']

    print(f"\n--- Сессия {idx+1}: после перемотки к PTS {seek_pts} ---")

    # Счётчики
    counts = defaultdict(int)
    first_video_idx = None
    last_video_idx = None
    stall_point = None

    for i, ev in enumerate(events):
        counts[ev['type']] += 1
        if ev['type'] == 'VIDEO':
            if first_video_idx is None:
                first_video_idx = i
            last_video_idx = i

    # Определяем момент, где после серии VIDEO идут FUTURE/EMPTY подряд
    consecutive_non_video = 0
    for i, ev in enumerate(events):
        if ev['type'] != 'VIDEO':
            consecutive_non_video += 1
        else:
            consecutive_non_video = 0

        # Если 3+ подряд не-VIDEO, считаем это точкой остановки
        if consecutive_non_video >= 3:
            stall_point = i
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
            # Анализируем, почему остановился
            if stall_ev['type'] == 'FUTURE':
                print("  Причина: кадры опережают audio_clock более чем на FUTURE_HORIZON "
                      "(960 сэмплов). Возможно, audio_clock не был сброшен при seek.")
            elif stall_ev['type'] == 'EMPTY':
                print("  Причина: буфер кадров пуст. Декодер не успевает наполнять буфер "
                      "после перемотки.")
            elif stall_ev['type'] == 'DROP':
                print("  Причина: все кадры отброшены как устаревшие. Возможно, "
                      "seek_reference слишком большой, или audio_clock ушёл вперёд.")
        else:
            print("  Остановка не обнаружена (все события в норме).")
    else:
        print("  Видео кадры не были показаны вообще.")


def main():
    # Путь к логу
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    else:
        log_path = Path("sync_monitor.log")

    if not log_path.exists():
        print(f"Файл {log_path} не найден.")
        sys.exit(1)

    sessions = analyze_log(log_path)

    if not sessions:
        print("В логе не найдено событий SYNC_SEEK_REF (перемоток).")
        return

    print(f"Найдено перемоток: {len(sessions)}")

    for i, session in enumerate(sessions):
        print_session_summary(session, i)

    # Итоговая рекомендация
    print("\n" + "=" * 60)
    print("ВЫВОДЫ И РЕКОМЕНДАЦИИ")
    print("=" * 60)
    print("Если после перемотки много SYNC_VIDEO_FUTURE — проверьте, что audio_clock")
    print("корректно устанавливается в PTS первого кадра после seek (set_clock).")
    print("Если много SYNC_VIDEO_EMPTY — декодер не успевает. Увеличьте буфер или")
    print("проверьте, что fill_buffer действительно получает кадры после переключения.")
    print("Если много SYNC_VIDEO_DROP — возможно, seek_reference слишком велик.")
    print("=" * 60)


if __name__ == "__main__":
    main()