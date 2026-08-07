#!/usr/bin/env python3
"""
audio_index_check.py – проверка согласованности глобальных/локальных индексов
для аудиобуферов после инициализации и запуска воспроизведения.
"""

import sys
import time
import logging
from pathlib import Path
import numpy as np

# Добавляем корень проекта
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Логирование – только важное
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("audio_index_check")

from config.timebase import video_frame_to_pts, pts_to_video_frame, SAMPLES_PER_CHUNK, FRAMES_PER_CHUNK
from core.stream_controller import StreamController
from index.idx_cache import prepare_mirror, get_mirror_path

def main():
    if len(sys.argv) < 2:
        print("Usage: python audio_index_check.py <mp4_path>")
        sys.exit(1)

    mp4_path = Path(sys.argv[1])
    if not mp4_path.exists():
        print(f"Файл не найден: {mp4_path}")
        sys.exit(1)

    # Индексные файлы
    stem = mp4_path.stem
    parent = mp4_path.parent
    ref = parent / f"{stem}.mp4.ref"
    if not ref.exists():
        ref = parent / f"{stem}.ref"
    idx = parent / "idx" / "mp4" / f"{stem}.idx"
    if not idx.exists():
        idx = parent / f"{stem}.idx"

    # Подготавливаем зеркало
    print("Подготовка зеркала индекса...")
    try:
        mirror_path = prepare_mirror(idx)
    except Exception:
        mirror_path = get_mirror_path(idx)
        if not mirror_path.exists():
            print("Зеркало не найдено")
            sys.exit(1)
    print(f"Зеркало: {mirror_path}")

    # Создаём StreamController с live-стартом
    print("Создание StreamController...")
    controller = StreamController(
        ref_path=ref,
        idx_path=idx,
        mp4_path=mp4_path,
        fps=25.0,
        buffer_size=800,
        start_from_live=True,
        mirror_path=str(mirror_path)
    )

    print("Ожидание инициализации...")
    if not controller._ready.wait(timeout=60):
        print("Ошибка: StreamController не готов")
        sys.exit(1)
    print("StreamController готов.\n")

    # Получаем ожидаемый глобальный стартовый кадр и его PTS
    total_frames = controller.total_frames
    live_offset = 1600
    expected_start_frame = max(0, total_frames - live_offset)
    expected_start_pts = video_frame_to_pts(expected_start_frame)
    print(f"Ожидаемый стартовый кадр: {expected_start_frame}")
    print(f"Ожидаемый стартовый PTS (audio_clock): {expected_start_pts}")

    # Проверяем состояние до start_playback
    print("\n--- ДО start_playback ---")
    print(f"audio_clock (PlaybackEngine): {controller.audio_clock}")
    if controller._playback:
        print(f"_audio_clock внутри PlaybackEngine: {controller._playback._audio_clock}")
    for tid in (2, 3):
        buf = controller.audio_buffers.buffers[tid]
        print(f"Буфер дорожки {tid}: read_pos={buf.read_pos}, write_pos={buf.write_pos}, available={buf.available_read}")

    # Запускаем воспроизведение (start_playback)
    print("\nВызов start_playback()...")
    controller.start_playback()
    time.sleep(0.2)

    print("--- ПОСЛЕ start_playback ---")
    print(f"audio_clock (PlaybackEngine): {controller.audio_clock}")
    if controller._playback:
        print(f"_audio_clock внутри PlaybackEngine: {controller._playback._audio_clock}")
    for tid in (2, 3):
        buf = controller.audio_buffers.buffers[tid]
        print(f"Буфер дорожки {tid}: read_pos={buf.read_pos}, write_pos={buf.write_pos}, available={buf.available_read}")

    # Проверяем, совпадает ли audio_clock с ожидаемым глобальным PTS
    if controller.audio_clock != expected_start_pts:
        print(f"\n!!! ВНИМАНИЕ: audio_clock ({controller.audio_clock}) != ожидаемый PTS ({expected_start_pts})")
    else:
        print("audio_clock соответствует ожидаемому глобальному PTS.")

    # Вызываем resume (как при нажатии Play)
    print("\nВызов resume()...")
    controller.resume()
    time.sleep(0.3)

    print("--- ПОСЛЕ resume ---")
    print(f"audio_clock (PlaybackEngine): {controller.audio_clock}")
    if controller._playback:
        print(f"_audio_clock внутри PlaybackEngine: {controller._playback._audio_clock}")
    for tid in (2, 3):
        buf = controller.audio_buffers.buffers[tid]
        print(f"Буфер дорожки {tid}: read_pos={buf.read_pos}, write_pos={buf.write_pos}, available={buf.available_read}")

    # Выводим, что должно было установиться в read_pos после reset_clock
    if controller.audio_output:
        # reset_clock устанавливает samples_written в позицию, и reset_all_read_to(position)
        print("\nОжидаемое read_pos после resume (должно быть равно audio_clock):",
              controller.audio_clock)
        for tid in (2, 3):
            buf = controller.audio_buffers.buffers[tid]
            if buf.read_pos != controller.audio_clock:
                print(f"!!! Дорожка {tid}: read_pos={buf.read_pos} != audio_clock={controller.audio_clock}")
            else:
                print(f"Дорожка {tid}: read_pos совпадает с audio_clock.")

    # Дополнительно: попробуем записать аудиоданные вручную и проверить try_write
    print("\nПопытка записи тестового аудио в буфер...")
    test_pts = controller.audio_clock  # глобальный PTS
    test_data = np.zeros(2048, dtype=np.float64)
    for tid in (2, 3):
        ok = controller.audio_buffers.try_write(tid, test_data, test_pts)
        print(f"Дорожка {tid}: try_write с pts={test_pts} -> {'OK' if ok else 'FAIL'}")
        if not ok:
            # Попробуем с локальным pts (ошибочным) для диагностики
            local_pts = test_pts % (1440000*48000)  # не имеет смысла, просто пример
            ok_local = controller.audio_buffers.try_write(tid, test_data, 0)
            print(f"  try_write с pts=0 -> {'OK' if ok_local else 'FAIL'} (для сравнения)")

    # Остановка
    controller.stop()
    controller.close()
    print("\nГотово.")

if __name__ == "__main__":
    main()