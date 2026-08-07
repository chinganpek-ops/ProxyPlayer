#!/usr/bin/env python3
"""
check_audio_clock_flow.py – проверка сквозного прохождения audio_clock
и его влияния на аудиобуферы. Запускает StreamController, эмулирует
нажатие Play и выводит ключевые значения.
"""

import sys, time, logging
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Включаем логирование для ключевых компонентов
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                    datefmt='%H:%M:%S')
logging.getLogger("core.playback_engine").setLevel(logging.DEBUG)
logging.getLogger("core.sync_manager").setLevel(logging.DEBUG)
logging.getLogger("buffer.audio_buffer").setLevel(logging.DEBUG)
logging.getLogger("output.audio_output").setLevel(logging.DEBUG)

from config.timebase import video_frame_to_pts, pts_to_video_frame
from core.stream_controller import StreamController
from index.idx_cache import prepare_mirror, get_mirror_path

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_audio_clock_flow.py <mp4_path>")
        sys.exit(1)

    mp4_path = Path(sys.argv[1])
    if not mp4_path.exists():
        print(f"File not found: {mp4_path}")
        sys.exit(1)

    # Индексы
    stem = mp4_path.stem
    parent = mp4_path.parent
    ref = parent / f"{stem}.mp4.ref"
    if not ref.exists():
        ref = parent / f"{stem}.ref"
    idx = parent / "idx" / "mp4" / f"{stem}.idx"
    if not idx.exists():
        idx = parent / f"{stem}.idx"

    # Зеркало
    print("Подготовка зеркала...")
    try:
        mirror_path = prepare_mirror(idx)
    except Exception:
        mirror_path = get_mirror_path(idx)
        if not mirror_path.exists():
            print("Зеркало не найдено")
            sys.exit(1)

    print("Создание StreamController (live)...")
    controller = StreamController(
        ref_path=ref, idx_path=idx, mp4_path=mp4_path,
        fps=25.0, buffer_size=800, start_from_live=True,
        mirror_path=str(mirror_path)
    )
    if not controller._ready.wait(timeout=60):
        print("StreamController не готов")
        sys.exit(1)
    print("StreamController готов.\n")

    # Запускаем start_playback (без GUI)
    print("=== start_playback ===")
    controller.start_playback()
    time.sleep(0.2)

    # Проверяем состояние буферов до resume
    print("\n--- После start_playback ---")
    print(f"audio_clock = {controller.audio_clock}")
    for tid in (2, 3):
        buf = controller.audio_buffers.buffers[tid]
        print(f"Track {tid}: read_pos={buf.read_pos}, write_pos={buf.write_pos}")

    # Эмулируем нажатие Play
    print("\n=== resume (нажатие Play) ===")
    controller.resume()
    time.sleep(0.3)

    print("\n--- Сразу после resume ---")
    print(f"audio_clock = {controller.audio_clock}")
    for tid in (2, 3):
        buf = controller.audio_buffers.buffers[tid]
        print(f"Track {tid}: read_pos={buf.read_pos}, write_pos={buf.write_pos}")

    # Ждём 2 секунды и смотрим динамику
    print("\n--- Динамика за 2 секунды ---")
    prev_clock = controller.audio_clock
    for i in range(4):
        time.sleep(0.5)
        clock = controller.audio_clock
        delta = clock - prev_clock
        prev_clock = clock
        print(f"t={i*0.5:.1f}s: audio_clock={clock}, delta={delta}")
        # Состояние буфера
        buf2 = controller.audio_buffers.buffers[2]
        print(f"  Track2: read_pos={buf2.read_pos}, available={buf2.available_read}, write_pos={buf2.write_pos}")

    # Проверяем, вызывается ли _update_clock_from_system
    print("\n--- Проверка обновления clock ---")
    if controller._playback:
        print(f"_clock_start_time={controller._playback._clock_start_time}")
        print(f"_clock_start_pts={controller._playback._clock_start_pts}")
        # Форсим вызов _update_clock_from_system
        controller._playback._update_clock_from_system()
        print(f"audio_clock после ручного _update_clock_from_system: {controller.audio_clock}")

    # Остановка
    controller.stop()
    controller.close()
    print("\nГотово.")

if __name__ == "__main__":
    main()