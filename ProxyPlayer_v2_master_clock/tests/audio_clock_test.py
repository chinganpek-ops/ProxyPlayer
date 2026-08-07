#!/usr/bin/env python3
"""
audio_clock_test.py – проверка обновления audio_clock и потребления аудиоданных.
Запускает StreamController, включает воспроизведение и в реальном времени
выводит audio_clock, таймкод, заполнение аудиобуферов и позицию чтения.
"""

import sys
import os
import time
import logging
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Принудительный DEBUG для аудио и движка
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stderr
)
logging.getLogger("core.playback_engine").setLevel(logging.DEBUG)
logging.getLogger("output.audio_output").setLevel(logging.DEBUG)
logging.getLogger("buffer.audio_buffer").setLevel(logging.DEBUG)

from config.config import load_config
from config.timebase import pts_to_video_frame
from core.stream_controller import StreamController
from index.idx_cache import prepare_mirror, get_mirror_path

def main():
    if len(sys.argv) < 2:
        print("Usage: python audio_clock_test.py <mp4_path>")
        sys.exit(1)

    mp4_path = Path(sys.argv[1])
    if not mp4_path.exists():
        print(f"File not found: {mp4_path}")
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

    config_path = Path.home() / "AppData" / "Roaming" / "ProxyPlayer" / "player_config.json"
    config = load_config(config_path)

    # Подготавливаем зеркало
    print("Подготовка зеркала индекса...")
    try:
        mirror_path = prepare_mirror(idx)
    except Exception:
        mirror_path = get_mirror_path(idx)
        if not mirror_path.exists():
            print("Зеркало не найдено")
            sys.exit(1)
    print(f"Зеркало готово: {mirror_path}")

    # Создаём контроллер
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

    # Ждём готовности
    print("Ожидание инициализации...")
    if not controller._ready.wait(timeout=60):
        print("StreamController не готов")
        sys.exit(1)
    print("StreamController готов.")

    # Запускаем воспроизведение
    print("Запуск воспроизведения...")
    controller.start_playback()
    time.sleep(1.0)

    # Включаем воспроизведение (нажимаем Play)
    print("Нажатие Play...")
    controller.resume()
    time.sleep(0.5)

    # Мониторинг в течение 10 секунд
    print("\nМониторинг audio_clock и аудиобуферов (10 секунд):")
    print(f"{'Time':>8s} | {'audio_clock':>12s} | {'Таймкод':>11s} | {'Avail2':>7s} | {'Avail3':>7s} | {'Read2':>10s} | {'Read3':>10s} | {'Written2':>10s} | {'Written3':>10s}")
    print("-" * 120)

    start_time = time.monotonic()
    while time.monotonic() - start_time < 10.0:
        aclock = controller.audio_clock
        frame = pts_to_video_frame(aclock)
        total_seconds = frame / 25.0
        h = int(total_seconds // 3600)
        m = int((total_seconds % 3600) // 60)
        s = int(total_seconds % 60)
        f = int(round((total_seconds - int(total_seconds)) * 25.0))
        tc = f"{h:02d}:{m:02d}:{s:02d};{f:02d}"

        # Статистика аудиобуферов
        buf2 = controller.audio_buffers.buffers[2]
        buf3 = controller.audio_buffers.buffers[3]

        elapsed = time.monotonic() - start_time
        print(f"{elapsed:7.2f}с | {aclock:>12d} | {tc:>11s} | {buf2.available_read:>7d} | {buf3.available_read:>7d} | {buf2.read_pos:>10d} | {buf3.read_pos:>10d} | {buf2._total_written:>10d} | {buf3._total_written:>10d}")

        time.sleep(1.0)

    # Финальное состояние
    print("-" * 120)
    print(f"Финальный audio_clock: {controller.audio_clock}")
    print(f"Активен AudioOutput: {controller.audio_output._active if controller.audio_output else False}")

    # Остановка
    controller.stop()
    controller.close()
    print("Тест завершён.")

if __name__ == "__main__":
    main()