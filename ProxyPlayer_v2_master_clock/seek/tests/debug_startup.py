#!/usr/bin/env python3
"""
Скрипт для отладки старта и наполнения буфера StreamController.
Запускает StreamController, запускает воспроизведение и каждую секунду выводит
состояние буферов, очередей, планировщика и потоков конвейера.
Завершается после появления первого кадра или по таймауту.
"""

import sys
import time
import logging
from pathlib import Path

# Настройка подробного логирования в консоль
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout,
)
logging.getLogger('av').setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.stream_controller import StreamController
from config.timebase import pts_to_video_frame


def monitor_startup(mp4_path_str: str, timeout: float = 20.0):
    mp4_path = Path(mp4_path_str)
    if not mp4_path.exists():
        print(f"ОШИБКА: файл {mp4_path} не существует")
        return

    stem = mp4_path.stem
    parent = mp4_path.parent
    idx_path = parent / "idx" / "mp4" / f"{stem}.idx"
    if not idx_path.exists():
        idx_path = parent / f"{stem}.idx"
    ref_path = parent / f"{stem}.mp4.ref"
    if not ref_path.exists():
        ref_path = parent / f"{stem}.ref"

    print(f"Создание StreamController для {mp4_path}")
    ctrl = StreamController(
        ref_path=ref_path if ref_path.exists() else None,
        idx_path=idx_path,
        mp4_path=mp4_path,
        start_from_live=False,
    )
    print("Ожидание инициализации...")
    if not ctrl._ready.wait(timeout=60):
        print("StreamController не готов за 60 секунд")
        return
    print("Инициализация завершена.")
    time.sleep(1)

    print("Запуск воспроизведения...")
    ctrl.start_playback()
    start_time = time.monotonic()
    frame_obtained = False

    while time.monotonic() - start_time < timeout:
        # Обновим состояние
        pipeline = ctrl._pipeline
        scheduler = pipeline._scheduler if pipeline else None
        playback = ctrl._playback

        print(f"\n--- t={time.monotonic() - start_time:.1f}s ---")
        print(f"  playback.playing={playback.playing if playback else 'N/A'}, paused={playback._paused if playback else 'N/A'}")
        print(f"  video_buffer.count={ctrl.buffer_main.count}, free_slots={ctrl.buffer_main.free_slots}")
        if scheduler:
            print(f"  scheduler: mode={scheduler._mode.name}, current_chunk={scheduler._current_chunk}, "
                  f"total_chunks={scheduler._total_chunks}, loaded={scheduler.loaded_count}")
        if pipeline:
            stages = pipeline._stages
            alive = {s.name: s.is_alive() for s in stages}
            print(f"  stages alive: {alive}")
            print(f"  queues: raw={pipeline._raw_queue.qsize()}, video_pkt={pipeline._video_queue.qsize()}, "
                  f"audio_pkt={pipeline._audio_queue.qsize()}")

        # Проверяем наличие кадра
        frame = ctrl.get_display_frame()
        if frame is not None:
            print(f"  >>> ПОЛУЧЕН КАДР: shape={frame.shape}, pts?={pts_to_video_frame(ctrl.audio_clock) if ctrl.audio_clock else 'N/A'}")
            frame_obtained = True
            break

        time.sleep(1.0)

    if not frame_obtained:
        print("\nКадр не был получен в течение таймаута.")
    ctrl.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python debug_startup.py <путь_к_mp4> [таймаут_сек]")
        sys.exit(1)
    mp4 = sys.argv[1]
    timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    monitor_startup(mp4, timeout)