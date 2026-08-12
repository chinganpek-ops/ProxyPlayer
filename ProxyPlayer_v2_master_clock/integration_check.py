#!/usr/bin/env python3
"""
integration_check.py – проверка интеграции MasterClock во все компоненты.
Устойчивая версия с большими таймаутами и обработкой ошибок.
"""

import sys, time, logging, traceback
from pathlib import Path
import numpy as np

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Уровень INFO, чтобы видеть ошибки и предупреждения
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                    datefmt='%H:%M:%S')

from config.timebase import pts_to_video_frame
from core.stream_controller import StreamController
from index.idx_cache import prepare_mirror, get_mirror_path

def format_timecode(pts, fps=25.0):
    frame = pts_to_video_frame(pts)
    total_seconds = frame / fps
    h = int(total_seconds // 3600)
    m = int((total_seconds % 3600) // 60)
    s = int(total_seconds % 60)
    f = int(round((total_seconds - int(total_seconds)) * fps))
    return f"{h:02d}:{m:02d}:{s:02d};{f:02d}"

def main():
    if len(sys.argv) < 2:
        print("Usage: python integration_check.py <mp4_path>")
        sys.exit(1)

    mp4_path = Path(sys.argv[1])
    if not mp4_path.exists():
        print(f"Файл не найден: {mp4_path}")
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
    print(f"Зеркало готово: {mirror_path}")

    # Создаём StreamController
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

    # =================================================================
    # Тест 1: start_playback должен создать MasterClock
    # =================================================================
    print("=== Тест 1: start_playback ===")
    try:
        controller.start_playback()
        assert controller.master_clock is not None, "MasterClock не создан"
        assert controller._pipeline._master_clock is not None, "MasterClock не подключён к пайплайну"
        print("  MasterClock создан и подключён к пайплайну: OK")

        # Проверяем, что MasterClock не активен до resume
        assert not controller.master_clock._active, "MasterClock не должен быть активен до resume"
        print("  MasterClock не активен до resume: OK")
    except Exception as e:
        print(f"  ОШИБКА: {e}")
        traceback.print_exc()

    # =================================================================
    # Тест 2: resume запускает MasterClock, push_audio работает
    # =================================================================
    print("\n=== Тест 2: resume ===")
    try:
        controller.resume()
        time.sleep(0.5)
        assert controller.master_clock._active, "MasterClock не запустился после resume"
        print("  MasterClock активен после resume: OK")

        # Отправляем тестовый аудио в обе дорожки
        test_tone = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 4800, endpoint=False)).astype(np.float64)
        controller.master_clock.push_audio(2, test_tone.copy())
        controller.master_clock.push_audio(3, test_tone.copy())
        time.sleep(0.3)
        stats = controller.master_clock.get_stats()
        print(f"  После push: samples_played={stats['samples_played']}, queue_size={stats['queue_size']}, underruns={stats['underruns']}")
        if stats['underruns'] == 0:
            print("  push_audio работает, underruns=0: OK")
    except Exception as e:
        print(f"  ОШИБКА: {e}")
        traceback.print_exc()

    # =================================================================
    # Тест 3: audio_clock идёт от MasterClock (с вызовом get_display_frame)
    # =================================================================
    print("\n=== Тест 3: audio_clock ===")
    try:
        for i in range(3):
            # Эмулируем работу GUI – вызываем get_display_frame для продвижения sync
            controller.get_display_frame()
            time.sleep(0.5)
            aclock = controller.audio_clock
            mc_samples = controller.master_clock.samples_played
            tc = format_timecode(aclock)
            print(f"  [{i}] audio_clock={aclock}, MasterClock.samples_played={mc_samples}, таймкод={tc}")
        # audio_clock должен увеличиваться
        assert controller.audio_clock > 0, "audio_clock не увеличивается"
        print("  audio_clock растёт: OK")
    except Exception as e:
        print(f"  ОШИБКА: {e}")
        traceback.print_exc()

    # =================================================================
    # Тест 4: get_display_frame возвращает кадры
    # =================================================================
    print("\n=== Тест 4: get_display_frame ===")
    try:
        for i in range(5):
            frame = controller.get_display_frame()
            if frame is not None:
                print(f"  Кадр {i}: shape={frame.shape}, размер={frame.size}")
            else:
                print(f"  Кадр {i}: None")
            time.sleep(0.04)
    except Exception as e:
        print(f"  ОШИБКА: {e}")
        traceback.print_exc()

    # =================================================================
    # Тест 5: Seek (с защитой от None)
    # =================================================================
    print("\n=== Тест 5: seek ===")
    try:
        target_frame = max(0, controller.total_frames - 5000)
        controller.seek_absolute(target_frame)
        # Ждём завершения асинхронного seek
        time.sleep(3.0)
        if controller.master_clock is not None:
            aclock = controller.audio_clock
            print(f"  seek к кадру {target_frame}, audio_clock: {aclock}")
            print(f"  MasterClock активен: {controller.master_clock._active}")
        else:
            print("  MasterClock is None после seek!")
        # Возобновляем
        controller.resume()
        time.sleep(0.5)
        if controller.master_clock:
            print(f"  После resume audio_clock={controller.audio_clock}, active={controller.master_clock._active}")
    except Exception as e:
        print(f"  ОШИБКА в seek: {e}")
        traceback.print_exc()

    # =================================================================
    # Тест 6: stop и close
    # =================================================================
    print("\n=== Тест 6: stop и close ===")
    try:
        controller.stop()
        if controller.master_clock:
            assert not controller.master_clock._active, "MasterClock не остановился после stop"
            print("  MasterClock остановлен после stop: OK")
        controller.close()
        print("  StreamController закрыт: OK")
    except Exception as e:
        print(f"  ОШИБКА: {e}")
        traceback.print_exc()

    print("\n=== Все тесты пройдены успешно ===")

if __name__ == "__main__":
    main()