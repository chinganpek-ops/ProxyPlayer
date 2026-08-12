#!/usr/bin/env python3
"""
integration_test.py – расширенный тест с принудительным DEBUG-выводом.
"""

import sys, os, time, logging
from pathlib import Path
import numpy as np

# --- Принудительный DEBUG для всех логгеров ---
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(threadName)-20s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stderr  # вывод в stderr, чтобы не перехватывался stdout
)
# Убеждаемся, что корневой логгер в DEBUG
logging.getLogger().setLevel(logging.DEBUG)

# Теперь импортируем всё остальное
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.logger import setup_logging
from config.config import load_config
from config.timebase import SAMPLES_PER_VIDEO_FRAME, pts_to_video_frame, FRAMES_PER_CHUNK
from core.stream_controller import StreamController
from buffer.frame_buffer import FrameRingBuffer
from buffer.audio_buffer import MultiTrackAudioBuffer
from pipeline.chunk_pipeline import ChunkPipeline
from pipeline.stream_scheduler import StreamScheduler, PlaybackMode
from pipeline.adaptive_chunk import AdaptiveChunkStrategy
from index.lazy_index import LazyIndex
from index.idx_cache import prepare_mirror, get_mirror_path

def print_separator(title=""):
    print("\n" + "="*60)
    if title:
        print(f"  {title}")
        print("="*60)

def main():
    if len(sys.argv) < 2:
        print("Usage: python integration_test.py <mp4_path> [--start-frame N]")
        sys.exit(1)

    mp4_path = Path(sys.argv[1])
    if not mp4_path.exists():
        print(f"File not found: {mp4_path}")
        sys.exit(1)

    start_frame_arg = None
    if '--start-frame' in sys.argv:
        idx = sys.argv.index('--start-frame')
        if idx + 1 < len(sys.argv):
            start_frame_arg = int(sys.argv[idx+1])

    config_path = Path.home() / "AppData" / "Roaming" / "ProxyPlayer" / "player_config.json"
    config = load_config(config_path)

    stem = mp4_path.stem
    parent = mp4_path.parent
    ref = parent / f"{stem}.mp4.ref"
    if not ref.exists():
        ref = parent / f"{stem}.ref"
    idx = parent / "idx" / "mp4" / f"{stem}.idx"
    if not idx.exists():
        idx = parent / f"{stem}.idx"

    print_separator("Подготовка зеркала")
    try:
        mirror_path = prepare_mirror(idx)
    except Exception as e:
        mirror_path = get_mirror_path(idx)
        if not mirror_path.exists():
            print("Зеркало не найдено")
            sys.exit(1)

    print_separator("Создание StreamController")
    controller = StreamController(
        ref_path=ref, idx_path=idx, mp4_path=mp4_path,
        fps=25.0, buffer_size=800,
        start_from_live=(start_frame_arg is None),
        mirror_path=str(mirror_path)
    )

    if not controller._ready.wait(timeout=60):
        print("StreamController не готов")
        sys.exit(1)

    # Определяем окно
    if start_frame_arg is not None:
        controller.seek_absolute(start_frame_arg)
        time.sleep(2.0)
        window = controller._lazy_index.window
    else:
        total = len(controller._lazy_index._video_records_full)
        start_frame = max(0, total - 1600)
        window = controller._lazy_index.open_window(start_frame)

    print_separator("Ручной тест планировщика")
    local_start = (start_frame - window.window_start_frame) // FRAMES_PER_CHUNK
    strategy = AdaptiveChunkStrategy()
    scheduler = StreamScheduler(strategy)
    video_buf = FrameRingBuffer(max_frames=800)
    scheduler.set_buffer(video_buf)
    scheduler.set_normal_mode(local_start, window.total_chunks)
    for i in range(5):
        chunk = scheduler.get_next_chunk()
        print(f"  get_next_chunk #{i}: {chunk}")

    print_separator("Запуск пайплайна")
    controller._pipeline.update_window(window)
    controller._playback.start_playback(start_frame, window.window_start_frame)

    print("Ожидание наполнения буфера...")
    waited = 0
    while controller.buffer_main.count == 0 and waited < 10:
        time.sleep(0.5)
        waited += 0.5
    print(f"Буфер: {controller.buffer_main.count} кадров")

    controller.stop()
    controller.close()

if __name__ == "__main__":
    main()