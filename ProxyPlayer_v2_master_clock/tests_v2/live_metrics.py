#!/usr/bin/env python3
"""
live_metrics.py – запускает плеер, эмулирует работу GUI и выводит метрики.
"""

import sys, time, logging
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.WARNING, format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s', datefmt='%H:%M:%S')

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
        print("Usage: python live_metrics.py <mp4_path>")
        sys.exit(1)

    mp4_path = Path(sys.argv[1])
    if not mp4_path.exists():
        print(f"File not found: {mp4_path}")
        sys.exit(1)

    stem = mp4_path.stem
    parent = mp4_path.parent
    ref = parent / f"{stem}.mp4.ref"
    if not ref.exists():
        ref = parent / f"{stem}.ref"
    idx = parent / "idx" / "mp4" / f"{stem}.idx"
    if not idx.exists():
        idx = parent / f"{stem}.idx"

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

    print("=== start_playback ===")
    controller.start_playback()
    time.sleep(0.3)

    print("=== resume (Play) ===")
    controller.resume()
    time.sleep(0.3)

    print("\nСбор метрик (10 секунд):")
    print(f"{'Time':>6s} | {'audio_clock':>12s} | {'Таймкод':>11s} | {'VB cnt':>6s} | {'AB2 avail':>10s} | {'AB2 read':>10s} | {'AB3 avail':>10s} | {'raw_q':>6s} | {'vid_q':>6s} | {'aud_q':>6s} | {'Playing':>8s}")
    print("-" * 130)

    start = time.monotonic()
    for i in range(20):
        time.sleep(0.5)
        # Эмулируем вызов рендер‑таймера
        controller.get_display_frame()
        
        now = time.monotonic() - start
        aclock = controller.audio_clock
        tc = format_timecode(aclock)
        vb_cnt = controller.buffer_main.count
        ab2 = controller.audio_buffers.buffers[2]
        ab3 = controller.audio_buffers.buffers[3]

        if controller._pipeline:
            raw_q = controller._pipeline._raw_queue.qsize()
            vid_q = controller._pipeline._video_queue.qsize()
            aud_q = controller._pipeline._audio_queue.qsize()
        else:
            raw_q = vid_q = aud_q = -1

        print(f"{now:5.1f}s | {aclock:>12d} | {tc:>11s} | {vb_cnt:>6d} | {ab2.available_read:>10d} | {ab2.read_pos:>10d} | {ab3.available_read:>10d} | {raw_q:>6d} | {vid_q:>6d} | {aud_q:>6d} | {str(controller.playing):>8s}")

    controller.stop()
    controller.close()
    print("\nГотово.")

if __name__ == "__main__":
    main()