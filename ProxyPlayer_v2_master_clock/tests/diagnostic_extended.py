#!/usr/bin/env python3
"""
diagnostic_extended.py – полная диагностика индексов, чтения и декодирования.
Проверяет каждый этап: от построения окна до записи кадров в буфер.
"""

import sys
import os
import time
from pathlib import Path
import numpy as np

# Добавляем корень проекта
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.timebase import SAMPLES_PER_VIDEO_FRAME, FRAMES_PER_CHUNK
from index.idx_cache import open_idx_mmap, get_mirror_path, prepare_mirror
from index.lazy_index import LazyIndex
from index.moov_builder import (
    DTYPE_193, SEGMENT_SIZE,
    fast_video_records,
    build_audio_tracks, build_audio_chunks_in_range,
    DEFAULT_TRACK_FILTER,
    _abs_offset,
)
from file_io.win_sequential_reader import WinSequentialReader
from decode.decoder import Decoder
from decode.audio_decoder import AudioDecoder
from buffer.frame_buffer import FrameRingBuffer
from buffer.audio_buffer import MultiTrackAudioBuffer
from utils.utils import get_real_size

DEFAULT_AVCC = bytes.fromhex(
    "014d001fffe1002e674d401f9652816824dff80200016a50101014000003"
    "000400000300cb8180009600000301e848fc6383b428532c01000568e9093520"
)
DEFAULT_ASC = b'\x11\x88'
LIVE_SEEK_OFFSET_FRAMES = 1600

def diagnostic(mp4_path: Path, start_frame: int = None, use_live: bool = True):
    # 1. Индексные файлы
    idx_path = mp4_path.parent / "idx" / "mp4" / f"{mp4_path.stem}.idx"
    if not idx_path.exists():
        idx_path = mp4_path.parent / f"{mp4_path.stem}.idx"
    if not idx_path.exists():
        print("ERROR: .idx not found")
        return

    mirror_path = get_mirror_path(idx_path)
    if not mirror_path.exists():
        print("Mirror not ready, running prepare_mirror...")
        mirror_path = prepare_mirror(idx_path)
    print(f"Mirror: {mirror_path}")

    mm_193, mm_c9 = open_idx_mmap(mirror_path)
    print(f"Full 193 entries: {len(mm_193)}, C9 entries: {len(mm_c9)}")

    video_full = fast_video_records(mm_193)
    total_frames = len(video_full)
    print(f"Total video frames: {total_frames}")

    if start_frame is None:
        if use_live:
            start_frame = max(0, total_frames - LIVE_SEEK_OFFSET_FRAMES)
        else:
            start_frame = 0
    start_frame = max(0, min(start_frame, total_frames - 1))
    print(f"Start frame (global): {start_frame}")

    # 2. Строим окно через LazyIndex
    mdat_end = get_real_size(str(mp4_path))
    lazy_idx = LazyIndex(mirror_path, mp4_path, mdat_end)
    window = lazy_idx.open_window(start_frame)
    print(f"Window: frames {window.window_start_frame}-{window.window_end_frame-1}, "
          f"chunks {window.window_start_chunk}-{window.window_start_chunk + window.total_chunks - 1}")
    print(f"Window video records: {len(window.video_records)}")
    print(f"Window cached_offsets: {len(window.cached_offsets)}")
    print(f"Window chunk_offsets: {len(window.chunk_offsets)}")

    # 3. Сравнение локального окна с глобальным индексом
    # Берём глобальные записи для диапазона окна и сверяем поля
    global_slice = video_full[window.window_start_frame:window.window_end_frame]
    assert len(global_slice) == len(window.video_records), "Размеры не совпадают!"
    # Сравниваем ключевые поля
    if not np.array_equal(global_slice['f1'], window.video_records['f1']):
        print("ERROR: f1 mismatch between global and window!")
    if not np.array_equal(global_slice['f2'], window.video_records['f2']):
        print("ERROR: f2 mismatch between global and window!")
    if not np.array_equal(global_slice['f3'], window.video_records['f3']):
        print("ERROR: f3 mismatch between global and window!")
    print("Global vs local slice comparison: OK (f1,f2,f3 match)")

    # Сравниваем cached_offsets с вычисленными от глобального среза
    global_offsets = _abs_offset(global_slice)
    if not np.array_equal(global_offsets, window.cached_offsets):
        print("ERROR: cached_offsets differ from global offsets!")
    else:
        print("cached_offsets match global offsets: OK")

    # 4. Создаём ридер и декодеры
    reader = WinSequentialReader(mp4_path, rate_limit=0, overlapped=False)
    decoder = Decoder(DEFAULT_AVCC, mp4_path, thread_type="AUTO", thread_count=0,
                      skip_frame=False, gpu_mode="off")
    audio_decoders = [AudioDecoder(DEFAULT_ASC), AudioDecoder(DEFAULT_ASC)]

    # 5. Пробуем прочитать и декодировать несколько чанков вокруг стартового
    local_start_chunk = (start_frame - window.window_start_frame) // FRAMES_PER_CHUNK
    chunks_to_test = [local_start_chunk, local_start_chunk + 1, local_start_chunk + 2]
    print(f"\nTesting chunks (local indices): {chunks_to_test}")
    video_buffer = FrameRingBuffer(max_frames=800)
    audio_buffers = MultiTrackAudioBuffer(capacity_samples=1440000)

    for local_chunk in chunks_to_test:
        if local_chunk < 0 or local_chunk >= window.total_chunks:
            continue
        print(f"\n--- Local chunk {local_chunk} ---")
        # Определяем смещения, размер
        first_frame = local_chunk * FRAMES_PER_CHUNK
        if first_frame >= len(window.cached_offsets):
            print("  first_frame out of bounds, skip")
            continue
        chunk_start = int(window.cached_offsets[first_frame])
        next_frame = min(first_frame + FRAMES_PER_CHUNK, len(window.video_records))
        if next_frame < len(window.cached_offsets):
            chunk_end = int(window.cached_offsets[next_frame])
        else:
            chunk_end = chunk_start + window.chunk_sizes[local_chunk]
        size = max(0, chunk_end - chunk_start)
        size = int(size)
        print(f"  Read offset={chunk_start}, size={size}")
        data = reader.read_sequential(chunk_start, size)
        if not data:
            print("  ERROR: no data read")
            continue
        print(f"  Read {len(data)} bytes, first 16: {data[:16].hex()}")

        # Эмулируем демукс видео
        frame_count = 0
        for i in range(FRAMES_PER_CHUNK):
            abs_idx = window.window_start_chunk * FRAMES_PER_CHUNK + first_frame + i
            local_frame = first_frame + i
            if local_frame >= len(window.video_records) or local_frame >= len(window.cached_offsets):
                break
            abs_off = int(window.cached_offsets[local_frame])
            # размер кадра
            if i < FRAMES_PER_CHUNK - 1:
                next_local = local_frame + 1
                if next_local < len(window.cached_offsets):
                    next_off = int(window.cached_offsets[next_local])
                    frame_size = next_off - abs_off
                else:
                    frame_size = 0
            else:
                next_local = local_frame + 1
                if next_local < len(window.cached_offsets):
                    chunk_end2 = int(window.cached_offsets[next_local])
                else:
                    chunk_end2 = chunk_start + len(data)
                frame_size = chunk_end2 - abs_off
            if frame_size <= 0:
                continue
            rel = abs_off - chunk_start
            if rel < 0 or rel + frame_size > len(data):
                print(f"  Frame {i}: rel={rel} size={frame_size} out of bounds, skip")
                continue
            sample = data[rel:rel+frame_size]
            filtered = decoder.filter_avcc(sample)
            if not filtered:
                print(f"  Frame {i}: filter_avcc returned empty")
                continue
            try:
                frames = decoder.decode_sample(filtered)
                if frames:
                    for f in frames:
                        if f is not None and f.size > 0:
                            pts = abs_idx * SAMPLES_PER_VIDEO_FRAME
                            video_buffer.try_push(f, pts)
                            frame_count += 1
            except Exception as e:
                print(f"  Frame {i} decode error: {e}")
        print(f"  Decoded {frame_count} video frames from this chunk")

    print(f"\nFinal video buffer count: {video_buffer.count}")
    if video_buffer.count > 0:
        first = video_buffer.peek_first()
        if first:
            print(f"  First frame PTS: {first[0]}, shape: {first[1].shape}")
    else:
        print("  WARNING: no frames in buffer!")

    # Дополнительно проверим аудиочанки
    if local_start_chunk < len(window.audio_chunks):
        achunk = window.audio_chunks[local_start_chunk]
        print(f"\nAudio chunk {local_start_chunk}: {list(achunk.keys()) if achunk else 'empty'}")
        if achunk:
            for t in achunk:
                entries = achunk[t]
                print(f"  Track {t}: {len(entries)} entries")
                if entries:
                    e = entries[0]
                    print(f"    First entry: offset={e['abs_offset']}, size1={e['size1']}, size2={e['size2']}, pts={e['pts']}")

    reader.close()
    decoder.close()
    for ad in audio_decoders:
        ad.close()
    print("Done.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to MP4 file")
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--live", action="store_true", default=False)
    args = parser.parse_args()
    mp4 = Path(args.file)
    if not mp4.exists():
        print(f"File not found: {mp4}")
        sys.exit(1)
    diagnostic(mp4, args.start_frame, args.live)