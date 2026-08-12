#!/usr/bin/env python3
"""
diagnose.py – диагностика индексов и смещений в ProxyPlayer v1.
Запуск: python diagnose.py <путь_к_mp4> [--start-frame N] [--live]
"""

import sys
import os
from pathlib import Path

# Добавляем корень проекта в sys.path, если скрипт запущен не из него
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.timebase import SAMPLES_PER_VIDEO_FRAME, FRAMES_PER_CHUNK
from index.idx_cache import open_idx_mmap, get_mirror_path, prepare_mirror
from index.lazy_index import LazyIndex
from index.moov_builder import (
    build_chunks_from_cached_offsets,
    _abs_offset,
    fast_video_records,
    get_idr_indices_from_mmap,
    build_audio_chunks_in_range,
    build_audio_tracks,
    DEFAULT_TRACK_FILTER,
    DTYPE_193,
)
from file_io.win_sequential_reader import WinSequentialReader
from utils.utils import get_real_size, find_ref_path

import numpy as np

LIVE_SEEK_OFFSET_FRAMES = 1600

def diagnose(mp4_path: Path, start_frame: int = None, use_live: bool = False):
    # 1. Находим .idx
    idx_path = mp4_path.parent / "idx" / "mp4" / f"{mp4_path.stem}.idx"
    if not idx_path.exists():
        idx_path = mp4_path.parent / f"{mp4_path.stem}.idx"
    if not idx_path.exists():
        print(f"ERROR: .idx not found for {mp4_path}")
        return

    # 2. Готовим зеркало (можно использовать существующее, чтобы не ждать)
    mirror_path = get_mirror_path(idx_path)
    if not mirror_path.exists():
        print("Mirror not ready, running prepare_mirror (may take time)...")
        mirror_path = prepare_mirror(idx_path)
    print(f"Mirror: {mirror_path}")

    # 3. Открываем mmap и получаем полные записи
    mm_193, mm_c9 = open_idx_mmap(mirror_path)
    print(f"mm_193 entries: {len(mm_193)}, mm_c9 entries: {len(mm_c9)}")

    # 4. Фильтруем видео
    video_full = fast_video_records(mm_193)
    total_frames = len(video_full)
    print(f"Total video frames: {total_frames}")

    # 5. Определяем стартовый кадр
    if start_frame is None:
        if use_live:
            start_frame = max(0, total_frames - LIVE_SEEK_OFFSET_FRAMES)
        else:
            start_frame = 0
    start_frame = max(0, min(start_frame, total_frames - 1))
    print(f"Start frame (global): {start_frame}")

    # 6. Строим окно вручную (как в LazyIndex, но без класса)
    window_seconds = 300.0
    half_frames = int(window_seconds * 48000 / 2) // 1920
    win_start = max(0, start_frame - half_frames)
    win_end = min(total_frames, start_frame + half_frames)
    # гарантируем минимальный размер окна
    min_frames = 100 * FRAMES_PER_CHUNK
    if win_end - win_start < min_frames:
        win_end = min(total_frames, win_start + min_frames)

    video_slice = video_full[win_start:win_end]
    print(f"Window: global frames {win_start}..{win_end-1} ({len(video_slice)} frames)")

    # cached_offsets
    cached_offsets = _abs_offset(video_slice) if len(video_slice) > 0 else np.array([], dtype=np.uint64)
    print(f"cached_offsets length: {len(cached_offsets)}")

    # чанки
    mdat_end = get_real_size(str(mp4_path))
    chunk_offsets, chunk_sizes = build_chunks_from_cached_offsets(
        video_slice, cached_offsets, mdat_end
    )
    n_chunks = len(chunk_offsets)
    print(f"Number of chunks in window: {n_chunks}")
    print(f"First few chunk_offsets (global file offsets): {chunk_offsets[:5]}")

    # локальный чанк для стартового кадра
    local_start_chunk = (start_frame - win_start) // FRAMES_PER_CHUNK
    print(f"Local start chunk index: {local_start_chunk}")

    # Проверяем несколько чанков вокруг стартового
    for offset_idx in range(max(0, local_start_chunk - 1), min(n_chunks, local_start_chunk + 3)):
        local_chunk = offset_idx
        # Смещение первого кадра чанка через cached_offsets
        first_frame_local = local_chunk * FRAMES_PER_CHUNK
        if first_frame_local >= len(cached_offsets):
            continue
        expected_offset = int(cached_offsets[first_frame_local])
        chunk_offset_from_array = int(chunk_offsets[local_chunk])
        print(f"\n--- Chunk {local_chunk} (global chunk {win_start//FRAMES_PER_CHUNK + local_chunk}) ---")
        print(f"  chunk_offsets[{local_chunk}] = {chunk_offset_from_array}")
        print(f"  cached_offsets[first_frame={first_frame_local}] = {expected_offset}")
        print(f"  Match: {chunk_offset_from_array == expected_offset}")

        # Проверим первый кадр чанка в сырых данных (эмуляция чтения, если нужно)
        # Если есть доступ к файлу, прочитаем маленький кусок и покажем структуру NAL
        try:
            reader = WinSequentialReader(mp4_path, rate_limit=0, overlapped=False)
            size_to_read = min(1024, chunk_sizes[local_chunk])
            data = reader.read_sequential(expected_offset, size_to_read)
            if data:
                print(f"  First 16 bytes at expected_offset: {data[:16].hex()}")
                # Проверим наличие стартового кода
                if data[:4] == b'\x00\x00\x00\x01':
                    print("  Starts with NAL start code (good)")
            reader.close()
        except Exception as e:
            print(f"  Could not read data: {e}")

    # Дополнительная проверка IDR
    idr_local = get_idr_indices_from_mmap(video_slice)
    print(f"\nIDR frames (local indices) in window: {len(idr_local)}")
    if len(idr_local) > 0:
        print(f"  First few IDR: {idr_local[:5]}")

    # Проверка аудио
    audio_tracks = build_audio_tracks(mm_c9, track_filter=DEFAULT_TRACK_FILTER)
    if len(audio_tracks) > 0:
        audio_chunks = build_audio_chunks_in_range(audio_tracks, win_start // FRAMES_PER_CHUNK, n_chunks)
        print(f"Audio chunks in window: {len(audio_chunks)}")
        if local_start_chunk < len(audio_chunks):
            print(f"  Audio for chunk {local_start_chunk}: {audio_chunks[local_start_chunk]}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to MP4 file")
    parser.add_argument("--start-frame", type=int, default=None, help="Global frame to start at")
    parser.add_argument("--live", action="store_true", help="Use live offset (1600 frames from end)")
    args = parser.parse_args()

    mp4 = Path(args.file)
    if not mp4.exists():
        print(f"File not found: {mp4}")
        sys.exit(1)

    diagnose(mp4, args.start_frame, args.live)