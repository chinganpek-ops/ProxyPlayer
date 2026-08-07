#!/usr/bin/env python3
"""
check_audio_chunks.py – проверка построения аудиочанков для live-окна.
"""

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from index.idx_cache import prepare_mirror, get_mirror_path, open_idx_mmap
from index.lazy_index import LazyIndex
from index.moov_builder import (
    fast_video_records,
    build_audio_tracks,
    DEFAULT_TRACK_FILTER,
    build_audio_chunks_in_range,
)
from utils.utils import get_real_size
from config.timebase import SAMPLES_PER_CHUNK, FRAMES_PER_CHUNK

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_audio_chunks.py <mp4_path>")
        sys.exit(1)

    mp4_path = Path(sys.argv[1])
    if not mp4_path.exists():
        print(f"Файл не найден: {mp4_path}")
        sys.exit(1)

    # Индексы
    stem = mp4_path.stem
    parent = mp4_path.parent
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
    print(f"Зеркало: {mirror_path}")

    # Открываем mmap
    mm_193, mm_c9 = open_idx_mmap(mirror_path)
    video_full = fast_video_records(mm_193)
    total_frames = len(video_full)
    print(f"Всего видео кадров: {total_frames}")

    # Live‑старт
    LIVE_SEEK_OFFSET_FRAMES = 1600
    start_frame = max(0, total_frames - LIVE_SEEK_OFFSET_FRAMES)
    print(f"Стартовый кадр: {start_frame}")

    mdat_end = get_real_size(str(mp4_path))
    lazy_idx = LazyIndex(mirror_path, mp4_path, mdat_end)

    # Получаем окно
    window = lazy_idx.open_window(start_frame)
    print(f"Окно: кадры {window.window_start_frame}-{window.window_end_frame-1}, чанков: {window.total_chunks}")

    # Параметры для построения аудиочанков
    start_chunk = window.window_start_chunk
    num_chunks = window.total_chunks
    print(f"\nПараметры для build_audio_chunks_in_range:")
    print(f"  start_chunk (глобальный) = {start_chunk}")
    print(f"  num_chunks = {num_chunks}")

    # Строим полные аудиотреки
    audio_tracks_full = build_audio_tracks(mm_c9, track_filter=DEFAULT_TRACK_FILTER)
    print(f"  audio_tracks_full записей = {len(audio_tracks_full)}")

    # Проверим, есть ли аудиоданные в нужном диапазоне PTS
    start_pts = start_chunk * SAMPLES_PER_CHUNK
    end_pts = start_pts + num_chunks * SAMPLES_PER_CHUNK
    print(f"\nДиапазон PTS для окна: [{start_pts}, {end_pts})")
    pts_array = audio_tracks_full['pts']
    mask = (pts_array >= start_pts) & (pts_array < end_pts)
    matching = np.sum(mask)
    print(f"Аудиозаписей в этом диапазоне: {matching}")
    if matching == 0:
        # Проверим, где вообще есть аудио
        min_pts = np.min(pts_array) if len(pts_array) > 0 else 0
        max_pts = np.max(pts_array) if len(pts_array) > 0 else 0
        print(f"  Минимальный PTS в audio_tracks_full: {min_pts}")
        print(f"  Максимальный PTS в audio_tracks_full: {max_pts}")
        print("  *** Аудиоданные не перекрываются с окном! ***")
    else:
        print("  Первые 5 PTS в диапазоне:")
        for p in pts_array[mask][:5]:
            print(f"    {p}")

    # Вызываем build_audio_chunks_in_range
    print("\nВызов build_audio_chunks_in_range...")
    audio_chunks = build_audio_chunks_in_range(audio_tracks_full, start_chunk, num_chunks)
    print(f"Результат: {len(audio_chunks)} чанков")
    empty_count = sum(1 for ac in audio_chunks if not ac)
    print(f"  Пустых чанков: {empty_count} из {len(audio_chunks)}")
    # Найдём первый непустой
    for i, ac in enumerate(audio_chunks):
        if ac:
            print(f"  Первый непустой чанк: индекс {i} (глобальный чанк {start_chunk + i})")
            for track_id in ac:
                print(f"    Дорожка {track_id}: {len(ac[track_id])} записей")
            break
    else:
        print("  *** Все чанки пусты! ***")

    print("\nГотово.")

if __name__ == "__main__":
    main()