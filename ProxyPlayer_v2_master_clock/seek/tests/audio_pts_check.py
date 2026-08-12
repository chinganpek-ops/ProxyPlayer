#!/usr/bin/env python3
"""
audio_pts_check.py – диагностика PTS аудиопакетов и позиции буфера.
Запускает StreamController, инициализирует окно и выводит ожидаемые PTS
аудиоданных для live‑чанка, сравнивая их с текущим read_pos буфера.
"""

import sys
import time
from pathlib import Path
import numpy as np

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.timebase import SAMPLES_PER_CHUNK, FRAMES_PER_CHUNK, video_frame_to_pts
from index.idx_cache import prepare_mirror, get_mirror_path, open_idx_mmap
from index.lazy_index import LazyIndex
from index.moov_builder import build_audio_chunks_in_range, build_audio_tracks, DEFAULT_TRACK_FILTER
from utils.utils import get_real_size

def main():
    if len(sys.argv) < 2:
        print("Usage: python audio_pts_check.py <mp4_path>")
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
    from index.moov_builder import fast_video_records
    total_frames = len(fast_video_records(mm_193))
    print(f"Всего видео кадров: {total_frames}")

    # Live‑старт
    LIVE_SEEK_OFFSET_FRAMES = 1600
    start_frame = max(0, total_frames - LIVE_SEEK_OFFSET_FRAMES)
    print(f"Стартовый кадр: {start_frame}")

    mdat_end = get_real_size(str(mp4_path))
    lazy_idx = LazyIndex(mirror_path, mp4_path, mdat_end)
    window = lazy_idx.open_window(start_frame)
    print(f"Окно: кадры {window.window_start_frame}-{window.window_end_frame-1}, чанков: {window.total_chunks}")

    local_chunk = (start_frame - window.window_start_frame) // FRAMES_PER_CHUNK
    print(f"Локальный чанк: {local_chunk}")

    # Строим аудиочанки для окна (как в LazyIndex)
    audio_tracks = build_audio_tracks(mm_c9, track_filter=DEFAULT_TRACK_FILTER)
    audio_chunks = build_audio_chunks_in_range(
        audio_tracks, window.window_start_chunk, window.total_chunks
    )

    # Извлекаем аудиоданные для нужного чанка
    if local_chunk < len(audio_chunks) and audio_chunks[local_chunk]:
        chunk_audio = audio_chunks[local_chunk]
        print(f"\nАудиоданные для чанка {local_chunk}:")
        for track_id in (2, 3):
            entries = chunk_audio.get(track_id, [])
            print(f"  Дорожка {track_id}: {len(entries)} записей")
            for i, entry in enumerate(entries[:5]):
                print(f"    [{i}] pts={entry['pts']} (глобальный PTS)")
    else:
        print(f"\nАудиоданные для чанка {local_chunk} отсутствуют!")
        # Если аудиоданных нет, проверим соседние чанки
        for offset in [-1, 1]:
            idx = local_chunk + offset
            if 0 <= idx < len(audio_chunks) and audio_chunks[idx]:
                print(f"  В соседнем чанке {idx} аудио есть: {len(audio_chunks[idx])} дорожек")
                for track_id in (2, 3):
                    entries = audio_chunks[idx].get(track_id, [])
                    if entries:
                        print(f"    Пример pts для дорожки {track_id}: {entries[0]['pts']}")

    # Теперь получим read_pos аудиобуферов после resume
    # Для этого создадим StreamController (он уже настроен на live) и вызовем resume
    from core.stream_controller import StreamController
    ref = parent / f"{stem}.mp4.ref"
    if not ref.exists():
        ref = parent / f"{stem}.ref"

    print("\nЗапуск StreamController для получения состояния буферов после resume...")
    controller = StreamController(
        ref_path=ref, idx_path=idx, mp4_path=mp4_path,
        fps=25.0, buffer_size=800, start_from_live=True,
        mirror_path=str(mirror_path)
    )
    if not controller._ready.wait(timeout=60):
        print("Ошибка инициализации")
        sys.exit(1)

    # Запускаем воспроизведение и сразу resume
    controller.start_playback()
    time.sleep(0.3)
    controller.resume()
    time.sleep(0.3)

    print("Состояние аудиобуферов после resume:")
    for track_id in (2, 3):
        buf = controller.audio_buffers.buffers[track_id]
        print(f"  Дорожка {track_id}: read_pos={buf.read_pos}, write_pos={buf.write_pos}, available={buf.available_read}")

    # Если аудиоданные для чанка есть, сравним их PTS с read_pos
    if local_chunk < len(audio_chunks) and audio_chunks[local_chunk]:
        first_pts = None
        for track_id in (2, 3):
            entries = audio_chunks[local_chunk].get(track_id, [])
            if entries:
                first_pts = entries[0]['pts']
                break
        if first_pts is not None:
            read_pos = controller.audio_buffers.buffers[2].read_pos
            print(f"\nСравнение: первый PTS аудиоданных = {first_pts}, read_pos буфера = {read_pos}")
            if first_pts > read_pos + 48000:  # расхождение больше 1 секунды
                print("!!! Обнаружено значительное расхождение – аудиоданные не будут записаны в буфер.")
                print("Причина: PTS аудио (из индекса) глобальный и опережает позицию буфера, сброшенную на audio_clock live-позиции.")
    else:
        print("\nНевозможно сравнить PTS – аудиоданных в live‑чанке нет.")

    controller.stop()
    controller.close()
    print("Готово.")

if __name__ == "__main__":
    main()