#!/usr/bin/env python3
"""
Скрипт для детальной отладки демультиплексирования одного чанка.
Показывает все промежуточные значения и определяет причину ошибок.
Запуск: python tests/debug_demux.py <путь_к_mp4> <локальный_чанк> [окно_сек]
"""

import sys
import logging
from pathlib import Path
import numpy as np

# Настройка логирования
logging.basicConfig(level=logging.DEBUG, format='%(levelname)-8s | %(message)s')
logging.getLogger('av').setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from index.lazy_index import LazyIndex
from index.idx_cache import prepare_mirror
from utils.utils import get_real_size
from file_io.win_sequential_reader import WinSequentialReader
from config.timebase import FRAMES_PER_CHUNK, SAMPLES_PER_VIDEO_FRAME


def debug_demux_chunk(mp4_path: Path, local_chunk: int, window_seconds: float = 300.0):
    # Находим .idx и .ref
    stem = mp4_path.stem
    parent = mp4_path.parent
    idx_path = parent / "idx" / "mp4" / f"{stem}.idx"
    if not idx_path.exists():
        idx_path = parent / f"{stem}.idx"
    if not idx_path.exists():
        print("IDX не найден")
        return

    # Инициализируем LazyIndex
    mirror = prepare_mirror(idx_path)
    mdat_end = get_real_size(str(mp4_path))
    lazy = LazyIndex(mirror, mp4_path, mdat_end)
    window = lazy.open_window(center_frame=0, window_seconds=window_seconds)

    print(f"\n=== Информация об окне ===")
    print(f"Кадры: {window.window_start_frame}–{window.window_end_frame}")
    print(f"Чанки: {window.window_start_chunk}–{window.window_start_chunk + window.total_chunks - 1}")
    print(f"Всего чанков в окне: {window.total_chunks}")
    print(f"Количество видео-записей в окне: {len(window.video_records)}")
    print(f"Размер cached_offsets: {len(window.cached_offsets) if window.cached_offsets is not None else 'None'}")

    if local_chunk >= window.total_chunks:
        print(f"ОШИБКА: локальный чанк {local_chunk} выходит за пределы окна (0..{window.total_chunks - 1})")
        return

    # Параметры выбранного чанка
    print(f"\n=== Чанк {local_chunk} (глобальный {window.window_start_chunk + local_chunk}) ===")
    chunk_start_offset = int(window.chunk_offsets[local_chunk])
    chunk_size = int(window.chunk_sizes[local_chunk])
    chunk_end_offset = chunk_start_offset + chunk_size
    print(f"Смещение в файле: {chunk_start_offset}–{chunk_end_offset} (размер {chunk_size})")

    # Глобальный номер первого кадра
    global_start_frame = (window.window_start_chunk + local_chunk) * FRAMES_PER_CHUNK
    print(f"Глобальный первый кадр: {global_start_frame}")

    # Читаем сырые данные
    reader = WinSequentialReader(mp4_path, rate_limit=0, overlapped=False)
    raw_data = reader.read_sequential(chunk_start_offset, chunk_size)
    reader.close()
    print(f"Прочитано сырых данных: {len(raw_data)} байт")

    # Эмулируем демукс
    video_packets = 0
    audio_packets = 0
    errors = 0
    cached_offs = window.cached_offsets

    for i in range(FRAMES_PER_CHUNK):
        abs_idx = global_start_frame + i
        local_frame = abs_idx - window.window_start_frame
        print(f"\n  Кадр {i}: abs_idx={abs_idx}, local_frame={local_frame}")

        if local_frame < 0 or local_frame >= len(window.video_records):
            print(f"    -> ПРОПУСК: local_frame вне диапазона")
            continue

        rec = window.video_records[local_frame]

        # Абсолютное смещение
        if cached_offs is not None and local_frame < len(cached_offs):
            abs_off = int(cached_offs[local_frame])
            off_source = "cached"
        else:
            abs_off = int(rec['f1']) - 4 + int(rec['f2']) * 4_294_967_296
            off_source = "calculated"
        print(f"    Абс. смещение: {abs_off} ({off_source})")

        # Размер кадра
        if i < FRAMES_PER_CHUNK - 1:
            next_local = local_frame + 1
            if next_local < len(window.video_records):
                if cached_offs is not None and next_local < len(cached_offs):
                    next_off = int(cached_offs[next_local])
                else:
                    next_rec = window.video_records[next_local]
                    next_off = int(next_rec['f1']) - 4 + int(next_rec['f2']) * 4_294_967_296
                size = next_off - abs_off
                size_source = "следующий кадр"
            else:
                size = 0
                size_source = "нет следующего"
        else:
            size = chunk_end_offset - abs_off
            size_source = "конец чанка"
        print(f"    Размер: {size} ({size_source})")

        if size <= 0:
            print(f"    -> ОШИБКА: неположительный размер")
            errors += 1
            continue

        rel_start = abs_off - chunk_start_offset
        if rel_start < 0 or rel_start + size > len(raw_data):
            print(f"    -> ОШИБКА: выход за границы raw_data (rel={rel_start}, size={size}, len={len(raw_data)})")
            errors += 1
            continue

        print(f"    -> OK: создан видео-пакет (PTS={abs_idx * SAMPLES_PER_VIDEO_FRAME})")
        video_packets += 1

    # Аудио
    audio_chunk = window.audio_chunks[local_chunk] if local_chunk < len(window.audio_chunks) else {}
    print(f"\n  Аудио-чанок: {'есть записи' if audio_chunk else 'пуст'}")

    print(f"\n=== Итог ===")
    print(f"Видео-пакеты: {video_packets}/12")
    print(f"Аудио-пакеты: {audio_packets}")
    print(f"Ошибки: {errors}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Использование: python tests/debug_demux.py <путь_к_mp4> <локальный_чанк> [окно_сек]")
        sys.exit(1)
    mp4 = Path(sys.argv[1])
    chunk = int(sys.argv[2])
    win_sec = float(sys.argv[3]) if len(sys.argv) > 3 else 300.0
    debug_demux_chunk(mp4, chunk, win_sec)