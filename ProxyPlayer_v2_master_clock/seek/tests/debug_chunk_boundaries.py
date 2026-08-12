#!/usr/bin/env python3
"""
Анализ границ чанков: сравнение смещений на стыках и выявление аномалий.
Запуск: python tests/debug_chunk_boundaries.py <путь_к_mp4> [окно_сек]
"""

import sys
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)-8s | %(message)s')
logging.getLogger('av').setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from index.lazy_index import LazyIndex
from index.idx_cache import prepare_mirror
from utils.utils import get_real_size
from config.timebase import FRAMES_PER_CHUNK


def analyze_boundaries(mp4_path: Path, window_seconds: float = 300.0, num_boundaries: int = 10):
    stem = mp4_path.stem
    parent = mp4_path.parent
    idx_path = parent / "idx" / "mp4" / f"{stem}.idx"
    if not idx_path.exists():
        idx_path = parent / f"{stem}.idx"
    if not idx_path.exists():
        print("IDX не найден")
        return

    mirror = prepare_mirror(idx_path)
    mdat_end = get_real_size(str(mp4_path))
    lazy = LazyIndex(mirror, mp4_path, mdat_end)
    window = lazy.open_window(center_frame=0, window_seconds=window_seconds)

    total_chunks = window.total_chunks
    n = min(num_boundaries, total_chunks)
    
    # Собираем данные
    chunk_offs = window.chunk_offsets
    chunk_sizes = window.chunk_sizes
    cached = window.cached_offsets
    video_recs = window.video_records
    
    print(f"\n=== Анализ {n} границ чанков ===")
    print(f"Окно: кадры {window.window_start_frame}-{window.window_end_frame}, чанки 0-{total_chunks-1}")
    print(f"{'Граница':<8} {'Пред.чанк':<10} {'Тек.чанк':<10} {'Смещ.конца пред.':<18} {'Смещ.начала тек.':<18} {'Разница':<12} {'Ожид.размер':<14} {'Статус'}")
    print("-" * 110)
    
    for i in range(1, n):
        prev_chunk = i - 1
        curr_chunk = i
        
        # Смещение начала текущего чанка
        curr_start = chunk_offs[curr_chunk]
        
        # Смещение КОНЦА предыдущего чанка = его начало + размер
        prev_start = chunk_offs[prev_chunk]
        prev_size = chunk_sizes[prev_chunk]
        prev_end = prev_start + prev_size
        
        # Последний кадр предыдущего чанка
        last_frame_prev = (window.window_start_chunk + prev_chunk + 1) * FRAMES_PER_CHUNK - 1
        local_last = last_frame_prev - window.window_start_frame
        if 0 <= local_last < len(cached):
            last_frame_offset = cached[local_last]
            # Размер последнего кадра можно оценить как разницу до конца чанка
            last_frame_size = prev_end - last_frame_offset
        else:
            last_frame_offset = -1
            last_frame_size = -1
        
        # Первый кадр текущего чанка
        first_frame_curr = (window.window_start_chunk + curr_chunk) * FRAMES_PER_CHUNK
        local_first = first_frame_curr - window.window_start_frame
        if 0 <= local_first < len(cached):
            first_frame_offset = cached[local_first]
        else:
            first_frame_offset = -1
        
        # Разница: начало текущего чанка минус конец предыдущего (должна быть 0 для правильной нарезки)
        diff = curr_start - prev_end
        
        # Ожидаемый размер последнего кадра предыдущего чанка (из cached)
        # Если размер > 1000000, это вероятно маркер, а не реальный кадр
        status = "OK" if diff == 0 else f"разрыв {diff}" if diff > 0 else f"наложение {-diff}"
        if last_frame_size > 1_000_000 or last_frame_size < 0:
            status += " (подозрительный размер)"
        
        print(f"{i:<8} {prev_chunk:<10} {curr_chunk:<10} {prev_end:<18} {curr_start:<18} {diff:<12} {last_frame_size:<14} {status}")
    
    # Сравнение с концом файла для последнего чанка
    last_chunk = n - 1
    last_start = chunk_offs[last_chunk]
    last_size = chunk_sizes[last_chunk]
    last_end = last_start + last_size
    print(f"\nПоследний анализируемый чанк {last_chunk}: начало={last_start}, размер={last_size}, конец={last_end}")
    print(f"mdat_end={mdat_end}")
    if last_end > mdat_end:
        print("ВНИМАНИЕ: чанк выходит за границу файла!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python tests/debug_chunk_boundaries.py <путь_к_mp4> [окно_сек]")
        sys.exit(1)
    mp4 = Path(sys.argv[1])
    win_sec = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
    analyze_boundaries(mp4, win_sec)