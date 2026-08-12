#!/usr/bin/env python3
"""
audio_offset_check.py – проверка аудиосмещений, декодирования и запись в буфер.
Создаёт тестовые WAV-файлы для дорожек 2 и 3.
"""

import sys
import os
import time
import struct
import wave
from pathlib import Path
import numpy as np

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.timebase import SAMPLES_PER_CHUNK, FRAMES_PER_CHUNK
from index.idx_cache import prepare_mirror, get_mirror_path, open_idx_mmap
from index.lazy_index import LazyIndex
from index.moov_builder import (
    DTYPE_193, SEGMENT_SIZE,
    fast_video_records,
    build_audio_tracks,
    DEFAULT_TRACK_FILTER,
    build_audio_chunks_in_range,
    _abs_offset,
)
from file_io.win_sequential_reader import WinSequentialReader
from decode.audio_decoder import AudioDecoder, DEFAULT_ASC
from buffer.audio_buffer import MultiTrackAudioBuffer
from utils.utils import get_real_size

LIVE_SEEK_OFFSET_FRAMES = 1600

def write_wav(filename, samples, sample_rate=48000):
    samples = np.clip(samples, -1.0, 1.0)
    int_samples = (samples * 32767).astype(np.int16)
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_samples.tobytes())
    print(f"WAV сохранён: {filename} ({len(samples)} сэмплов, {len(samples)/sample_rate:.2f} с)")

def main():
    if len(sys.argv) < 2:
        print("Usage: python audio_offset_check.py <mp4_path>")
        sys.exit(1)

    mp4_path = Path(sys.argv[1])
    if not mp4_path.exists():
        print(f"Файл не найден: {mp4_path}")
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
    print(f"Зеркало: {mirror_path}")

    # Открываем полный индекс
    mm_193, mm_c9 = open_idx_mmap(mirror_path)
    total_frames = len(fast_video_records(mm_193))
    print(f"Всего видео кадров: {total_frames}")

    start_frame = max(0, total_frames - LIVE_SEEK_OFFSET_FRAMES)
    print(f"Стартовый кадр (live): {start_frame}")

    mdat_end = get_real_size(str(mp4_path))
    lazy_idx = LazyIndex(mirror_path, mp4_path, mdat_end)
    window = lazy_idx.open_window(start_frame)
    print(f"Окно: кадры {window.window_start_frame}-{window.window_end_frame-1}, чанков: {window.total_chunks}")

    # Глобальный номер чанка для стартового кадра
    global_start_chunk = start_frame // FRAMES_PER_CHUNK
    local_chunk = global_start_chunk - window.window_start_chunk
    print(f"Глобальный стартовый чанк: {global_start_chunk}, локальный в окне: {local_chunk}")

    # Диагностика: ищем аудиозаписи в полном C9 вокруг этого чанка
    start_pts = global_start_chunk * SAMPLES_PER_CHUNK
    end_pts = start_pts + SAMPLES_PER_CHUNK
    print(f"\nПоиск аудиозаписей в полном C9 для PTS [{start_pts}, {end_pts})")
    c9_pts = mm_c9['f3']
    mask_global = (c9_pts >= start_pts) & (c9_pts < end_pts)
    global_audio_count = np.sum(mask_global)
    print(f"Найдено {global_audio_count} записей")
    if global_audio_count > 0:
        print("Первые 5 записей:")
        for rec in mm_c9[mask_global][:5]:
            print(f"  pts={rec['f3']}, f1={rec['f1']}, f2={rec['f2']}, f7={rec['f7']}")
    else:
        # Расширим диапазон для анализа
        wide_start = start_pts - 5 * SAMPLES_PER_CHUNK
        wide_end = end_pts + 5 * SAMPLES_PER_CHUNK
        mask_wide = (c9_pts >= wide_start) & (c9_pts <= wide_end)
        print(f"В расширенном диапазоне [{wide_start}, {wide_end}) найдено {np.sum(mask_wide)} записей")
        if np.sum(mask_wide) > 0:
            print("Первые 5 в расширенном диапазоне:")
            for rec in mm_c9[mask_wide][:5]:
                print(f"  pts={rec['f3']}, f1={rec['f1']}, f2={rec['f2']}, f7={rec['f7']}")

    # Построение аудиочанков для окна (как это делает LazyIndex)
    audio_tracks_full = build_audio_tracks(mm_c9, track_filter=DEFAULT_TRACK_FILTER)
    print(f"\nПолный audio_tracks: {len(audio_tracks_full)} записей")
    audio_chunks_window = build_audio_chunks_in_range(
        audio_tracks_full, window.window_start_chunk, window.total_chunks
    )
    print(f"Аудиочанков в окне: {len(audio_chunks_window)} (локальный индекс {local_chunk} существует: {local_chunk < len(audio_chunks_window)})")
    if local_chunk < len(audio_chunks_window):
        chunk_data = audio_chunks_window[local_chunk]
        print(f"Содержимое audio_chunks[{local_chunk}]: {list(chunk_data.keys()) if chunk_data else 'пусто'}")
        for t in chunk_data:
            print(f"  Дорожка {t}: {len(chunk_data[t])} записей")
            for e in chunk_data[t][:3]:
                print(f"    offset={e['abs_offset']}, size1={e['size1']}, size2={e['size2']}, pts={e['pts']}")
    else:
        print(f"Локальный чанк {local_chunk} выходит за пределы списка аудиочанков (длина {len(audio_chunks_window)})")

    # Если аудиоданные для чанка есть, пробуем декодировать и записать WAV
    if local_chunk < len(audio_chunks_window) and audio_chunks_window[local_chunk]:
        print("\nДекодирование аудио...")
        reader = WinSequentialReader(mp4_path, 0, False)
        dec2 = AudioDecoder(DEFAULT_ASC)
        dec3 = AudioDecoder(DEFAULT_ASC)
        audio_buffers = MultiTrackAudioBuffer(capacity_samples=1440000)

        for track_id in (2, 3):
            entries = audio_chunks_window[local_chunk].get(track_id, [])
            for entry in entries:
                read_size = entry['size1'] + entry.get('size2', 0)
                if read_size == 0:
                    continue
                try:
                    data = reader.read_sequential(entry['abs_offset'], read_size)
                except Exception as e:
                    print(f"Ошибка чтения offset={entry['abs_offset']}: {e}")
                    continue
                decoder = dec2 if track_id == 2 else dec3
                pcm1 = decoder.decode(data[:entry['size1']]) if len(data) >= entry['size1'] else np.zeros(1024, dtype=np.float64)
                pcm2 = np.zeros(0, dtype=np.float64)
                if entry.get('size2', 0) > 0 and len(data) >= entry['size1'] + entry['size2']:
                    pcm2 = decoder.decode(data[entry['size1']:entry['size1']+entry['size2']])
                pcm_block = np.concatenate([pcm1, pcm2])
                success = audio_buffers.try_write(track_id, pcm_block, entry['pts'])
                status = "OK" if success else "FAIL"
                print(f"Трек {track_id}, pts={entry['pts']}: {len(pcm_block)} сэмплов, запись {status}")

        # Извлекаем из буфера и пишем WAV
        for track_id in (2, 3):
            buf = audio_buffers.buffers[track_id]
            avail = buf.available_read
            print(f"Буфер дорожки {track_id}: available_read={avail}")
            if avail > 0:
                # Перемещаем указатель чтения на начало накопленных данных
                buf.reset_read_to(buf.read_pos - avail)
                pcm = buf.read(avail, timeout=1.0)
                if len(pcm) > 0:
                    write_wav(f"track{track_id}_chunk{local_chunk}.wav", pcm)

        reader.close()
        dec2.close()
        dec3.close()
    else:
        print("\nНет аудиоданных для декодирования в выбранном чанке.")

    print("Готово.")

if __name__ == "__main__":
    main()