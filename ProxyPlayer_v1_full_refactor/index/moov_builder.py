"""
moov_builder.py – финальная версия для ProxyPlayer v1.
Содержит полный набор функций v5 + инкрементальные обновления +
быстрая фильтрация fast_video_records и адаптивное построение IDR-карты.
Добавлен предвычисленный индекс аудиочанков для ускорения построения.
"""

import logging
import numpy as np
from typing import List, Dict
from collections import defaultdict

logger = logging.getLogger(__name__)

SEGMENT_SIZE = 4_294_967_296
FRAMES_PER_CHUNK = 12
AUDIO_SAMPLERATE = 48000
VIDEO_FPS = 25
SAMPLES_PER_VIDEO_FRAME = AUDIO_SAMPLERATE // VIDEO_FPS          # 1920
SAMPLES_PER_CHUNK = SAMPLES_PER_VIDEO_FRAME * FRAMES_PER_CHUNK   # 23040

DTYPE_193 = np.dtype([
    ('f0', '<u4'), ('f1', '<u4'), ('f2', '<u4'), ('f3', '<u4'),
    ('f4', '<u4'), ('f5', '<u4'), ('f6', '<u4'), ('f7', '<u4'),
    ('f8', '<u4'), ('f9', '<u4'), ('f10','<u4'), ('f11','<u4'),
    ('f12','<u4'), ('f13','<u4'), ('f14','<u4'), ('f15','<u4'),
    ('f16','<u4')
])

DTYPE_C9 = DTYPE_193

DTYPE_AUDIO = np.dtype([
    ('abs_offset', '<u8'),
    ('size1', '<u4'),
    ('size2', '<u4'),
    ('pts', '<u8'),
    ('track', '<u4'),
])

DEFAULT_TRACK_FILTER = [2, 3]
AUDIO_CHUNK_TAIL_UPDATE = 200   # сколько последних существующих чанков пересчитывать

# ----------------------------------------------------------------------
# Фильтрация и построение цепочек (v5)
# ----------------------------------------------------------------------
def _filter_normal_records(records): return records[records['f2'] < 256]
def _deduplicate_by_f3(records):
    _, u = np.unique(records['f3'], return_index=True)
    return records[np.sort(u)]
def _build_chains(records, min_len=12):
    if len(records) == 0: return []
    f3 = records['f3']
    breaks = np.where(np.diff(f3) != 1)[0] + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [len(records)]))
    chains = [records[s:e] for s, e in zip(starts, ends) if e-s >= min_len]
    chains.sort(key=lambda c: len(c), reverse=True)
    return chains
def _merge_chains(chains):
    if not chains: return np.empty(0, dtype=DTYPE_193)
    chains.sort(key=lambda c: c[0]['f3'])
    return np.concatenate(chains)
def _abs_offset(rec):
    return (rec['f1'].astype(np.uint64) + rec['f2'].astype(np.uint64) * np.uint64(SEGMENT_SIZE)
            - np.uint64(4))

def build_chain_simple(all_193):
    if len(all_193) == 0: return np.empty(0, dtype=DTYPE_193)
    normal = _filter_normal_records(all_193)
    if len(normal) == 0: return np.empty(0, dtype=DTYPE_193)
    uniq = _deduplicate_by_f3(normal)
    chains = _build_chains(uniq, 12)
    return _merge_chains(chains) if chains else np.empty(0, dtype=DTYPE_193)

def incremental_append_video(existing_video: np.ndarray, new_193: np.ndarray) -> np.ndarray:
    if len(new_193) == 0:
        return existing_video
    normal_new = _filter_normal_records(new_193)
    if len(normal_new) == 0:
        return existing_video
    existing_f3 = set(existing_video['f3']) if len(existing_video) > 0 else set()
    mask_new = ~np.isin(normal_new['f3'], list(existing_f3))
    unique_new = normal_new[mask_new]
    if len(unique_new) == 0:
        return existing_video
    sorted_idx = np.argsort(unique_new['f3'])
    unique_new = unique_new[sorted_idx]
    if len(existing_video) == 0:
        return unique_new
    last_f3 = existing_video[-1]['f3']
    if unique_new[0]['f3'] == last_f3 + 1:
        return np.concatenate([existing_video, unique_new])
    return np.concatenate([existing_video, unique_new])

# ----------------------------------------------------------------------
# Чанки
# ----------------------------------------------------------------------
def get_chunks(all_193, chunk_ends_raw, mdat_end):
    chain = build_chain_simple(all_193)
    if len(chain) == 0: return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    offs = _abs_offset(chain)
    n = len(chain)
    nchunks = (n + 11) // 12
    start_idx = np.arange(0, nchunks*12, 12, dtype=np.int64)
    start_idx = np.clip(start_idx, 0, n-1)
    chunk_offs = offs[start_idx]
    next_idx = start_idx + 12
    next_idx[-1] = n
    next_offs = np.empty(nchunks, dtype=np.uint64)
    mask = next_idx < n
    next_offs[mask] = offs[next_idx[mask]]
    next_offs[~mask] = np.uint64(mdat_end)
    chunk_sizes = np.maximum(0, next_offs.astype(np.int64) - chunk_offs.astype(np.int64))
    return chunk_offs.astype(np.int64), chunk_sizes.astype(np.int64)

def incremental_get_chunks(video_records, mdat_end):
    if len(video_records) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    offs = _abs_offset(video_records)
    n = len(video_records)
    nchunks = (n + 11) // 12
    start_idx = np.arange(0, nchunks*12, 12, dtype=np.int64)
    start_idx = np.clip(start_idx, 0, n-1)
    chunk_offs = offs[start_idx]
    next_idx = start_idx + 12
    next_idx[-1] = n
    next_offs = np.empty(nchunks, dtype=np.uint64)
    mask = next_idx < n
    next_offs[mask] = offs[next_idx[mask]]
    next_offs[~mask] = np.uint64(mdat_end)
    chunk_sizes = np.maximum(0, next_offs.astype(np.int64) - chunk_offs.astype(np.int64))
    return chunk_offs.astype(np.int64), chunk_sizes.astype(np.int64)

def build_chunks_in_range(video_records, mdat_end, start_chunk=0, num_chunks=1200):
    """
    Построение чанков только для указанного диапазона [start_chunk, start_chunk+num_chunks).
    Возвращает кортеж (chunk_offsets, chunk_sizes) для этого диапазона.
    Используется для быстрого старта.
    """
    if len(video_records) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    offs = _abs_offset(video_records)
    n = len(video_records)
    total_chunks = (n + 11) // 12

    start_chunk = max(0, min(start_chunk, total_chunks - 1))
    end_chunk = min(start_chunk + num_chunks, total_chunks)

    chunk_offs_list = []
    chunk_sizes_list = []

    for chunk_idx in range(start_chunk, end_chunk):
        start_frame = chunk_idx * 12
        if start_frame >= n:
            break
        off = offs[start_frame]
        next_frame = min(start_frame + 12, n)
        if next_frame < n:
            next_off = offs[next_frame]
        else:
            next_off = np.uint64(mdat_end)
        size = max(0, int(next_off) - int(off))
        chunk_offs_list.append(off)
        chunk_sizes_list.append(size)

    return np.array(chunk_offs_list, dtype=np.int64), np.array(chunk_sizes_list, dtype=np.int64)

def build_chunks_from_cached_offsets(video_slice, cached_offsets, mdat_end):
    """Версия build_chunks_in_range с предвычисленными смещениями."""
    if len(video_slice) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    
    offs = cached_offsets
    n = len(video_slice)
    nchunks = (n + FRAMES_PER_CHUNK - 1) // FRAMES_PER_CHUNK
    chunk_offs = np.empty(nchunks, dtype=np.int64)
    chunk_sizes = np.empty(nchunks, dtype=np.int64)
    
    for i in range(nchunks):
        start_frame = i * FRAMES_PER_CHUNK
        chunk_offs[i] = offs[start_frame]
        next_frame = min(start_frame + FRAMES_PER_CHUNK, n)
        if next_frame < n:
            next_off = offs[next_frame]
        else:
            next_off = np.uint64(mdat_end)
        chunk_sizes[i] = max(0, int(next_off) - int(chunk_offs[i]))
    
    return chunk_offs, chunk_sizes

# ----------------------------------------------------------------------
# Аудио (исправленный расчёт size2, векторизован)
# ----------------------------------------------------------------------
def build_audio_tracks(c9_records, track_filter: List[int] = None):
    if track_filter is None:
        track_filter = DEFAULT_TRACK_FILTER
    if len(c9_records) == 0:
        return np.empty(0, dtype=DTYPE_AUDIO)

    track_nums = c9_records['f8'] & 0xFF
    mask = np.isin(track_nums, track_filter)
    valid = c9_records[mask]
    track_nums = track_nums[mask]
    if len(valid) == 0:
        return np.empty(0, dtype=DTYPE_AUDIO)

    abs_offs = (valid['f2'].astype(np.uint64) * np.uint64(SEGMENT_SIZE) +
                valid['f1'].astype(np.uint64))
    sort_idx = np.argsort(abs_offs)
    valid = valid[sort_idx]
    track_nums = track_nums[sort_idx]
    abs_offs = abs_offs[sort_idx]

    n_entries = len(valid)
    audio_arr = np.zeros(n_entries, dtype=DTYPE_AUDIO)
    audio_arr['abs_offset'] = abs_offs
    audio_arr['size1'] = valid['f7']
    audio_arr['pts'] = valid['f3']
    audio_arr['track'] = track_nums

    # Векторизованный расчёт size2 для всех дорожек
    for t in track_filter:
        idx = np.where(audio_arr['track'] == t)[0]
        if len(idx) > 1:
            next_offs = audio_arr['abs_offset'][idx[1:]]
            current_ends = audio_arr['abs_offset'][idx[:-1]] + audio_arr['size1'][idx[:-1]]
            audio_arr['size2'][idx[:-1]] = next_offs - current_ends

    audio_arr['size2'] = np.clip(audio_arr['size2'], 0, None)
    return audio_arr

def incremental_append_audio_tracks(
    existing_audio: np.ndarray,
    new_c9: np.ndarray,
    track_filter: List[int] = None
) -> np.ndarray:
    if track_filter is None:
        track_filter = DEFAULT_TRACK_FILTER
    if len(new_c9) == 0:
        return existing_audio

    new_audio = build_audio_tracks(new_c9, track_filter)
    if len(new_audio) == 0:
        return existing_audio
    if len(existing_audio) == 0:
        return new_audio

    merged = np.concatenate([existing_audio, new_audio])
    merged.sort(order='abs_offset')

    for t in track_filter:
        idx = np.where(merged['track'] == t)[0]
        if len(idx) > 1:
            next_offs = merged['abs_offset'][idx[1:]]
            current_ends = merged['abs_offset'][idx[:-1]] + merged['size1'][idx[:-1]]
            merged['size2'][idx[:-1]] = next_offs - current_ends
        if len(idx) > 0:
            merged['size2'][idx[-1]] = 0

    merged['size2'] = np.clip(merged['size2'], 0, None)
    return merged

def build_audio_chunks_from_structured(audio_arr: np.ndarray, num_video_chunks: int):
    if len(audio_arr) == 0 or num_video_chunks == 0:
        return [{} for _ in range(num_video_chunks)]

    # ИСПРАВЛЕНИЕ: приведение к int64 для безопасного вычитания
    chunk_indices = (audio_arr['pts'] // SAMPLES_PER_CHUNK).astype(np.int64)
    chunks = [{} for _ in range(num_video_chunks)]

    for i in range(len(audio_arr)):
        chunk_idx = chunk_indices[i]
        if 0 <= chunk_idx < num_video_chunks:
            entry = {
                'abs_offset': int(audio_arr[i]['abs_offset']),
                'size1': int(audio_arr[i]['size1']),
                'size2': int(audio_arr[i]['size2']),
                'pts': int(audio_arr[i]['pts']),
                'track': int(audio_arr[i]['track']),
            }
            track = entry['track']
            if track not in chunks[chunk_idx]:
                chunks[chunk_idx][track] = []
            chunks[chunk_idx][track].append(entry)

    return chunks

def build_audio_chunks_v2(audio_tracks, num_video_chunks):
    return build_audio_chunks_from_structured(audio_tracks, num_video_chunks)

def build_audio_chunks_in_range(audio_arr: np.ndarray, start_chunk: int, num_chunks: int):
    """
    Построение аудиочанков только для указанного диапазона чанков.
    Возвращает список словарей, по длине равный num_chunks.
    Использует предвычисленный индекс.
    """
    if len(audio_arr) == 0 or num_chunks == 0:
        return [{} for _ in range(num_chunks)]

    # ИСПРАВЛЕНИЕ: приведение к int64, чтобы избежать переполнения uint64 при вычитании
    chunk_indices = (audio_arr['pts'] // SAMPLES_PER_CHUNK).astype(np.int64)
    chunks = [{} for _ in range(num_chunks)]

    for i in range(len(audio_arr)):
        global_chunk = chunk_indices[i]
        local_chunk = global_chunk - start_chunk
        if 0 <= local_chunk < num_chunks:
            entry = {
                'abs_offset': int(audio_arr[i]['abs_offset']),
                'size1': int(audio_arr[i]['size1']),
                'size2': int(audio_arr[i]['size2']),
                'pts': int(audio_arr[i]['pts']),
                'track': int(audio_arr[i]['track']),
            }
            track = entry['track']
            if track not in chunks[local_chunk]:
                chunks[local_chunk][track] = []
            chunks[local_chunk][track].append(entry)

    return chunks

def incremental_build_audio_chunks_v2(existing_chunks, audio_arr, num_video_chunks):
    """
    Инкрементально обновляет список аудиочанков.
    Обновляет последние AUDIO_CHUNK_TAIL_UPDATE существующих чанков
    и достраивает все новые до num_video_chunks-1.
    """
    if num_video_chunks <= len(existing_chunks):
        start_update = max(0, len(existing_chunks) - AUDIO_CHUNK_TAIL_UPDATE)
    else:
        start_update = max(0, len(existing_chunks) - AUDIO_CHUNK_TAIL_UPDATE)

    # ИСПРАВЛЕНИЕ: приведение к int64
    chunk_indices = (audio_arr['pts'] // SAMPLES_PER_CHUNK).astype(np.int64)
    chunks = existing_chunks[:start_update]

    for chunk_idx in range(start_update, num_video_chunks):
        mask = chunk_indices == chunk_idx
        chunk_entries = defaultdict(list)
        for i in np.where(mask)[0]:
            entry = {
                'abs_offset': int(audio_arr[i]['abs_offset']),
                'size1': int(audio_arr[i]['size1']),
                'size2': int(audio_arr[i]['size2']),
                'pts': int(audio_arr[i]['pts']),
                'track': int(audio_arr[i]['track']),
            }
            chunk_entries[entry['track']].append(entry)
        chunks.append(dict(chunk_entries))
    return chunks

def rebuild_audio_chunks_from(audio_arr, existing_chunks, start_chunk_idx, num_video_chunks):
    """
    Перестраивает аудиочанки начиная с start_chunk_idx до num_video_chunks-1.
    Чанки до start_chunk_idx остаются без изменений.
    """
    if start_chunk_idx >= num_video_chunks:
        return existing_chunks

    # ИСПРАВЛЕНИЕ: приведение к int64
    chunk_indices = (audio_arr['pts'] // SAMPLES_PER_CHUNK).astype(np.int64)
    chunks = existing_chunks[:start_chunk_idx]
    for chunk_idx in range(start_chunk_idx, num_video_chunks):
        mask = chunk_indices == chunk_idx
        chunk_entries = defaultdict(list)
        for i in np.where(mask)[0]:
            entry = {
                'abs_offset': int(audio_arr[i]['abs_offset']),
                'size1': int(audio_arr[i]['size1']),
                'size2': int(audio_arr[i]['size2']),
                'pts': int(audio_arr[i]['pts']),
                'track': int(audio_arr[i]['track']),
            }
            chunk_entries[entry['track']].append(entry)
        chunks.append(dict(chunk_entries))
    return chunks

def get_first_chunk(video_records, audio_chunks):
    v = video_records[:12].copy() if len(video_records) >= 12 else video_records.copy()
    a = audio_chunks[0] if audio_chunks else {}
    return v, a

# ----------------------------------------------------------------------
# IDR (старый метод, оставлен для совместимости)
# ----------------------------------------------------------------------
def get_idr_indices_from_mmap(mm_193: np.ndarray) -> np.ndarray:
    """Возвращает индексы IDR-кадров (f7 = 27, 28, 29, 30)."""
    idr_mask = (mm_193['f7'] == 27) | (mm_193['f7'] == 28) | \
               (mm_193['f7'] == 29) | (mm_193['f7'] == 30)
    return np.where(idr_mask)[0].astype(np.int64)

# ----------------------------------------------------------------------
# НОВЫЕ ФУНКЦИИ (быстрая фильтрация + адаптивная IDR-карта)
# ----------------------------------------------------------------------
def fast_video_records(mm_193: np.ndarray) -> np.ndarray:
    """
    Быстрая фильтрация видео‑записей для production‑окружения.
    Только f2 < 256, сортировка по f3, удаление дубликатов.
    Возвращает DTYPE_193.
    """
    if len(mm_193) == 0:
        return np.empty(0, dtype=DTYPE_193)

    mask = mm_193['f2'] < 256
    valid = mm_193[mask]

    if len(valid) == 0:
        return np.empty(0, dtype=DTYPE_193)

    sorted_idx = np.argsort(valid['f3'])
    valid = valid[sorted_idx]

    _, unique_idx = np.unique(valid['f3'], return_index=True)
    valid = valid[unique_idx]

    return valid


def build_idr_map(
    video_records: np.ndarray,
    reader=None,
    check_all: bool = False,
    initial_check: int = 300,
    step_tolerance: int = 2
) -> np.ndarray:
    """
    Построение карты IDR‑кадров с проверкой NAL type 5.

    Если reader передан, проверяет до initial_check первых кандидатов,
    определяет шаг и экстраполирует остальные. Если reader=None,
    используется старый метод (только по f7).

    Возвращает индексы в video_records.
    """
    if reader is None:
        return get_idr_indices_from_mmap(video_records)

    idr_mask = (video_records['f7'] >= 27) & (video_records['f7'] <= 30)
    candidates = np.where(idr_mask)[0]

    if len(candidates) == 0:
        return candidates

    check_count = len(candidates) if check_all else min(len(candidates), initial_check)

    confirmed = []
    for i, idx in enumerate(candidates[:check_count]):
        rec = video_records[idx]
        abs_off = int(rec['f1']) - 4 + int(rec['f2']) * SEGMENT_SIZE
        if idx + 1 < len(video_records):
            next_rec = video_records[idx + 1]
            next_off = int(next_rec['f1']) - 4 + int(next_rec['f2']) * SEGMENT_SIZE
            frame_size = next_off - abs_off
        else:
            frame_size = 1_000_000
        if frame_size <= 0:
            frame_size = 1_000_000
        if _has_nal_type5_in_frame(reader, abs_off, frame_size):
            confirmed.append(idx)

    if len(confirmed) == 0:
        logger.warning("Не найдено ни одного IDR через NAL type 5 – перемотка будет недоступна")
        return np.array([], dtype=np.int64)

    if len(confirmed) < 2:
        logger.info(f"Подтверждён только один IDR, экстраполяция не выполняется")
        return np.array(confirmed, dtype=np.int64)

    step = int(np.median(np.diff(confirmed)))
    step = max(step, 1)

    last_confirmed = confirmed[-1]
    extrapolated = []
    for idx in candidates[check_count:]:
        dist = idx - last_confirmed
        if dist % step <= step_tolerance or step - (dist % step) <= step_tolerance:
            extrapolated.append(idx)

    all_idr = np.concatenate([confirmed, extrapolated])
    all_idr = np.unique(all_idr)
    all_idr.sort()
    return all_idr.astype(np.int64)


def _has_nal_type5_in_frame(reader, abs_offset: int, frame_size: int) -> bool:
    """
    Проверяет наличие NAL unit type 5 в кадре размером frame_size.
    Читает данные до конца кадра (но не более 2 МБ для защиты).
    """
    read_size = min(frame_size, 2_000_000)
    if read_size <= 0:
        return False
    try:
        data = reader.read_sequential(int(abs_offset), read_size)
    except Exception:
        return False
    if len(data) < 5:
        return False
    pos = 0
    while pos + 4 <= len(data):
        nal_size = int.from_bytes(data[pos:pos+4], 'big')
        if nal_size == 0 or pos + 4 + nal_size > len(data):
            break
        if (data[pos+4] & 0x1F) == 5:
            return True
        pos += 4 + nal_size
    return False