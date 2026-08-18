"""
moov_builder.py – построение видео/аудио индекса из mmap .idx-записей.
Используется LazyIndex (index/lazy_index.py) для построения окна вокруг
текущей позиции воспроизведения, и idx_cache.py (типы записей).

ИЗМЕНЕНИЯ (правки продакшен-ревью):
- FRAMES_PER_CHUNK / SAMPLES_PER_VIDEO_FRAME / SAMPLES_PER_CHUNK раньше
  определялись здесь ЛОКАЛЬНО, отдельной копией от тех же констант в
  config/timebase.py. Числовые значения совпадали (12 / 1920 / 23040), но
  дублирование рисковало разойтись, если бы кто-то поменял константу
  только в одном месте. Теперь единственный источник правды —
  config/timebase.py, здесь только import.

- УДАЛЁН мёртвый код (после сверки с index_builder.py, который заменяется
  на index_service.py): index_builder.py оказался предельно простым —
  он только вызывал idx_cache.prepare_mirror() по таймеру и не использовал
  ни одну из функций построения цепочек/чанков этого файла. Значит,
  следующие функции нигде в системе не вызывались и были удалены:
    _deduplicate_by_f3, _build_chains, _merge_chains, build_chain_simple,
    incremental_append_video, get_chunks, incremental_get_chunks,
    build_chunks_in_range, incremental_append_audio_tracks,
    build_audio_chunks_from_structured, build_audio_chunks_v2,
    incremental_build_audio_chunks_v2, rebuild_audio_chunks_from,
    get_first_chunk, fast_video_records, build_idr_map,
    _has_nal_type5_in_frame.
  Вместе с ними убраны константа AUDIO_CHUNK_TAIL_UPDATE (нужна была только
  incremental_build_audio_chunks_v2) и AUDIO_SAMPLERATE/VIDEO_FPS (нужны
  были только для локального вычисления констант, теперь импортируемых).

  Если где-то в вашей инфраструктуре (вне присланных файлов) всё же есть
  код, вызывающий что-то из удалённого списка — дайте знать, восстановлю
  точечно из истории правок.

Оставлены и НЕ менялись по логике: _filter_normal_records, _abs_offset,
build_audio_tracks, build_chunks_from_cached_offsets,
build_audio_chunks_in_range, get_idr_indices_from_mmap — это то, что
реально используется LazyIndex и idx_cache.
"""

import logging
import numpy as np
from typing import List, Dict

from config.timebase import FRAMES_PER_CHUNK, SAMPLES_PER_VIDEO_FRAME, SAMPLES_PER_CHUNK

logger = logging.getLogger(__name__)

SEGMENT_SIZE = 4_294_967_296

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

# ----------------------------------------------------------------------
# Фильтрация записей и абсолютные смещения
# ----------------------------------------------------------------------
def _filter_normal_records(records): return records[records['f2'] < 256]


def _abs_offset(rec):
    return (rec['f1'].astype(np.uint64) + rec['f2'].astype(np.uint64) * np.uint64(SEGMENT_SIZE)
            - np.uint64(4))


# ----------------------------------------------------------------------
# Чанки
# ----------------------------------------------------------------------
def build_chunks_from_cached_offsets(video_slice, cached_offsets, mdat_end):
    """Строит (chunk_offsets, chunk_sizes) для окна, используя предвычисленные смещения."""
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
# Аудио (векторизованный расчёт size2)
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


def build_audio_chunks_in_range(audio_arr: np.ndarray, start_chunk: int, num_chunks: int):
    """
    Построение аудиочанков только для указанного диапазона чанков.
    Возвращает список словарей, по длине равный num_chunks.
    Использует предвычисленный индекс.
    """
    if len(audio_arr) == 0 or num_chunks == 0:
        return [{} for _ in range(num_chunks)]

    # Приведение к int64, чтобы избежать переполнения uint64 при вычитании
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


# ----------------------------------------------------------------------
# IDR
# ----------------------------------------------------------------------
def get_idr_indices_from_mmap(mm_193: np.ndarray) -> np.ndarray:
    """Возвращает индексы IDR-кадров (f7 = 27, 28, 29, 30)."""
    idr_mask = (mm_193['f7'] == 27) | (mm_193['f7'] == 28) | \
               (mm_193['f7'] == 29) | (mm_193['f7'] == 30)
    return np.where(idr_mask)[0].astype(np.int64)
