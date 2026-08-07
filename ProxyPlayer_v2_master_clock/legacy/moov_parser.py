"""
moov_parser.py – извлечение индексных структур из moov атома стандартного MP4.
Генерирует avcC, ASC, video_records (DTYPE_193), chunk_offsets/sizes и audio_tracks.
Совместим с PlayerController без изменений.

Версия production: детальное логирование, строгий контроль ошибок, явные fallback'и.
"""

import logging
from pathlib import Path
from collections import defaultdict
import numpy as np
import av

from timebase import AUDIO_SAMPLE_RATE, SAMPLES_PER_AAC, SAMPLES_PER_VIDEO_FRAME

logger = logging.getLogger(__name__)

# Константы, имитирующие .idx
FRAMES_PER_CHUNK = 12
FRAMES_PER_C9 = 2


def _get_pts_samples(packet, stream):
    """Переводит PTS пакета в сэмплы 48 кГц."""
    pts_sec = packet.pts * stream.time_base.numerator / stream.time_base.denominator
    return int(pts_sec * AUDIO_SAMPLE_RATE)


def parse_moov(mp4_path: Path):
    """
    Извлекает из MP4-файла все данные, необходимые плееру.
    Возвращает:
        avcc: bytes
        asc: bytes
        video_records: np.ndarray (DTYPE_193)
        chunk_offsets, chunk_sizes: np.ndarray int64
        audio_tracks: dict {track_id: [{'abs_offset','size1','size2','pts'}, ...]}
        total_frames: int
        start_frame_offset: int (всегда 0)
    В случае фатальных ошибок выбрасывает исключение с описанием.
    """
    logger.info(f"Парсинг moov для {mp4_path}")
    try:
        container = av.open(str(mp4_path))
    except Exception as e:
        raise ValueError(f"Не удалось открыть MP4-файл {mp4_path}: {e}")

    video_stream = None
    audio_streams = []
    for s in container.streams:
        if s.type == 'video' and video_stream is None:
            video_stream = s
        elif s.type == 'audio':
            audio_streams.append(s)

    if not video_stream:
        container.close()
        raise ValueError("Видеопоток не найден в файле")

    # --- avcC ---
    avcc = b''
    if video_stream.codec_context and video_stream.codec_context.extradata:
        avcc = bytes(video_stream.codec_context.extradata)
    if not avcc:
        container.close()
        raise ValueError("avcC не найден в видео")

    # --- ASC ---
    asc = b'\x11\x88'   # fallback: AAC-LC 48 кГц моно
    asc_found = False
    for s in audio_streams:
        if s.codec_context and s.codec_context.extradata:
            asc = bytes(s.codec_context.extradata)
            asc_found = True
            break
    if not asc_found:
        logger.warning("AudioSpecificConfig не найден в аудиопотоках, использую стандартный ASC 0x1188")

    # Сбор видеопакетов
    video_packets = []
    audio_packets_by_track = defaultdict(list)

    try:
        for packet in container.demux():
            if packet.stream == video_stream and packet.size > 0:
                pts_samples = _get_pts_samples(packet, video_stream)
                video_packets.append({
                    'offset': packet.pos,
                    'size': packet.size,
                    'pts': pts_samples,
                    'is_key': packet.is_keyframe
                })
            elif packet.stream in audio_streams and packet.size > 0:
                pts_samples = _get_pts_samples(packet, packet.stream)
                audio_packets_by_track[packet.stream.id].append({
                    'offset': packet.pos,
                    'size': packet.size,
                    'pts': pts_samples
                })
    except Exception as e:
        container.close()
        raise ValueError(f"Ошибка при демультиплексировании: {e}")
    finally:
        container.close()

    if not video_packets:
        raise ValueError("Не получено ни одного видеопакета")

    logger.info(f"Извлечено {len(video_packets)} видеопакетов, {sum(len(v) for v in audio_packets_by_track.values())} аудиопакетов")

    # Сортируем видеопакеты по PTS (из-за B-кадров порядок может отличаться)
    video_packets.sort(key=lambda p: p['pts'])
    total_frames = len(video_packets)

    # --- Строим video_records (DTYPE_193) ---
    dtype = np.dtype([
        ('f0', '<u4'), ('f1', '<u4'), ('f2', '<u4'), ('f3', '<u4'),
        ('f4', '<u4'), ('f5', '<u4'), ('f6', '<u4'), ('f7', '<u4'),
        ('f8', '<u4'), ('f9', '<u4'), ('f10','<u4'), ('f11','<u4'),
        ('f12','<u4'), ('f13','<u4'), ('f14','<u4'), ('f15','<u4'),
        ('f16','<u4')
    ])
    video_records = np.zeros(total_frames, dtype=dtype)
    video_records['f0'] = 0x193
    video_records['f2'] = 0
    for i, p in enumerate(video_packets):
        video_records[i]['f1'] = p['offset'] + 4   # чтобы abs_offset = f1-4
        video_records[i]['f3'] = p['pts'] // SAMPLES_PER_VIDEO_FRAME
        video_records[i]['f7'] = 29 if p['is_key'] else 1

    # --- Чанки (по 12 кадров) ---
    num_chunks = (total_frames + FRAMES_PER_CHUNK - 1) // FRAMES_PER_CHUNK
    chunk_offsets = np.zeros(num_chunks, dtype=np.int64)
    chunk_sizes = np.zeros(num_chunks, dtype=np.int64)
    for i in range(num_chunks):
        start_idx = i * FRAMES_PER_CHUNK
        end_idx = min((i + 1) * FRAMES_PER_CHUNK, total_frames) - 1
        start_offset = video_packets[start_idx]['offset']
        end_offset = video_packets[end_idx]['offset'] + video_packets[end_idx]['size']
        chunk_offsets[i] = start_offset
        chunk_sizes[i] = end_offset - start_offset

    # --- Аудио ---
    audio_tracks = {}
    for track_id, packets in audio_packets_by_track.items():
        if not packets:
            continue
        packets.sort(key=lambda p: p['pts'])
        c9_list = []
        for i in range(0, len(packets), FRAMES_PER_C9):
            p1 = packets[i]
            size2 = packets[i+1]['size'] if i+1 < len(packets) else 0
            c9_list.append({
                'abs_offset': p1['offset'],
                'size1': p1['size'],
                'size2': size2,
                'pts': p1['pts']
            })
        audio_tracks[track_id] = c9_list
        logger.debug(f"Аудиотрек {track_id}: {len(c9_list)} C9-блоков")

    logger.info(f"Moov-парсинг завершён: {total_frames} кадров, {num_chunks} чанков")
    return avcc, asc, video_records, chunk_offsets, chunk_sizes, audio_tracks, total_frames, 0