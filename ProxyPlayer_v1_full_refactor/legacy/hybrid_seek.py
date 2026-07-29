#!/usr/bin/env python3
"""
hybrid_seek.py – асинхронный гибридный seek для DaletProxy V6 (статический режим).
Работает с video_records (цепочкой), как в v5.
"""

import logging
from pathlib import Path
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from timebase import SAMPLES_PER_VIDEO_FRAME
from frame_buffer import FrameRingBuffer
from decoder import Decoder
from moov_builder import SEGMENT_SIZE
from win_sequential_reader import WinSequentialReader

logger = logging.getLogger(__name__)

LOOKAHEAD_FRAMES = 12


class HybridSeekWorker(QObject):
    finished = pyqtSignal(bool, int, int)

    def __init__(self, mp4_path: Path, mdat_end: int, decoder: Decoder,
                 buffer: FrameRingBuffer, frame_idx: int,
                 video_records: np.ndarray,
                 callback=None):
        super().__init__()
        self.mp4_path = mp4_path
        self.mdat_end = mdat_end
        self.decoder = decoder
        self.buffer = buffer
        self.frame_idx = frame_idx
        self.video_records = video_records
        self.callback = callback
        self.next_chunk_to_load = 0
        self._stopped = False

    def stop(self):
        self._stopped = True

    def run(self):
        if self._stopped:
            self._report(False, self.frame_idx, 0)
            return
        success = self._do_seek()
        if not self._stopped:
            self._report(success, self.frame_idx, self.next_chunk_to_load)

    def _report(self, success, f_idx, next_chunk):
        if self.callback:
            self.callback(success, f_idx, next_chunk)
        else:
            self.finished.emit(success, f_idx, next_chunk)

    def _find_nearest_idr(self, target_frame):
        """Ищет ближайший IDR-кадр в video_records."""
        if len(self.video_records) == 0:
            return 0
        idr_mask = (self.video_records['f7'] == 29) | (self.video_records['f7'] == 30)
        idr_indices = np.where(idr_mask)[0]
        if len(idr_indices) == 0:
            return 0
        pos = np.searchsorted(idr_indices, target_frame, side='right') - 1
        if pos < 0:
            return idr_indices[0]
        return idr_indices[pos]

    def _do_seek(self) -> bool:
        if len(self.video_records) == 0:
            return False

        logger.info(f"Гибридный seek к кадру {self.frame_idx}")
        idr_index = self._find_nearest_idr(self.frame_idx)
        idr_rec = self.video_records[idr_index]
        idr_f3 = int(idr_rec['f3'])
        idr_offset = int(idr_rec['f1']) - 4 + int(idr_rec['f2']) * SEGMENT_SIZE

        # Декодируем до целевого кадра, но не меньше LOOKAHEAD_FRAMES от IDR
        end_index = min(len(self.video_records) - 1, max(idr_index + LOOKAHEAD_FRAMES, self.frame_idx))
        if end_index + 1 < len(self.video_records):
            next_rec = self.video_records[end_index + 1]
            read_end_offset = int(next_rec['f1']) - 4 + int(next_rec['f2']) * SEGMENT_SIZE
        else:
            read_end_offset = self.mdat_end

        read_size = read_end_offset - idr_offset
        if read_size <= 0:
            return False

        try:
            with WinSequentialReader(self.mp4_path, rate_limit=0, overlapped=False) as rdr:
                raw_data = rdr.read_sequential(idr_offset, read_size)
        except Exception as e:
            logger.error(f"Ошибка чтения: {e}")
            return False

        if not raw_data:
            return False

        base_offset = idr_offset
        decoded_frames = []
        for i in range(idr_index, end_index + 1):
            if self._stopped:
                return False
            rec = self.video_records[i]
            off = int(rec['f1']) - 4 + int(rec['f2']) * SEGMENT_SIZE
            rel_start = off - base_offset
            if i + 1 < len(self.video_records):
                next_off = int(self.video_records[i + 1]['f1']) - 4 + int(self.video_records[i + 1]['f2']) * SEGMENT_SIZE
                rel_end = next_off - base_offset
            else:
                rel_end = len(raw_data)
            rel_end = min(rel_end, len(raw_data))
            if rel_start < 0 or rel_end <= rel_start:
                continue
            sample = raw_data[rel_start:rel_end]
            filtered = self.decoder.filter_avcc(sample)
            if not filtered:
                continue
            try:
                frames = self.decoder.decode_sample(filtered)
                decoded_frames.extend(frames)
            except Exception as e:
                logger.warning(f"Ошибка декодирования {i}: {e}")
                continue

        if not decoded_frames:
            return False

        self.buffer.clear()
        for i, frame in enumerate(decoded_frames):
            if self._stopped:
                return False
            pts = (idr_f3 + i) * SAMPLES_PER_VIDEO_FRAME
            if not self.buffer.try_push(frame, pts):
                break

        self.next_chunk_to_load = max(0, (idr_f3 // 12) + 1)
        logger.info(f"Seek успешен: {self.buffer.count} кадров (IDR={idr_index}, target={self.frame_idx})")
        return True