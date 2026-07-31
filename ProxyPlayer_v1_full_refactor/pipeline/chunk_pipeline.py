"""
chunk_pipeline.py – трёхэтапный конвейер загрузки и декодирования чанков.
Работает с локальными индексами чанков внутри IndexWindow.
"""

import queue
import threading
import logging
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np

from buffer.frame_buffer import FrameRingBuffer
from buffer.audio_buffer import MultiTrackAudioBuffer
from decode.decoder import Decoder
from decode.audio_decoder import AudioDecoder
from file_io.win_sequential_reader import WinSequentialReader
from index.lazy_index import IndexWindow
from index.moov_builder import SEGMENT_SIZE
from pipeline.stream_scheduler import StreamScheduler, PlaybackMode
from pipeline.adaptive_chunk import AdaptiveChunkStrategy
from config.timebase import SAMPLES_PER_VIDEO_FRAME, FRAMES_PER_CHUNK

logger = logging.getLogger(__name__)

VideoPacket = Tuple[bytes, int]                       # (data, pts)
AudioPacket = Tuple[int, bytes, bytes, int, bool]     # (track, data1, data2, pts, need_fade)

RAW_QUEUE_SIZE = 3
VIDEO_QUEUE_SIZE = 5
AUDIO_QUEUE_SIZE = 5


class Stage(threading.Thread):
    def __init__(self, name: str):
        super().__init__(daemon=True, name=name)
        self.stop_event = threading.Event()

    def stop(self):
        self.stop_event.set()


class ReaderStage(Stage):
    def __init__(self, mp4_path: Path, window: IndexWindow,
                 scheduler: StreamScheduler, output_queue: queue.Queue,
                 adaptive: AdaptiveChunkStrategy):
        super().__init__("ReaderStage")
        self._reader = WinSequentialReader(mp4_path, rate_limit=0, overlapped=False)
        self._window = window
        self._scheduler = scheduler
        self._output_queue = output_queue
        self._adaptive = adaptive

    def run(self):
        while not self.stop_event.is_set():
            local_chunk = self._scheduler.get_next_chunk()
            if local_chunk is None:
                self.stop_event.wait(0.1)
                continue

            if local_chunk >= self._window.total_chunks:
                self._scheduler.mark_chunk_failed(local_chunk)
                continue

            off = int(self._window.chunk_offsets[local_chunk])
            size = int(self._window.chunk_sizes[local_chunk])
            if size <= 0:
                self._scheduler.mark_chunk_failed(local_chunk)
                continue

            try:
                data = self._reader.read_sequential(off, size)
                if data:
                    self._output_queue.put((local_chunk, data), timeout=1.0)
                else:
                    self._scheduler.mark_chunk_failed(local_chunk)
            except Exception as e:
                logger.error(f"Ошибка чтения чанка {local_chunk}: {e}")
                self._scheduler.mark_chunk_failed(local_chunk)

    def stop(self):
        super().stop()
        self._reader.close()


class DemuxerStage(Stage):
    def __init__(self, window: IndexWindow, input_queue: queue.Queue,
                 video_queue: queue.Queue, audio_queue: queue.Queue):
        super().__init__("DemuxerStage")
        self._window = window
        self._input_queue = input_queue
        self._video_queue = video_queue
        self._audio_queue = audio_queue

    def run(self):
        while not self.stop_event.is_set():
            try:
                local_chunk, raw_data = self._input_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                v_packets, a_packets = self._demux(local_chunk, raw_data)
                if v_packets:
                    self._video_queue.put((local_chunk, v_packets), timeout=1.0)
                if a_packets:
                    self._audio_queue.put((local_chunk, a_packets), timeout=1.0)
            except Exception as e:
                logger.error(f"Ошибка демукса чанка {local_chunk}: {e}")

    def _demux(self, local_chunk: int, raw_data: bytes) -> Tuple[List[VideoPacket], List[AudioPacket]]:
        video_packets = []
        audio_packets = []

        # Глобальный номер первого кадра этого чанка
        global_start_frame = (self._window.window_start_chunk + local_chunk) * FRAMES_PER_CHUNK
        chunk_start_offset = int(self._window.chunk_offsets[local_chunk])
        cached_offs = self._window.cached_offsets

        # ---------- видео ----------
        for i in range(FRAMES_PER_CHUNK):
            abs_idx = global_start_frame + i
            local_frame = abs_idx - self._window.window_start_frame
            if local_frame < 0 or local_frame >= len(self._window.video_records):
                continue

            rec = self._window.video_records[local_frame]
            if cached_offs is not None and local_frame < len(cached_offs):
                abs_off = int(cached_offs[local_frame])
            else:
                abs_off = int(rec['f1']) - 4 + int(rec['f2']) * SEGMENT_SIZE

            # размер кадра
            if i < FRAMES_PER_CHUNK - 1:
                next_local = local_frame + 1
                if next_local < len(self._window.video_records):
                    if cached_offs is not None and next_local < len(cached_offs):
                        next_off = int(cached_offs[next_local])
                    else:
                        next_rec = self._window.video_records[next_local]
                        next_off = int(next_rec['f1']) - 4 + int(next_rec['f2']) * SEGMENT_SIZE
                    size = next_off - abs_off
                else:
                    size = 0
            else:
                # последний кадр чанка
                chunk_end = chunk_start_offset + int(self._window.chunk_sizes[local_chunk])
                size = chunk_end - abs_off

            if size <= 0:
                continue

            rel_start = abs_off - chunk_start_offset
            if rel_start < 0 or rel_start + size > len(raw_data):
                continue

            sample = raw_data[rel_start:rel_start + size]
            pts = abs_idx * SAMPLES_PER_VIDEO_FRAME
            video_packets.append((sample, pts))

        # ---------- аудио ----------
        audio_chunk = self._window.audio_chunks[local_chunk] if local_chunk < len(self._window.audio_chunks) else {}
        if audio_chunk:
            for track_id, entries in audio_chunk.items():
                if track_id == 0 or track_id > 3:
                    continue
                for entry in entries:
                    size2 = entry.get('size2', 0)
                    rel = entry['abs_offset'] - chunk_start_offset

                    d1 = b''
                    if 0 <= rel < len(raw_data):
                        d1 = raw_data[rel:rel + entry['size1']]

                    d2 = b''
                    if size2 > 0:
                        rel2 = rel + entry['size1']
                        if 0 <= rel2 < len(raw_data):
                            d2 = raw_data[rel2:rel2 + size2]

                    audio_packets.append((track_id, d1, d2, entry['pts'], False))

        return video_packets, audio_packets


class VideoDecoderStage(Stage):
    def __init__(self, decoder: Decoder, video_buffer: FrameRingBuffer,
                 input_queue: queue.Queue):
        super().__init__("VideoDecoderStage")
        self._decoder = decoder
        self._buffer = video_buffer
        self._input_queue = input_queue

    def run(self):
        while not self.stop_event.is_set():
            try:
                _, packets = self._input_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            for data, pts in packets:
                try:
                    filtered = self._decoder.filter_avcc(data)
                    if not filtered:
                        continue
                    frames = self._decoder.decode_sample(filtered)
                    for frame in frames:
                        while not self.stop_event.is_set():
                            if self._buffer.try_push(frame, pts):
                                break
                            self.stop_event.wait(0.01)
                except Exception as e:
                    logger.debug(f"Ошибка декодирования видео: {e}")


class AudioDecoderStage(Stage):
    def __init__(self, decoders: List[AudioDecoder], audio_buffers: MultiTrackAudioBuffer,
                 input_queue: queue.Queue):
        super().__init__("AudioDecoderStage")
        self._decoders = decoders
        self._audio_buffers = audio_buffers
        self._input_queue = input_queue
        self._fade_len = 8

    def run(self):
        while not self.stop_event.is_set():
            try:
                _, packets = self._input_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            for track_id, d1, d2, pts, need_fade in packets:
                if track_id < 2 or track_id > 3:
                    continue
                decoder_idx = track_id - 2
                if decoder_idx >= len(self._decoders) or self._decoders[decoder_idx] is None:
                    continue

                decoder = self._decoders[decoder_idx]
                try:
                    pcm1 = decoder.decode(d1)
                    pcm2 = decoder.decode(d2)
                    pcm_block = np.concatenate([pcm1, pcm2])
                except Exception:
                    continue

                if need_fade and len(pcm_block) >= self._fade_len:
                    pcm_block[:self._fade_len] *= np.linspace(0, 1, self._fade_len)

                while not self.stop_event.is_set():
                    if self._audio_buffers.try_write(track_id, pcm_block, pts):
                        break
                    self.stop_event.wait(0.01)


class ChunkPipeline:
    def __init__(self, mp4_path: Path, window: IndexWindow,
                 video_decoder: Decoder, audio_decoders: List[AudioDecoder],
                 video_buffer: FrameRingBuffer, audio_buffers: MultiTrackAudioBuffer):
        self._mp4_path = mp4_path
        self._window = window
        self._video_decoder = video_decoder
        self._audio_decoders = audio_decoders
        self._video_buffer = video_buffer
        self._audio_buffers = audio_buffers

        self._scheduler = StreamScheduler(AdaptiveChunkStrategy())
        self._raw_queue = queue.Queue(maxsize=RAW_QUEUE_SIZE)
        self._video_queue = queue.Queue(maxsize=VIDEO_QUEUE_SIZE)
        self._audio_queue = queue.Queue(maxsize=AUDIO_QUEUE_SIZE)

        self._stages: List[Stage] = []

    def start(self, start_local_chunk: int):
        self._scheduler.set_normal_mode(start_local_chunk, self._window.total_chunks)

        reader = ReaderStage(self._mp4_path, self._window, self._scheduler,
                             self._raw_queue, AdaptiveChunkStrategy())
        demuxer = DemuxerStage(self._window, self._raw_queue,
                               self._video_queue, self._audio_queue)
        video_dec = VideoDecoderStage(self._video_decoder, self._video_buffer,
                                      self._video_queue)
        audio_dec = AudioDecoderStage(self._audio_decoders, self._audio_buffers,
                                      self._audio_queue)

        self._stages = [reader, demuxer, video_dec, audio_dec]
        for stage in self._stages:
            stage.start()

    def stop(self):
        for stage in self._stages:
            stage.stop()
        for stage in self._stages:
            stage.join(timeout=2.0)
        self._stages.clear()

    def update_window(self, new_window: IndexWindow):
        self._window = new_window
        # Перезапускаем планировщик с начала нового окна
        self._scheduler.set_normal_mode(0, new_window.total_chunks)