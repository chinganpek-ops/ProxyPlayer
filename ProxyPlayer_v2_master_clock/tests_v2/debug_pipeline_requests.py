#!/usr/bin/env python3
"""
Отслеживание запросов чанков конвейером и состояния буфера.
"""

import sys, time, logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)-8s | %(message)s')
logging.getLogger('av').setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from buffer.frame_buffer import FrameRingBuffer
from buffer.audio_buffer import MultiTrackAudioBuffer
from decode.decoder import Decoder
from decode.audio_decoder import AudioDecoder
from index.lazy_index import LazyIndex
from index.idx_cache import prepare_mirror
from utils.utils import get_real_size
from pipeline.chunk_pipeline import ChunkPipeline
from pipeline.stream_scheduler import StreamScheduler
from pipeline.adaptive_chunk import AdaptiveChunkStrategy
from config.timebase import FRAMES_PER_CHUNK


def monitor_requests(mp4_path: Path, timeout: float = 15.0):
    # Инициализация
    stem = mp4_path.stem
    parent = mp4_path.parent
    idx_path = parent / "idx" / "mp4" / f"{stem}.idx"
    if not idx_path.exists():
        idx_path = parent / f"{stem}.idx"
    ref_path = parent / f"{stem}.mp4.ref"

    mirror = prepare_mirror(idx_path)
    mdat_end = get_real_size(str(mp4_path))
    lazy = LazyIndex(mirror, mp4_path, mdat_end)
    window = lazy.open_window(center_frame=0)

    avcc = bytes.fromhex("014d001fffe1002e674d401f9652816824dff80200016a50101014000003000400000300cb8180009600000301e848fc6383b428532c01000568e9093520")
    decoder = Decoder(avcc, mp4_path)
    audio_decoders = [AudioDecoder(b'\x11\x88'), AudioDecoder(b'\x11\x88')]
    video_buf = FrameRingBuffer(max_frames=300)
    audio_buf = MultiTrackAudioBuffer(capacity_samples=480000)

    pipeline = ChunkPipeline(mp4_path, window, decoder, audio_decoders, video_buf, audio_buf)
    scheduler = pipeline._scheduler

    pipeline.start(start_local_chunk=0)
    start_time = time.monotonic()
    last_count = 0

    print(f"{'Время':<8} {'Видео':<6} {'Своб.':<6} {'Режим':<12} {'Тек.ч':<6} {'Всего':<6} {'Загр':<6} {'Ошб.чт':<6} {'Raw':<4} {'VidQ':<4} {'AudQ':<4}")
    print("-" * 85)

    while time.monotonic() - start_time < timeout:
        video_count = video_buf.count
        free = video_buf.free_slots
        mode = scheduler._mode.name if hasattr(scheduler, '_mode') else '?'
        cur = scheduler._current_chunk
        total = scheduler._total_chunks
        loaded = scheduler.loaded_count

        # Счётчики ошибок
        raw_q = pipeline._raw_queue.qsize()
        vid_q = pipeline._video_queue.qsize()
        aud_q = pipeline._audio_queue.qsize()

        print(f"{time.monotonic()-start_time:<8.1f} {video_count:<6} {free:<6} {mode:<12} {cur:<6} {total:<6} {loaded:<6} {'?':<6} {raw_q:<4} {vid_q:<4} {aud_q:<4}")

        time.sleep(0.5)

    pipeline.stop()
    print("Остановлено.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python tests/debug_pipeline_requests.py <путь_к_mp4>")
        sys.exit(1)
    monitor_requests(Path(sys.argv[1]))