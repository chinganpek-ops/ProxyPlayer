"""
Интеграционный тест ChunkPipeline на реальном файле.
Требует переменную окружения TEST_MP4_PATH.
"""

import os
import sys
import time
import queue
from pathlib import Path

import pytest

from buffer.frame_buffer import FrameRingBuffer
from buffer.audio_buffer import MultiTrackAudioBuffer
from decode.decoder import Decoder
from decode.audio_decoder import AudioDecoder
from index.lazy_index import LazyIndex
from index.idx_cache import prepare_mirror
from file_io.win_sequential_reader import WinSequentialReader
from utils.utils import get_real_size
from pipeline.chunk_pipeline import ChunkPipeline

TEST_MP4 = os.environ.get('TEST_MP4_PATH')
if not TEST_MP4:
    pytest.skip("TEST_MP4_PATH не задан", allow_module_level=True)

mp4_path = Path(TEST_MP4)
stem = mp4_path.stem
parent = mp4_path.parent
idx_path = parent / "idx" / "mp4" / f"{stem}.idx"
if not idx_path.exists():
    idx_path = parent / f"{stem}.idx"
ref_path = parent / f"{stem}.mp4.ref"

avcc_data = b''
if ref_path.exists():
    from file_io.ref_parser import extract_ftyp_avcc
    _, avcc_data = extract_ftyp_avcc(ref_path)
if not avcc_data:
    avcc_hex = "014d001fffe1002e674d401f9652816824dff80200016a50101014000003000400000300cb8180009600000301e848fc6383b428532c01000568e9093520"
    avcc_data = bytes.fromhex(avcc_hex)

asc = b'\x11\x88'

@pytest.fixture(scope="module")
def pipeline():
    mirror = prepare_mirror(idx_path)
    mdat_end = get_real_size(str(mp4_path))
    lazy = LazyIndex(mirror, mp4_path, mdat_end)
    window = lazy.open_window(center_frame=0)

    decoder = Decoder(avcc_data)
    audio_decoders = [AudioDecoder(asc), AudioDecoder(asc)]
    video_buf = FrameRingBuffer(max_frames=300)
    audio_buf = MultiTrackAudioBuffer(capacity_samples=480000)

    pipe = ChunkPipeline(mp4_path, window, decoder, audio_decoders, video_buf, audio_buf)
    return pipe, video_buf, audio_buf

def test_pipeline_start_stop(pipeline):
    pipe, video_buf, audio_buf = pipeline
    pipe.start(start_chunk=0)
    # Даём время на загрузку нескольких чанков
    timeout = 10
    waited = 0
    while video_buf.count < 12 and waited < timeout:
        time.sleep(0.5)
        waited += 0.5
    pipe.stop()

    assert video_buf.count > 0, f"Видео-буфер пуст после {waited}с"
    print(f"Загружено {video_buf.count} видеокадров, "
          f"аудио: трек2={audio_buf.buffers[2].available_read} сэмплов, "
          f"трек3={audio_buf.buffers[3].available_read} сэмплов")