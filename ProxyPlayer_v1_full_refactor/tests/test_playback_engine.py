"""
Интеграционный тест PlaybackEngine на реальном файле.
Требует TEST_MP4_PATH. Аудиовыход замокан, чтобы не требовать звукового устройства.
"""

import os
import time
import threading
from pathlib import Path

import pytest
import numpy as np

from buffer.frame_buffer import FrameRingBuffer
from buffer.audio_buffer import MultiTrackAudioBuffer
from decode.decoder import Decoder
from decode.audio_decoder import AudioDecoder
from index.lazy_index import LazyIndex
from index.idx_cache import prepare_mirror
from file_io.win_sequential_reader import WinSequentialReader
from utils.utils import get_real_size
from pipeline.chunk_pipeline import ChunkPipeline
from seek.seek_engine import SeekEngine
from core.sync_manager import SyncManager
from core.playback_engine import PlaybackEngine

# --- настройка тестового файла ---
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


class MockAudioOutput:
    """Заглушка AudioOutput для тестов."""
    def __init__(self):
        self._active = False
        self._volume = 0.8
        self._muted = False
        self._clock = 0

    def start(self):
        self._active = True

    def stop(self):
        self._active = False

    def reset_clock(self, pts):
        self._clock = pts

    def set_volume(self, vol):
        self._volume = vol

    def set_muted(self, muted):
        self._muted = muted

    @property
    def active(self):
        return self._active


@pytest.fixture(scope="module")
def engine():
    mirror = prepare_mirror(idx_path)
    mdat_end = get_real_size(str(mp4_path))
    lazy = LazyIndex(mirror, mp4_path, mdat_end)
    window = lazy.open_window(0)

    decoder = Decoder(avcc_data)
    audio_decoders = [AudioDecoder(asc), AudioDecoder(asc)]
    video_buf = FrameRingBuffer(max_frames=300)
    audio_buf = MultiTrackAudioBuffer(capacity_samples=480000)

    pipeline = ChunkPipeline(mp4_path, window, decoder, audio_decoders, video_buf, audio_buf)
    reader = WinSequentialReader(mp4_path, rate_limit=0, overlapped=False)
    seek_engine = SeekEngine(lazy, decoder, reader)
    sync_mgr = SyncManager()
    audio_out = MockAudioOutput()

    engine = PlaybackEngine(
        pipeline, seek_engine, sync_mgr, audio_out,
        video_buf, audio_buf,
        total_frames=len(window.video_records),
        fps=25.0,
    )
    yield engine
    engine.close()


class TestPlaybackEngine:
    def test_start_and_get_frame(self, engine):
        engine.start_playback(0)
        # Ждём появления кадров (до 10 сек)
        waited = 0
        frame = None
        while waited < 10:
            frame = engine.get_display_frame()
            if frame is not None:
                break
            time.sleep(0.5)
            waited += 0.5
        assert frame is not None, "Не получен кадр после старта"

    def test_seek(self, engine):
        engine.seek(1500)
        time.sleep(2)  # даём время на seek и подгрузку
        frame = engine.get_display_frame()
        # После seek движок на паузе, get_display_frame возвращает keep_last
        assert frame is not None, "Нет кадра после seek"

    def test_jkl_speed(self, engine):
        engine.set_speed(1)  # L
        assert engine._seek_speed in [2.0, 4.0, 8.0]
        engine.reset_speed()
        assert engine._seek_speed == 1.0
        assert engine._seek_direction == 0