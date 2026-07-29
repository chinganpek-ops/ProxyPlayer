"""
Тестирование SeekEngine на реальном файле.
Требует путь к MP4 файлу через переменную окружения TEST_MP4_PATH.
"""
import os
import sys
import time
from pathlib import Path

import pytest

from index.lazy_index import LazyIndex
from index.idx_cache import prepare_mirror
from file_io.win_sequential_reader import WinSequentialReader
from decode.decoder import Decoder
from seek.seek_engine import SeekEngine
from utils.utils import get_real_size

TEST_MP4 = os.environ.get('TEST_MP4_PATH')
if not TEST_MP4:
    pytest.skip("TEST_MP4_PATH не задан", allow_module_level=True)

mp4_path = Path(TEST_MP4)
stem = mp4_path.stem
parent = mp4_path.parent
idx_path = parent / "idx" / "mp4" / f"{stem}.idx"
ref_path = parent / f"{stem}.mp4.ref"

if not idx_path.exists():
    idx_path = parent / f"{stem}.idx"

avcc_data = b''
if ref_path.exists():
    from file_io.ref_parser import extract_ftyp_avcc
    _, avcc_data = extract_ftyp_avcc(ref_path)

if not avcc_data:
    avcc_hex = "014d001fffe1002e674d401f9652816824dff80200016a50101014000003000400000300cb8180009600000301e848fc6383b428532c01000568e9093520"
    avcc_data = bytes.fromhex(avcc_hex)

@pytest.fixture(scope="module")
def lazy_index():
    mirror = prepare_mirror(idx_path)
    mdat_end = get_real_size(str(mp4_path))
    return LazyIndex(mirror, mp4_path, mdat_end)

@pytest.fixture(scope="module")
def seek_engine(lazy_index):
    decoder = Decoder(avcc_data)
    reader = WinSequentialReader(mp4_path, rate_limit=0, overlapped=False)
    return SeekEngine(lazy_index, decoder, reader)

def test_seek_sync(seek_engine):
    buffer = seek_engine.seek_sync(1500)
    assert buffer.count > 0
    first = buffer.peek_first()
    assert first is not None
    print(f"Seek к 1500: первый кадр pts={first[0]}")

def test_seek_async(seek_engine):
    result_buffer = None
    error_msg = []

    def on_complete(buf):
        nonlocal result_buffer
        result_buffer = buf

    def on_error(msg):
        error_msg.append(msg)

    request = seek_engine.seek_async(3000, on_complete, on_error)
    request.wait(timeout=10)
    assert result_buffer is not None, f"Seek failed: {error_msg}"
    assert result_buffer.count > 0