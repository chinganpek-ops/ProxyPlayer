"""
test_chunk_pipeline.py – тесты для ChunkPipeline (исправленная версия).
Совместим с обновлённым chunk_pipeline.py: VideoDecoderStage с Condition,
дорожки 2/3, корректные размеры чанков.
"""

import pytest
import numpy as np
import queue
import threading
import time
from unittest.mock import MagicMock, patch

from pipeline.chunk_pipeline import (
    ChunkPipeline, DemuxerStage, VideoDecoderStage, AudioDecoderStage
)
from index.moov_builder import DTYPE_193
from config.timebase import FRAMES_PER_CHUNK, SAMPLES_PER_VIDEO_FRAME


# ------------------------------------------------------------------
# Вспомогательная функция для создания фейкового IndexWindow
# ------------------------------------------------------------------
def make_fake_window(num_frames=12, start_frame=0, chunk_size=None):
    """
    Создаёт IndexWindow с video_records, cached_offsets и chunk_sizes.
    По умолчанию генерирует один чанк из num_frames кадров.
    """
    recs = np.zeros(num_frames, dtype=DTYPE_193)
    cached = np.zeros(num_frames, dtype=np.uint64)
    for i in range(num_frames):
        abs_off = 1000 + i * 5000
        recs['f1'][i] = abs_off + 4
        recs['f2'][i] = 0
        recs['f3'][i] = i
        cached[i] = abs_off
    # chunk_sizes — размер одного чанка (если чанк один)
    if chunk_size is None:
        chunk_size = num_frames * 5000
    chunk_sizes = np.array([chunk_size], dtype=np.int64)

    window = MagicMock()
    window.video_records = recs
    window.window_start_frame = start_frame
    window.window_start_chunk = start_frame // FRAMES_PER_CHUNK
    window.total_chunks = 1
    window.cached_offsets = cached
    window.chunk_sizes = chunk_sizes
    window.audio_chunks = [{}]   # один пустой аудиочанк
    return window


@pytest.fixture
def mock_video_decoder():
    dec = MagicMock()
    dec.filter_avcc.return_value = b'filtered'
    dec.decode_sample.return_value = [np.zeros((10, 10, 3), dtype=np.uint8)]
    return dec


@pytest.fixture
def mock_audio_decoders():
    return [MagicMock(), MagicMock()]


@pytest.fixture
def mock_video_buffer():
    buf = MagicMock()
    buf.try_push.return_value = True
    return buf


@pytest.fixture
def mock_master_clock():
    return MagicMock()


# ------------------------------------------------------------------
# Тесты DemuxerStage
# ------------------------------------------------------------------
class TestDemuxerStage:
    def test_demux_video_packets(self):
        window = make_fake_window(num_frames=12)
        # raw_data должно быть не меньше суммы размеров кадров
        raw_data = bytes([0] * (12 * 5000))
        demuxer = DemuxerStage(queue.Queue(), queue.Queue(), queue.Queue())
        v_packets, _ = demuxer._demux(0, raw_data, window)
        assert len(v_packets) == 12
        expected_base_pts = window.window_start_chunk * FRAMES_PER_CHUNK * SAMPLES_PER_VIDEO_FRAME
        assert v_packets[0][1] == expected_base_pts

    def test_demux_audio_valid_tracks(self):
        window = make_fake_window(num_frames=12)
        window.audio_chunks[0] = {
            2: [{'abs_offset': 1000, 'size1': 100, 'size2': 50, 'pts': 0}],
            3: [{'abs_offset': 2000, 'size1': 80, 'size2': 40, 'pts': 1024}]
        }
        raw_data = bytes([0] * (12 * 5000))  # достаточно
        demuxer = DemuxerStage(queue.Queue(), queue.Queue(), queue.Queue())
        _, a_packets = demuxer._demux(0, raw_data, window)
        assert len(a_packets) == 2
        tracks = {p[0] for p in a_packets}
        assert tracks == {2, 3}

    def test_demux_audio_ignores_invalid_tracks(self):
        window = make_fake_window(num_frames=12)
        window.audio_chunks[0] = {
            1: [{'abs_offset': 100, 'size1': 10, 'size2': 0, 'pts': 0}],
            4: [{'abs_offset': 200, 'size1': 10, 'size2': 0, 'pts': 0}],
            2: [{'abs_offset': 1000, 'size1': 100, 'size2': 0, 'pts': 0}]
        }
        raw_data = bytes([0] * (12 * 5000))
        demuxer = DemuxerStage(queue.Queue(), queue.Queue(), queue.Queue())
        _, a_packets = demuxer._demux(0, raw_data, window)
        # только дорожка 2
        assert len(a_packets) == 1
        assert a_packets[0][0] == 2

    def test_safe_put_retries_on_full_queue(self):
        demuxer = DemuxerStage(queue.Queue(), queue.Queue(), queue.Queue())
        q = queue.Queue(maxsize=1)
        q.put("block")
        demuxer._safe_put(q, "test", "test_q", max_retries=2, retry_delay=0.01)
        assert q.get() == "block"
        assert q.empty()


# ------------------------------------------------------------------
# Тесты VideoDecoderStage (с Condition)
# ------------------------------------------------------------------
class TestVideoDecoderStage:
    def test_push_frame_success(self, mock_video_decoder, mock_video_buffer):
        stage = VideoDecoderStage(mock_video_decoder, mock_video_buffer, queue.Queue())
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        stage._push_frame(frame, 100)
        mock_video_buffer.try_push.assert_called_once_with(frame, 100)

    def test_notify_buffer_available_exists(self, mock_video_decoder, mock_video_buffer):
        stage = VideoDecoderStage(mock_video_decoder, mock_video_buffer, queue.Queue())
        # Убедимся, что метод есть
        assert hasattr(stage, 'notify_buffer_available')

    def test_run_processes_packets(self, mock_video_decoder, mock_video_buffer):
        q = queue.Queue()
        stage = VideoDecoderStage(mock_video_decoder, mock_video_buffer, q)
        q.put((0, [(b'\x00\x01', 1920)]))
        stage.start()
        time.sleep(0.2)
        stage.stop()
        stage.join(timeout=1)
        mock_video_decoder.filter_avcc.assert_called()
        mock_video_decoder.decode_sample.assert_called()
        mock_video_buffer.try_push.assert_called()


# ------------------------------------------------------------------
# Тесты AudioDecoderStage
# ------------------------------------------------------------------
class TestAudioDecoderStage:
    def test_audio_pushes_to_master_clock(self, mock_audio_decoders, mock_master_clock):
        q = queue.Queue()
        stage = AudioDecoderStage(mock_audio_decoders, mock_master_clock, q)
        q.put((0, [(2, b'\x00', b'\x01', 1000, False)]))
        stage.start()
        time.sleep(0.1)
        stage.stop()
        stage.join(timeout=1)
        mock_master_clock.push_audio.assert_called()
        track_id = mock_master_clock.push_audio.call_args[0][0]
        assert track_id == 2

    def test_invalid_track_skipped(self, mock_audio_decoders, mock_master_clock):
        q = queue.Queue()
        stage = AudioDecoderStage(mock_audio_decoders, mock_master_clock, q)
        q.put((0, [(1, b'\x00', b'\x00', 0, False)]))
        stage.start()
        time.sleep(0.1)
        stage.stop()
        stage.join(timeout=1)
        mock_master_clock.push_audio.assert_not_called()


# ------------------------------------------------------------------
# Интеграционные тесты ChunkPipeline
# ------------------------------------------------------------------
class TestChunkPipelineIntegration:
    def test_start_stop_pipeline(self, mock_video_decoder, mock_audio_decoders,
                                 mock_video_buffer, mock_master_clock):
        window = make_fake_window(num_frames=12)
        pipeline = ChunkPipeline(
            mp4_path=MagicMock(),
            window=window,
            video_decoder=mock_video_decoder,
            audio_decoders=mock_audio_decoders,
            video_buffer=mock_video_buffer,
            master_clock=mock_master_clock
        )
        with patch('pipeline.chunk_pipeline.WinSequentialReader') as MockReader:
            mock_reader_instance = MockReader.return_value
            mock_reader_instance.read_sequential.return_value = bytes([0] * (12 * 5000))
            pipeline.start(0)
            time.sleep(0.3)
            pipeline.stop()
            mock_reader_instance.read_sequential.assert_called()

    def test_pipeline_with_audio(self, mock_video_decoder, mock_audio_decoders,
                                 mock_video_buffer, mock_master_clock):
        window = make_fake_window(num_frames=12)
        window.audio_chunks[0] = {
            2: [{'abs_offset': 1000, 'size1': 200, 'size2': 100, 'pts': 0}]
        }
        pipeline = ChunkPipeline(
            mp4_path=MagicMock(),
            window=window,
            video_decoder=mock_video_decoder,
            audio_decoders=mock_audio_decoders,
            video_buffer=mock_video_buffer,
            master_clock=mock_master_clock
        )
        with patch('pipeline.chunk_pipeline.WinSequentialReader') as MockReader:
            MockReader.return_value.read_sequential.return_value = bytes([0] * (12 * 5000))
            pipeline.start(0)
            time.sleep(0.3)
            pipeline.stop()
            mock_master_clock.push_audio.assert_called()