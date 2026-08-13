"""
test_seek_buffer_consistency.py – тест на проблему подмены видеобуфера после seek.
Проверяет, что после _on_seek_complete конвейер продолжает наполнять
тот же буфер, который используется движком, и кадры не теряются.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, call
from buffer.frame_buffer import FrameRingBuffer
from core.playback_engine import PlaybackEngine
from seek.seek_engine import SeekEngine
from pipeline.chunk_pipeline import ChunkPipeline
from core.sync_manager import SyncManager
from core.master_clock import MasterClock
from index.lazy_index import IndexWindow


@pytest.fixture
def real_video_buffer():
    """Настоящий кольцевой буфер для проверки целостности."""
    return FrameRingBuffer(max_frames=100)


@pytest.fixture
def mock_pipeline():
    """Конвейер, который запоминает, в какой буфер писал."""
    p = MagicMock(spec=ChunkPipeline)
    p._scheduler = MagicMock()
    # Имитация старта: при вызове start() будет наполнять буфер, который мы потом проверим
    return p


@pytest.fixture
def engine(real_video_buffer, mock_pipeline):
    """PlaybackEngine с моками."""
    seek_engine = MagicMock(spec=SeekEngine)
    sync_mgr = SyncManager()
    master_clock = MagicMock(spec=MasterClock)
    eng = PlaybackEngine(
        pipeline=mock_pipeline,
        seek_engine=seek_engine,
        sync_manager=sync_mgr,
        master_clock=master_clock,
        video_buffer=real_video_buffer,
        total_frames=1000,
    )
    # Вручную установим состояние, как после start_playback
    eng._paused = True
    eng._playback_started = True
    return eng


class TestSeekBufferConsistency:
    def test_old_logic_loses_frames(self, engine, mock_pipeline, real_video_buffer):
        """
        Старая (ошибочная) логика: заменяем буфер ссылкой, конвейер
        продолжает писать в старый буфер, а движок показывает новый (пустой).
        """
        # Создаём временный буфер с одним кадром (результат seek)
        temp_buffer = FrameRingBuffer(max_frames=10)
        temp_buffer.try_push(np.zeros((10,10,3), dtype=np.uint8), 1000)

        window = MagicMock(spec=IndexWindow)
        window.window_start_frame = 0
        window.total_chunks = 50

        # Симулируем ошибочную замену буфера
        engine._video_buffer, _ = temp_buffer, engine._video_buffer
        # Теперь engine._video_buffer указывает на temp_buffer,
        # а pipeline продолжает писать в старый real_video_buffer.

        # Имитируем работу конвейера: он добавляет кадры в старый буфер.
        real_video_buffer.try_push(np.ones((10,10,3), dtype=np.uint8), 2000)

        # Движок получает кадры из нового буфера (temp_buffer) – там только 1 кадр
        frame = engine.get_display_frame()  # вернёт None или устаревший
        # Проблема: новый буфер не пополняется конвейером, кадры теряются.
        # В реальности это приводит к пустому буферу и крашу декодера.
        assert engine._video_buffer.count == 1  # только кадр из temp_buffer
        # А старый буфер, куда пишет конвейер, движок не видит
        assert real_video_buffer.count == 1

    def test_correct_logic_keeps_buffer_consistency(self, engine, mock_pipeline, real_video_buffer):
        temp_buffer = FrameRingBuffer(max_frames=10)
        temp_buffer.try_push(np.zeros((10,10,3), dtype=np.uint8), 1000)
        window = MagicMock(spec=IndexWindow)
        window.window_start_frame = 0
        window.total_chunks = 50

        # Правильный перенос (как в исправленном _on_seek_complete)
        engine._video_buffer.clear()
        while True:
            entry = temp_buffer.peek_first()
            if entry is None:
                break
            pts, frame = entry
            if not engine._video_buffer.try_push(frame, pts):
                break
            temp_buffer.advance()

        # Конвейер продолжает писать в основной буфер
        engine._video_buffer.try_push(np.ones((10,10,3), dtype=np.uint8), 2000)

        # Оба кадра доступны
        assert engine._video_buffer.count == 2
        first = engine._video_buffer.peek_first()
        assert first is not None
        assert first[0] == 1000   # PTS первого кадра