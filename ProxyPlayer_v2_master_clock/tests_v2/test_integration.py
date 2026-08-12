"""
test_integration.py – комплексный интеграционный тест для ProxyPlayer v2.
Проверяет сквозной сценарий: StreamController → PlaybackEngine → SeekEngine → MasterClock.
Все внешние зависимости замоканы (файлы, sounddevice, декодеры).
"""

import pytest
import numpy as np
import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path

from core.stream_controller import StreamController
from buffer.frame_buffer import FrameRingBuffer
from config.timebase import SAMPLES_PER_VIDEO_FRAME


# ------------------------------------------------------------------
# Фейковое окно индекса и LazyIndex
# ------------------------------------------------------------------
def make_fake_index_window():
    """Создаёт минимальное IndexWindow с несколькими чанками."""
    from index.moov_builder import DTYPE_193
    num_frames = 36  # три чанка по 12 кадров
    recs = np.zeros(num_frames, dtype=DTYPE_193)
    cached = np.zeros(num_frames, dtype=np.uint64)
    for i in range(num_frames):
        abs_off = 1000 + i * 5000
        recs['f1'][i] = abs_off + 4
        recs['f2'][i] = 0
        recs['f3'][i] = i
        recs['f7'][i] = 1
        cached[i] = abs_off
    # Делаем каждый 12-й кадр IDR
    recs['f7'][0] = 27
    recs['f7'][12] = 27
    recs['f7'][24] = 27

    window = MagicMock()
    window.video_records = recs
    window.window_start_frame = 0
    window.window_start_chunk = 0
    window.total_chunks = 3
    window.cached_offsets = cached
    window.chunk_sizes = np.array([5000*12]*3, dtype=np.int64)
    window.audio_chunks = [{}, {}, {}]  # без аудио для простоты
    window.idr_frames = np.array([0, 12, 24], dtype=np.int64)
    return window


@pytest.fixture
def mock_dependencies():
    """
    Патчит все внешние зависимости StreamController.
    Возвращает словарь с замоканными объектами для проверки вызовов.
    """
    fake_window = make_fake_index_window()

    # Мокаем LazyIndex и его методы
    mock_lazy_index = MagicMock()
    mock_lazy_index.open_window.return_value = fake_window
    mock_lazy_index._video_records_full = np.arange(36)  # 36 кадров

    # Мокаем WinSequentialReader – чтение возвращает достаточно байт
    mock_reader = MagicMock()
    mock_reader.read_sequential.return_value = bytes([0] * 5000 * 36)

    # Мокаем Decoder (видео) – возвращает по одному фейковому кадру на вызов
    mock_video_decoder = MagicMock()
    mock_video_decoder.filter_avcc.return_value = b'filtered'
    mock_video_decoder.decode_sample.return_value = [np.zeros((1080, 1920, 3), dtype=np.uint8)]

    # Мокаем AudioDecoder
    mock_audio_decoder = MagicMock()
    mock_audio_decoder.decode.return_value = np.zeros(1024, dtype=np.float64)

    # Мокаем MasterClock – не запускаем реальный sounddevice
    mock_master_clock = MagicMock()
    mock_master_clock.get_audio_clock.return_value = 0
    mock_master_clock.samples_played = 0

    # Мокаем SeekEngine – будем контролировать его поведение
    mock_seek_engine = MagicMock()

    # Патчим внутренние импорты StreamController
    with patch('core.stream_controller.LazyIndex', return_value=mock_lazy_index), \
         patch('core.stream_controller.WinSequentialReader', return_value=mock_reader), \
         patch('core.stream_controller.Decoder', return_value=mock_video_decoder), \
         patch('core.stream_controller.AudioDecoder', return_value=mock_audio_decoder), \
         patch('core.stream_controller.MasterClock', return_value=mock_master_clock), \
         patch('core.stream_controller.SeekEngine', return_value=mock_seek_engine), \
         patch('core.stream_controller.open_idx_mmap', return_value=(None, None)), \
         patch('core.stream_controller.get_mirror_path', return_value=Path('/fake/mirror.idx')), \
         patch('core.stream_controller.get_real_size', return_value=5000 * 36):

        yield {
            'lazy_index': mock_lazy_index,
            'reader': mock_reader,
            'video_decoder': mock_video_decoder,
            'audio_decoder': mock_audio_decoder,
            'master_clock': mock_master_clock,
            'seek_engine': mock_seek_engine,
            'window': fake_window,
        }


@pytest.fixture
def stream_controller(mock_dependencies):
    """Создаёт StreamController с замоканными зависимостями."""
    controller = StreamController(
        ref_path=Path('/fake/file.mp4'),
        idx_path=Path('/fake/file.idx'),
        mp4_path=Path('/fake/file.mp4'),
        fps=25.0,
        buffer_size=100,
        free_slots_required=5,
        group_chunks=2,
        start_from_live=False,
        mirror_path='/fake/mirror.idx',
    )
    # Ждём, пока фоновая инициализация завершится
    assert controller._ready.wait(timeout=5.0), "StreamController не инициализирован"
    assert controller._init_error is None, f"Ошибка инициализации: {controller._init_error}"
    return controller


# ------------------------------------------------------------------
# Интеграционные тесты
# ------------------------------------------------------------------
class TestIntegration:
    def test_start_playback_initializes_correctly(self, stream_controller, mock_dependencies):
        """После start_playback плеер должен быть в состоянии paused, буфер не пуст."""
        stream_controller.start_playback()
        # Даём время на загрузку первого кадра
        time.sleep(0.5)
        # Проверяем, что PlaybackEngine создан и запущен
        assert stream_controller._playback is not None
        # Плеер должен быть на паузе (paused)
        assert stream_controller._playback._paused is True
        assert stream_controller._playback.playing is False
        # В буфере должен быть хотя бы один кадр
        assert stream_controller.buffer_main.count >= 1

    def test_resume_pause_toggle(self, stream_controller):
        """Проверка переключения play/pause."""
        stream_controller.start_playback()
        time.sleep(0.3)

        # Возобновляем
        stream_controller.resume()
        assert stream_controller.playing is True
        stream_controller._playback._master_clock.start.assert_called_once()

        # Ставим на паузу
        stream_controller.pause()
        assert stream_controller.playing is False
        stream_controller._playback._master_clock.stop.assert_called_once()

    def test_seek_absolute(self, stream_controller, mock_dependencies):
        """Проверка, что seek вызывает SeekEngine и переводит в паузу."""
        stream_controller.start_playback()
        time.sleep(0.3)

        # Вызываем seek
        stream_controller.seek_absolute(12)
        # Должен быть вызван seek_engine.seek_async
        mock_dependencies['seek_engine'].seek_async.assert_called_once()
        # Плеер должен перейти в паузу
        assert stream_controller.playing is False
        # После завершения seek (симулируем вызов on_complete)
        # Для этого нужно найти аргументы вызова seek_async и вызвать on_complete с буфером
        args, kwargs = mock_dependencies['seek_engine'].seek_async.call_args
        on_complete = kwargs['on_complete']
        # Создаём буфер с одним кадром
        buf = FrameRingBuffer(max_frames=10)
        buf.try_push(np.zeros((10,10,3), dtype=np.uint8), 12 * 1920)
        on_complete(buf)
        # Теперь буфер должен обновиться, а плеер остаться на паузе
        assert stream_controller._playback._paused is True

    def test_jkl_speed_control(self, stream_controller):
        """JKL перемотка изменяет скорость и mute."""
        stream_controller.start_playback()
        time.sleep(0.2)

        # Включаем перемотку вперёд
        stream_controller.set_seek_speed(1)
        # MasterClock должен быть замьючен
        stream_controller._playback._master_clock.set_muted.assert_called_with(True)
        # Скорость должна быть установлена
        assert stream_controller._playback._seek_speed == 2.0

        # Сбрасываем скорость
        stream_controller.reset_seek_speed()
        stream_controller._playback._master_clock.set_muted.assert_called_with(False)
        assert stream_controller._playback._seek_speed == 1.0

    def test_get_display_frame_returns_frame(self, stream_controller):
        """Проверка, что get_display_frame возвращает numpy массив."""
        stream_controller.start_playback()
        time.sleep(0.3)
        # Во время паузы должен возвращать keep_last
        frame = stream_controller.get_display_frame()
        assert frame is not None
        assert isinstance(frame, np.ndarray)

    def test_close_releases_resources(self, stream_controller, mock_dependencies):
        """При закрытии все компоненты должны быть остановлены."""
        stream_controller.start_playback()
        stream_controller.close()
        # MasterClock должен быть закрыт
        mock_dependencies['master_clock'].close.assert_called()
        # PlaybackEngine остановлен
        # (Здесь можно добавить проверки на остановку конвейера, если нужно)
        assert stream_controller._closed is True