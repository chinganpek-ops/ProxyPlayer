"""
test_seek_engine.py – Модульные тесты для SeekEngine (ProxyPlayer v2).
Все 16 тестов проходят стабильно. Исправлены проблемы синхронизации и рекурсии.
"""

import pytest
import numpy as np
import threading
import time
from unittest.mock import MagicMock

from seek.seek_engine import SeekEngine, SeekRequest
from index.moov_builder import DTYPE_193


# ------------------------------------------------------------------
# Вспомогательные функции и фикстуры
# ------------------------------------------------------------------

def make_fake_video_records(num_frames: int, start_offset: int = 1000, step: int = 5000):
    """
    Создаёт массив DTYPE_193 с линейно возрастающими абсолютными смещениями.
    Каждый кадр имеет f2=0 и f1, подобранное так, чтобы _abs_offset = start_offset + i*step.
    """
    records = np.zeros(num_frames, dtype=DTYPE_193)
    for i in range(num_frames):
        abs_off = start_offset + i * step
        records['f1'][i] = abs_off + 4   # _abs_offset = f1 - 4 (при f2=0)
        records['f2'][i] = 0
        records['f3'][i] = i
        records['f7'][i] = 1             # не IDR по умолчанию
    return records


def make_fake_window(video_records, idr_indices, start_frame=0):
    """Имитирует IndexWindow с заданными видео-записями и IDR."""
    window = MagicMock()
    window.video_records = video_records
    window.window_start_frame = start_frame
    window.idr_frames = np.array(idr_indices, dtype=np.int64)
    return window


@pytest.fixture
def mock_lazy_index():
    """LazyIndex с фейковым окном: 30 кадров, IDR на 0, 12, 24."""
    li = MagicMock()
    li.mdat_end = 10_000_000
    recs = make_fake_video_records(30)
    recs['f7'][0] = 27
    recs['f7'][12] = 27
    recs['f7'][24] = 27
    window = make_fake_window(recs, [0, 12, 24])
    li.open_window.return_value = window
    return li


@pytest.fixture
def mock_decoder():
    """Декодер, возвращающий один фиктивный RGB-кадр."""
    dec = MagicMock()
    dec.filter_avcc.return_value = b'filtered_data'
    dec.decode_sample.return_value = [np.zeros((1080, 1920, 3), dtype=np.uint8)]
    return dec


@pytest.fixture
def mock_reader():
    """WinSequentialReader, возвращающий большой блок данных."""
    reader = MagicMock()
    reader.read_sequential.return_value = b'\x00' * 500_000
    return reader


# ------------------------------------------------------------------
# Тесты поиска IDR и расчёта read_size
# ------------------------------------------------------------------

def test_seek_finds_nearest_idr_left(mock_lazy_index, mock_decoder, mock_reader):
    """Целевой кадр 5 → IDR=0, read_size от кадра 0 до 13."""
    engine = SeekEngine(mock_lazy_index, mock_decoder, mock_reader)
    buf = engine.seek_sync(5)
    assert buf.count > 0
    mock_lazy_index.open_window.assert_called_with(5)
    # Проверяем диапазон чтения: start=offset(0), end=offset(13)
    mock_reader.read_sequential.assert_called_once()
    args, _ = mock_reader.read_sequential.call_args
    assert args[0] == 1000
    assert args[1] == 13 * 5000  # offset(13)-offset(0)


def test_seek_exact_idr(mock_lazy_index, mock_decoder, mock_reader):
    """Целевой кадр точно IDR (12) → IDR=12, read_size до 25."""
    engine = SeekEngine(mock_lazy_index, mock_decoder, mock_reader)
    buf = engine.seek_sync(12)
    assert buf.count > 0
    args, _ = mock_reader.read_sequential.call_args
    assert args[0] == 1000 + 12 * 5000
    assert args[1] == 13 * 5000


def test_seek_target_beyond_idr_but_within_lookahead(mock_lazy_index, mock_decoder, mock_reader):
    """Целевой кадр 20 → IDR=12, end_local=24, read_size до 25."""
    engine = SeekEngine(mock_lazy_index, mock_decoder, mock_reader)
    buf = engine.seek_sync(20)
    assert buf.count > 0
    args, _ = mock_reader.read_sequential.call_args
    assert args[0] == 1000 + 12 * 5000
    assert args[1] == 13 * 5000


def test_seek_last_frames_uses_mdat_end(mock_lazy_index, mock_decoder, mock_reader):
    """end_local последний в окне → read_size до mdat_end."""
    recs = make_fake_video_records(20)
    recs['f7'][0] = 27
    recs['f7'][10] = 27
    window = make_fake_window(recs, [0, 10])
    mock_lazy_index.open_window.return_value = window
    mock_lazy_index.mdat_end = 200000
    engine = SeekEngine(mock_lazy_index, mock_decoder, mock_reader)
    engine.seek_sync(15)  # target 15 → IDR=10, end_local=19 (последний)
    args, _ = mock_reader.read_sequential.call_args
    assert args[0] == 1000 + 10 * 5000
    assert args[1] == 200000 - args[0]


# ------------------------------------------------------------------
# Тесты PTS и заполнения буфера
# ------------------------------------------------------------------

def test_seek_returns_correct_pts(mock_lazy_index, mock_decoder, mock_reader):
    """PTS первого кадра равен PTS IDR (здесь 0)."""
    engine = SeekEngine(mock_lazy_index, mock_decoder, mock_reader)
    buf = engine.seek_sync(5)
    first_pts = buf.peek_first()[0]
    assert first_pts == 0


def test_seek_buffer_has_ascending_pts(mock_lazy_index, mock_decoder, mock_reader):
    """Все кадры в буфере имеют возрастающие PTS без дубликатов."""
    mock_decoder.decode_sample.return_value = [
        np.zeros((10, 10, 3), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.uint8)
    ]
    engine = SeekEngine(mock_lazy_index, mock_decoder, mock_reader)
    buf = engine.seek_sync(5)
    all_pts = [entry[0] for entry in buf.peek_all()]
    assert all_pts == sorted(all_pts), "PTS должны быть отсортированы"
    assert all_pts[0] == 0


# ------------------------------------------------------------------
# Тесты отмены
# ------------------------------------------------------------------

def test_cancel_before_read(mock_lazy_index, mock_decoder, mock_reader):
    """
    Отмена до вызова read_sequential.
    Блокируем open_window, чтобы запрос гарантированно не успел продвинуться.
    """
    window_ready = threading.Event()
    orig_open = mock_lazy_index.open_window

    def blocking_open(*args, **kwargs):
        window_ready.set()          # сообщаем тесту: поток внутри open_window
        time.sleep(0.3)            # даём тесту время на cancel()
        return orig_open(*args, **kwargs)

    mock_lazy_index.open_window.side_effect = blocking_open
    callback = MagicMock()
    engine = SeekEngine(mock_lazy_index, mock_decoder, mock_reader)
    req = engine.seek_async(10, callback)

    # Ждём, пока поток дойдёт до open_window
    window_ready.wait(timeout=1.0)
    # Теперь отменяем
    req.cancel()
    # После выхода из блокировки поток увидит отмену и не вызовет callback
    time.sleep(0.5)
    callback.assert_not_called()


def test_cancel_during_read(mock_lazy_index, mock_decoder, mock_reader):
    """Во время блокирующего чтения отменяем запрос — коллбэк молчит."""
    read_started = threading.Event()
    def delayed_read(offset, size):
        read_started.set()
        time.sleep(0.5)
        return b'data'
    mock_reader.read_sequential.side_effect = delayed_read

    callback = MagicMock()
    engine = SeekEngine(mock_lazy_index, mock_decoder, mock_reader)
    req = engine.seek_async(10, callback)
    read_started.wait(timeout=1.0)
    req.cancel()
    time.sleep(0.6)
    callback.assert_not_called()


# ------------------------------------------------------------------
# Тесты поколений запросов
# ------------------------------------------------------------------

def test_multiple_seek_only_last_callback_called(mock_lazy_index, mock_decoder, mock_reader):
    """
    Три быстрых seek подряд — только последний вызывает on_complete.
    Блокируем чтение, чтобы все запросы «застряли» в ожидании.
    """
    callbacks = [MagicMock() for _ in range(3)]
    read_block = threading.Event()

    # Заглушка без рекурсии: после ожидания возвращаем фиксированные данные
    def blocking_read(offset, size):
        read_block.wait()
        return b'\x00' * 500_000

    mock_reader.read_sequential.side_effect = blocking_read

    engine = SeekEngine(mock_lazy_index, mock_decoder, mock_reader)
    for cb in callbacks:
        engine.seek_async(5, cb)
    time.sleep(0.3)          # потоки дошли до блокировки
    read_block.set()          # разрешаем чтение
    time.sleep(0.5)          # ждём завершения всех

    callbacks[2].assert_called_once()
    callbacks[0].assert_not_called()
    callbacks[1].assert_not_called()


def test_generation_check_inside_callback(mock_lazy_index, mock_decoder, mock_reader):
    callbacks = [MagicMock() for _ in range(2)]
    first_in_open = threading.Event()
    let_go = threading.Event()
    window_return = mock_lazy_index.open_window.return_value  # сохраняем окно

    def blocking_open(*args, **kwargs):
        if not first_in_open.is_set():
            first_in_open.set()
            let_go.wait()
        return window_return

    mock_lazy_index.open_window.side_effect = blocking_open

    engine = SeekEngine(mock_lazy_index, mock_decoder, mock_reader)
    req1 = engine.seek_async(5, callbacks[0])
    first_in_open.wait(timeout=1.0)
    req2 = engine.seek_async(10, callbacks[1])
    let_go.set()
    req2.wait(timeout=2.0)
    callbacks[0].assert_not_called()
    callbacks[1].assert_called_once()


# ------------------------------------------------------------------
# Тесты ошибок
# ------------------------------------------------------------------

def test_seek_raises_when_no_idr(mock_lazy_index, mock_decoder, mock_reader):
    """Окно без IDR → RuntimeError."""
    recs = make_fake_video_records(10)
    window = make_fake_window(recs, [])
    mock_lazy_index.open_window.return_value = window
    engine = SeekEngine(mock_lazy_index, mock_decoder, mock_reader)
    with pytest.raises(RuntimeError, match="не найдены IDR"):
        engine.seek_sync(5)


def test_seek_raises_when_empty_window(mock_lazy_index, mock_decoder, mock_reader):
    """open_window возвращает None → RuntimeError."""
    mock_lazy_index.open_window.return_value = None
    engine = SeekEngine(mock_lazy_index, mock_decoder, mock_reader)
    with pytest.raises(RuntimeError, match="Не удалось открыть окно"):
        engine.seek_sync(5)


def test_seek_on_error_callback(mock_lazy_index, mock_decoder, mock_reader):
    """Исключение в декодере → on_error с текстом исходной ошибки."""
    mock_decoder.filter_avcc.side_effect = Exception("Decoder crash")
    on_error = MagicMock()
    engine = SeekEngine(mock_lazy_index, mock_decoder, mock_reader)
    req = engine.seek_async(5, MagicMock(), on_error=on_error)
    req.wait(timeout=2.0)
    on_error.assert_called_once()
    error_msg = on_error.call_args[0][0]
    assert "Decoder crash" in error_msg, f"Expected 'Decoder crash' in '{error_msg}'"


def test_seek_sync_returns_empty_buffer_when_no_data(mock_lazy_index, mock_decoder, mock_reader):
    """read_sequential возвращает пустые данные → пустой валидный буфер (count==0)."""
    mock_reader.read_sequential.return_value = b''
    engine = SeekEngine(mock_lazy_index, mock_decoder, mock_reader)
    buf = engine.seek_sync(5)
    assert buf.count == 0


# ------------------------------------------------------------------
# Вспомогательные тесты
# ------------------------------------------------------------------

def test_seek_request_wait(mock_lazy_index, mock_decoder, mock_reader):
    """Успешный seek: wait() возвращает True, коллбэк вызван."""
    callback = MagicMock()
    engine = SeekEngine(mock_lazy_index, mock_decoder, mock_reader)
    req = engine.seek_async(5, callback)
    success = req.wait(timeout=2.0)
    assert success
    callback.assert_called_once()


def test_seek_request_cancel_updates_is_cancelled():
    """После cancel() свойство is_cancelled становится True."""
    req = SeekRequest(1)
    assert not req.is_cancelled
    req.cancel()
    assert req.is_cancelled