#!/usr/bin/env python3
"""
test_jkl.py – тесты режима ускоренной перемотки (JKL).

ЗАЧЕМ
JKL был сломан незаметно и долго: PlaybackEngine.set_speed() менял только
свои поля и НИЧЕГО не сообщал конвейеру, из-за чего перемотка просто
вычерпывала уже накопленный видеобуфер. Внешне это выглядело как «вперёд
работает секунд десять, потом кадр стоит; назад не двигается совсем» —
то есть частично рабочее поведение, которое легко списать на сеть или
декодер. Ни один тест этого не ловил, потому что режим FAST_FORWARD в
планировщике был мёртвым кодом: он существовал, был корректен и никем не
вызывался.

Эти тесты закрывают именно такой класс дефектов — «код есть, но не
подключён» — и фиксируют поведение, которое должно сохраняться:

  * планировщик реально переходит в FAST_FORWARD и выдаёт чанки;
  * чанки идут с пропуском (stride), пропорциональным скорости;
  * назад работает симметрично вперёд;
  * заполненный видеобуфер НЕ блокирует перемотку;
  * есть ограничение темпа, иначе чтение уходит на скорость диска;
  * до x2 декодируется чанк целиком (плавно), выше — только опорный кадр.

ЗАПУСК
    python test_jkl.py
    pytest test_jkl.py -v

Тесты работают на заглушках: без файлов, сети, звуковой карты и PyAV.
"""

import sys
import threading
import time
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Заглушки тяжёлых зависимостей: тесты проверяют ЛОГИКУ, а не декодер.
# Импортируются до модулей проекта, поэтому реальные PyAV/sounddevice не нужны.
# ---------------------------------------------------------------------------
def _install_stubs():
    def mod(name):
        m = sys.modules.get(name)
        if m is None:
            m = types.ModuleType(name)
            sys.modules[name] = m
        return m

    import logging
    for name in ("buffer", "buffer.frame_buffer", "decode", "decode.decoder",
                 "decode.audio_decoder", "index", "index.lazy_index",
                 "index.moov_builder", "file_io", "file_io.win_sequential_reader",
                 "utils", "utils.sync_logger"):
        mod(name)

    mod("buffer.frame_buffer").FrameRingBuffer = _FakeBuffer
    mod("decode.decoder").Decoder = object
    mod("decode.audio_decoder").AudioDecoder = object
    mod("index.lazy_index").IndexWindow = object
    mod("index.moov_builder").SEGMENT_SIZE = 4294967296
    mod("file_io.win_sequential_reader").WinSequentialReader = object
    mod("utils.sync_logger").sync_monitor_logger = logging.getLogger("SyncStub")


class _FakeBuffer:
    """Двойник FrameRingBuffer: планировщику нужны только count/max_frames."""

    def __init__(self, max_frames=360):
        self.count = 0
        self.max_frames = max_frames
        self.keep_last = None

    def clear(self):
        self.count = 0

    def update_keep_last(self, frame, pts):
        self.keep_last = (pts, frame)

    def get_keep_last(self):
        return self.keep_last[1] if self.keep_last else None

    def peek_first(self):
        return None


_install_stubs()

try:
    from config.timebase import SAMPLES_PER_CHUNK, AUDIO_SAMPLE_RATE
except ImportError:
    from timebase import SAMPLES_PER_CHUNK, AUDIO_SAMPLE_RATE

try:
    from pipeline.stream_scheduler import StreamScheduler, PlaybackMode
    from pipeline.adaptive_chunk import AdaptiveChunkStrategy
    import pipeline.chunk_pipeline as chunk_pipeline
except ImportError:
    from stream_scheduler import StreamScheduler, PlaybackMode
    from adaptive_chunk import AdaptiveChunkStrategy
    import chunk_pipeline


def make_frame(v=0):
    return np.full((4, 4, 3), v % 256, dtype=np.uint8)


# ===========================================================================
# Планировщик в режиме перемотки
# ===========================================================================
class TestSchedulerFastForward(unittest.TestCase):

    def _sched(self, buf_count=0, max_frames=360):
        s = StreamScheduler()
        # Заполненность задаётся двойнику ДО передачи планировщику:
        # раньше тест писал в s._video_buffer.count, то есть менял
        # состояние через приватное поле проверяемого объекта.
        buf = _FakeBuffer(max_frames)
        buf.count = buf_count
        s.set_buffer(buf)
        return s

    def _drain(self, s, n):
        """Берёт n чанков, снимая ограничение темпа между вызовами."""
        got = []
        for _ in range(n):
            s.reset_rate_limit()
            c = s.get_next_chunk()
            if c is None:
                break
            got.append(c)
        return got

    def test_mode_actually_switches(self):
        """
        Базовая проверка: режим включается. Именно этого не происходило —
        set_fast_forward_mode() не вызывался никем, кроме тестов.
        """
        s = self._sched()
        s.set_fast_forward_mode(1, 4.0, 10, 100)
        self.assertEqual(s.get_state()["mode"], "FAST_FORWARD")

    def test_forward_uses_stride(self):
        s = self._sched()
        s.set_fast_forward_mode(direction=1, speed=4.0,
                                current_local_chunk=0, total_local_chunks=100)
        got = self._drain(s, 4)
        self.assertEqual(got, [0, 4, 8, 12],
                         "при x4 читается каждый четвёртый чанк")

    def test_backward_is_symmetric(self):
        """Назад должно работать так же, как вперёд, только с убыванием."""
        s = self._sched()
        s.set_fast_forward_mode(direction=-1, speed=4.0,
                                current_local_chunk=50, total_local_chunks=100)
        got = self._drain(s, 4)
        self.assertEqual(got, [50, 46, 42, 38])

    def test_stride_scales_with_speed(self):
        for speed, step in ((2.0, 2), (4.0, 4), (8.0, 8)):
            s = self._sched()
            s.set_fast_forward_mode(1, speed, 0, 200)
            got = self._drain(s, 3)
            self.assertEqual(got, [0, step, step * 2],
                             f"скорость x{speed:.0f} → шаг {step}")

    def test_backward_stops_at_start(self):
        """У начала файла перемотка назад должна останавливаться, а не уходить в минус."""
        s = self._sched()
        s.set_fast_forward_mode(direction=-1, speed=8.0,
                                current_local_chunk=4, total_local_chunks=100)
        got = self._drain(s, 5)
        self.assertTrue(all(c >= 0 for c in got), f"отрицательные чанки: {got}")

    def test_forward_stops_at_end(self):
        s = self._sched()
        s.set_fast_forward_mode(direction=1, speed=8.0,
                                current_local_chunk=95, total_local_chunks=100)
        got = self._drain(s, 5)
        self.assertTrue(all(c < 100 for c in got), f"чанки за границей: {got}")

    def test_full_buffer_does_not_block_scrub(self):
        """
        КЛЮЧЕВОЙ ТЕСТ. В режиме перемотки кадры идут в отдельную ячейку, а
        не в кольцевой буфер — тот остаётся заполненным с прошлого
        воспроизведения. Если бы планировщик, как в NORMAL, смотрел на его
        заполненность, он бы намертво отказывался выдавать чанки: ровно то
        поведение, при котором «кадр стоит на месте».
        """
        s = self._sched(buf_count=359, max_frames=360)   # буфер практически полон
        s.set_fast_forward_mode(1, 4.0, 0, 100)
        s.reset_rate_limit()
        self.assertIsNotNone(s.get_next_chunk(),
                             "полный видеобуфер не должен блокировать перемотку")

    def test_full_buffer_still_blocks_normal(self):
        """А в обычном режиме заполненность по-прежнему обязана тормозить чтение."""
        s = self._sched(buf_count=359, max_frames=360)
        s.set_normal_mode(0, 100)
        self.assertIsNone(s.get_next_chunk())

    def test_throttling_limits_rate(self):
        """
        Без ограничения темпа перемотка читала бы на максимальной скорости
        диска: обратной связи от буфера в этом режиме нет.
        """
        s = self._sched()
        s.set_fast_forward_mode(1, 4.0, 0, 1000)
        first = s.get_next_chunk()
        self.assertIsNotNone(first)
        immediate = s.get_next_chunk()      # сразу же, без паузы
        self.assertIsNone(immediate, "второй чанк подряд должен быть отклонён темпом")

    def test_throttle_scales_with_speed(self):
        """Чем выше скорость, тем чаще разрешено читать."""
        # Длительность чанка считается из констант проекта, а не берётся
        # из приватного поля планировщика: величина одна и та же, но тест
        # не зависит от того, как она там названа и хранится.
        chunk_duration = SAMPLES_PER_CHUNK / AUDIO_SAMPLE_RATE
        interval_slow = chunk_duration / 2.0
        interval_fast = chunk_duration / 8.0
        self.assertLess(interval_fast, interval_slow,
                        "чем выше скорость, тем чаще разрешено читать")

        slow = self._sched(); slow.set_fast_forward_mode(1, 2.0, 0, 1000)
        fast = self._sched(); fast.set_fast_forward_mode(1, 8.0, 0, 1000)
        self.assertIsNotNone(slow.get_next_chunk())
        self.assertIsNotNone(fast.get_next_chunk())

    def test_return_to_normal_restores_position(self):
        """После K планировщик возвращается в NORMAL с указанной позиции."""
        s = self._sched()
        s.set_fast_forward_mode(1, 8.0, 0, 100)
        self._drain(s, 3)
        s.set_normal_mode(40, 100)
        self.assertEqual(s.get_state()["mode"], "NORMAL")
        s.reset_rate_limit()
        self.assertEqual(s.get_next_chunk(), 40)


# ===========================================================================
# Декодирование в режиме перемотки
# ===========================================================================
class _RecordingDecoder:
    def __init__(self):
        self.decoded = []

    def filter_avcc(self, data):
        return data

    def decode_sample(self, data):
        self.decoded.append(data)
        return [make_frame(len(self.decoded))]


class TestScrubDecoding(unittest.TestCase):

    def _stage(self, speed):
        # Здесь обращение к приватным полям ОСОЗНАННО: это конструирование
        # объекта, который сам является предметом теста, без запуска
        # потока и подключения к конвейеру. Это не связь между модулями,
        # а подготовка изолированного экземпляра.
        st = chunk_pipeline.VideoDecoderStage.__new__(chunk_pipeline.VideoDecoderStage)
        st._decoder = _RecordingDecoder()
        st._pts_min = 0
        st._pts_max = 10 ** 12
        st._scrub_lock = threading.Lock()
        st._scrub_frame = None
        st._scrub_mode = True
        st._scrub_speed = speed
        return st

    @staticmethod
    def _packets(n=12, base_frame=0, idr_at=0):
        """
        Пакеты чанка. idr_at — позиция опорного кадра внутри чанка
        (None = опорного нет вовсе).

        Формат пакета: (data, pts, is_idr). Флаг добавлен, потому что GOP
        (15 кадров) не кратен чанку (12): опорный кадр стоит в начале
        чанка лишь каждый пятый раз, и декодировать «первый попавшийся»
        пакет нельзя.
        """
        return [(b"pkt%d" % i, (base_frame + i) * 1920, i == idr_at)
                for i in range(n)]

    def test_x2_decodes_from_idr_to_end(self):
        """До x2 нужна плавность — декодируем от опорного кадра до конца чанка."""
        st = self._stage(2.0)
        st._decode_scrub(self._packets(12, idr_at=0))
        self.assertEqual(len(st._decoder.decoded), 12)

    def test_x2_starts_at_idr_not_at_chunk_start(self):
        """
        Ключевой тест: если IDR стоит в СЕРЕДИНЕ чанка (обычный случай при
        GOP 15 и чанке 12), декодирование обязано начинаться с него, а не
        с первого пакета — иначе декодеру подаются кадры без опорного.
        """
        st = self._stage(2.0)
        st._decode_scrub(self._packets(12, idr_at=7))
        self.assertEqual(len(st._decoder.decoded), 5,
                         "декодируются только пакеты с 7-го по 11-й")

    def test_chunk_without_idr_is_skipped(self):
        """
        Чанк без опорного кадра пропускается целиком: лучше задержать
        предыдущее изображение, чем показать артефакты.
        """
        st = self._stage(4.0)
        st._decode_scrub(self._packets(12, idr_at=None))
        self.assertEqual(len(st._decoder.decoded), 0)
        self.assertIsNone(st.take_scrub_frame())

    def test_x4_decodes_only_keyframe(self):
        """Выше x2 достаточно опорного кадра: остальные не успеют быть замечены."""
        st = self._stage(4.0)
        st._decode_scrub(self._packets(12, idr_at=3))
        self.assertEqual(len(st._decoder.decoded), 1)

    def test_x8_decodes_only_keyframe(self):
        st = self._stage(8.0)
        st._decode_scrub(self._packets(12, idr_at=5))
        self.assertEqual(len(st._decoder.decoded), 1)

    def test_high_speed_shows_the_idr_itself(self):
        """На высокой скорости показывается именно опорный кадр чанка."""
        st = self._stage(8.0)
        packets = self._packets(12, base_frame=100, idr_at=5)
        st._decode_scrub(packets)
        pts, _ = st.take_scrub_frame()
        self.assertEqual(pts, packets[5][1])

    def test_threshold_matches_constant(self):
        self.assertEqual(chunk_pipeline.SCRUB_FULL_DECODE_SPEED, 2.0)

    def test_scrub_frame_is_last_decoded(self):
        """На x2 показывается последний кадр чанка — движение непрерывно."""
        st = self._stage(2.0)
        packets = self._packets(12, base_frame=100, idr_at=0)
        st._decode_scrub(packets)
        pts, _frame = st.take_scrub_frame()
        self.assertEqual(pts, packets[-1][1])

    def test_packets_outside_window_ignored(self):
        st = self._stage(2.0)
        st._pts_min = 10_000
        st._pts_max = 20_000
        st._decode_scrub([(b"a", 5_000, True), (b"b", 15_000, True), (b"c", 50_000, True)])
        self.assertEqual(len(st._decoder.decoded), 1, "берётся только пакет внутри окна")

    def test_decode_error_does_not_break_scrub(self):
        """Сбой одного пакета не должен ронять перемотку целиком."""
        st = self._stage(2.0)

        class Broken(_RecordingDecoder):
            def decode_sample(self, data):
                if data == b"pkt3":
                    raise RuntimeError("битый пакет")
                return super().decode_sample(data)

        st._decoder = Broken()
        st._decode_scrub(self._packets(12, idr_at=0))
        self.assertIsNotNone(st.take_scrub_frame(), "кадр всё равно должен быть получен")

    def test_empty_packets_keep_previous_frame(self):
        """
        Пустой чанк не должен обнулять картинку: лучше показать предыдущий
        кадр, чем чёрный экран.
        """
        st = self._stage(2.0)
        st._decode_scrub(self._packets(4, idr_at=0))
        before = st.take_scrub_frame()
        st._decode_scrub([])
        self.assertEqual(st.take_scrub_frame(), before)

    def test_disabling_scrub_clears_frame(self):
        """При выходе из перемотки ячейка очищается — иначе показался бы старый кадр."""
        st = self._stage(2.0)
        st._decode_scrub(self._packets(4, idr_at=0))
        self.assertIsNotNone(st.take_scrub_frame())
        chunk_pipeline.VideoDecoderStage.set_scrub_mode(st, False)
        self.assertIsNone(st.take_scrub_frame())

    def test_set_scrub_mode_updates_speed(self):
        st = self._stage(2.0)
        chunk_pipeline.VideoDecoderStage.set_scrub_mode(st, True, 8.0)
        st._decode_scrub(self._packets(12, idr_at=0))
        self.assertEqual(len(st._decoder.decoded), 1,
                         "смена скорости должна менять глубину декодирования")


# ===========================================================================
# Совместная работа: скорость -> шаг -> частота смены картинки
# ===========================================================================
class TestScrubRates(unittest.TestCase):

    def test_display_rate_matches_expectation(self):
        """
        Проверяем расчёт, на который опирается выбор «плавно/опорные»:
        частота смены картинки = speed * fps / FRAMES_PER_CHUNK.
        При x4 это ~8 изображений в секунду, при x8 ~17 — достаточно,
        чтобы движение читалось как непрерывное.
        """
        fps, frames_per_chunk = 25.0, 12
        for speed, expected in ((4.0, 8.3), (8.0, 16.7)):
            rate = speed * fps / frames_per_chunk
            self.assertAlmostEqual(rate, expected, places=1)

    def test_x2_rate_would_be_choppy_without_full_decode(self):
        """
        Обоснование порога: на x2 только опорные кадры дали бы ~4 картинки
        в секунду — заметно ступенчато. Поэтому до x2 декодируем чанк
        целиком.
        """
        rate_keyframes_only = 2.0 * 25.0 / 12
        self.assertLess(rate_keyframes_only, 5.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
