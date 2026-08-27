#!/usr/bin/env python3
"""
test_units.py – юнит-тесты чистой логики ProxyPlayer.

Покрываются три модуля, выбранные по одному критерию: сложная логика
состояний БЕЗ внешних зависимостей. Их можно тестировать без PyQt5,
PyAV, sounddevice, звуковой карты и файлов на диске — то есть быстро и
на любой машине.

  * config/timebase.py       – преобразования времени и таймкодов
  * buffer/frame_buffer.py   – кольцевой буфер кадров, wraparound, keep-last
  * pipeline/stream_scheduler.py – автомат режимов, гистерезис, сдвиг окна

Именно здесь ошибки не видны глазом: буфер «почти работает», планировщик
выдаёт «почти те» чанки, таймкод парсится «почти правильно». В рантайме
это проявляется как редкие рассинхроны и пропавшие кадры, которые потом
ищутся часами по логам.

ЗАПУСК
    python test_units.py              # без pytest, обычным запуском
    pytest test_units.py -v           # если pytest установлен

Тесты не пишут на диск, не открывают сеть и завершаются за доли секунды.
"""

import sys
import threading
import unittest
from pathlib import Path

import numpy as np

# Проект может лежать как пакетами (config/, buffer/, pipeline/), так и
# плоско рядом с main.py — поддерживаем оба варианта.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    from config.timebase import (
        video_frame_to_pts, pts_to_video_frame, timecode_to_frame,
        samples_to_seconds, seconds_to_samples, compare_pts, pts_delta,
        SAMPLES_PER_VIDEO_FRAME, FRAMES_PER_CHUNK, SAMPLES_PER_CHUNK,
        AUDIO_SAMPLE_RATE,
    )
    from buffer.frame_buffer import FrameRingBuffer
    from pipeline.stream_scheduler import StreamScheduler, PlaybackMode
except ImportError as exc:  # плоская раскладка
    print(f"Импорт по пакетам не удался ({exc}), пробую плоскую раскладку...")
    from timebase import (
        video_frame_to_pts, pts_to_video_frame, timecode_to_frame,
        samples_to_seconds, seconds_to_samples, compare_pts, pts_delta,
        SAMPLES_PER_VIDEO_FRAME, FRAMES_PER_CHUNK, SAMPLES_PER_CHUNK,
        AUDIO_SAMPLE_RATE,
    )
    from frame_buffer import FrameRingBuffer
    from stream_scheduler import StreamScheduler, PlaybackMode


def make_frame(value: int = 0, h: int = 4, w: int = 4) -> np.ndarray:
    """Маленький кадр: содержимое не важно, важна идентификация по значению."""
    return np.full((h, w, 3), value % 256, dtype=np.uint8)


# ===========================================================================
# timebase
# ===========================================================================
class TestTimebase(unittest.TestCase):

    def test_frame_pts_roundtrip(self):
        for frame in (0, 1, 12, 1000, 157500):
            self.assertEqual(pts_to_video_frame(video_frame_to_pts(frame)), frame)

    def test_pts_scale(self):
        self.assertEqual(video_frame_to_pts(0), 0)
        self.assertEqual(video_frame_to_pts(1), SAMPLES_PER_VIDEO_FRAME)
        self.assertEqual(video_frame_to_pts(25), AUDIO_SAMPLE_RATE)  # 1 секунда

    def test_pts_truncation_inside_frame(self):
        """PTS в середине кадра должен относиться к этому же кадру."""
        base = video_frame_to_pts(10)
        self.assertEqual(pts_to_video_frame(base + SAMPLES_PER_VIDEO_FRAME - 1), 10)
        self.assertEqual(pts_to_video_frame(base + SAMPLES_PER_VIDEO_FRAME), 11)

    def test_chunk_constants_consistent(self):
        """Константы чанка должны быть согласованы между собой."""
        self.assertEqual(SAMPLES_PER_CHUNK, SAMPLES_PER_VIDEO_FRAME * FRAMES_PER_CHUNK)

    def test_seconds_roundtrip(self):
        self.assertAlmostEqual(samples_to_seconds(seconds_to_samples(3.5)), 3.5, places=6)

    def test_compare_and_delta(self):
        """
        compare_pts возвращает РАЗНОСТЬ (pts1 - pts2), а не -1/0/1, несмотря
        на название. Тест фиксирует фактический контракт: вызывающий код
        обязан проверять знак, а не сравнивать с ±1.
        """
        a, b = video_frame_to_pts(10), video_frame_to_pts(20)
        self.assertLess(compare_pts(a, b), 0)
        self.assertGreater(compare_pts(b, a), 0)
        self.assertEqual(compare_pts(a, a), 0)
        self.assertEqual(compare_pts(b, a), video_frame_to_pts(10))
        self.assertEqual(pts_delta(b, a), video_frame_to_pts(10))

    # --- таймкоды -------------------------------------------------------
    def test_timecode_plain_number(self):
        self.assertEqual(timecode_to_frame("1234"), 1234)

    def test_timecode_full_format(self):
        self.assertEqual(timecode_to_frame("00:00:00;00"), 0)
        self.assertEqual(timecode_to_frame("00:00:01;00", fps=25.0), 25)
        self.assertEqual(timecode_to_frame("00:01:00;00", fps=25.0), 25 * 60)
        self.assertEqual(timecode_to_frame("01:00:00;00", fps=25.0), 25 * 3600)

    def test_timecode_last_frame_of_second(self):
        self.assertEqual(timecode_to_frame("00:00:00;24", fps=25.0), 24)

    def test_timecode_short_ss_ff(self):
        """Формат 'SS;FF' — две части."""
        self.assertEqual(timecode_to_frame("05;10", fps=25.0), 5 * 25 + 10)

    @unittest.expectedFailure
    def test_timecode_mm_ss_ff(self):
        """
        Формат 'MM:SS;FF' из докстринга timecode_to_frame.

        ИЗВЕСТНЫЙ ДЕФЕКТ: ветка для трёх частей написана как
            h, m, s = 0, *map(int, parts)
        — справа получается 4 значения (0 + три разобранных), слева три
        имени, поэтому распаковка всегда падает с ValueError, который тут
        же перехватывается и превращается в "Некорректные числа в
        таймкоде". То есть заявленный формат не работает никогда.

        Помечен @expectedFailure: набор остаётся зелёным (иначе постоянно
        красный прогон приучает игнорировать результат), но дефект
        зафиксирован. После исправления — заменить в timecode_to_frame
        строку `h, m, s = 0, *map(int, parts)` на разбор трёх частей как
        (m, s, f) — тест сообщит "unexpected success", и декоратор нужно
        будет снять.
        """
        self.assertEqual(timecode_to_frame("01:30;00", fps=25.0), 90 * 25)

    def test_timecode_rejects_empty(self):
        with self.assertRaises(ValueError):
            timecode_to_frame("   ")

    def test_timecode_rejects_out_of_range(self):
        for bad in ("00:60:00;00", "24:00:00;00", "00:00:60;00", "00:00:00;25"):
            with self.assertRaises(ValueError, msg=f"должно отклоняться: {bad}"):
                timecode_to_frame(bad, fps=25.0)

    def test_timecode_rejects_garbage(self):
        for bad in ("abc", "1:2:3:4:5", "aa:bb;cc"):
            with self.assertRaises(ValueError, msg=f"должно отклоняться: {bad}"):
                timecode_to_frame(bad)


# ===========================================================================
# FrameRingBuffer
# ===========================================================================
class TestFrameRingBuffer(unittest.TestCase):

    def test_rejects_tiny_capacity(self):
        with self.assertRaises(ValueError):
            FrameRingBuffer(max_frames=1)

    def test_push_and_peek_order(self):
        buf = FrameRingBuffer(max_frames=4)
        for i in range(3):
            self.assertTrue(buf.try_push(make_frame(i), video_frame_to_pts(i)))
        self.assertEqual(buf.count, 3)
        pts, _frame = buf.peek_first()
        self.assertEqual(pts, video_frame_to_pts(0))
        buf.advance()
        self.assertEqual(buf.peek_first()[0], video_frame_to_pts(1))
        self.assertEqual(buf.count, 2)

    def test_try_push_refuses_when_full(self):
        buf = FrameRingBuffer(max_frames=2)
        self.assertTrue(buf.try_push(make_frame(1), 100))
        self.assertTrue(buf.try_push(make_frame(2), 200))
        self.assertFalse(buf.try_push(make_frame(3), 300),
                         "переполненный буфер обязан отказывать, а не терять кадр молча")
        self.assertEqual(buf.count, 2)

    def test_wraparound_keeps_order(self):
        """
        Ключевой тест кольцевой природы: после многих циклов записи/чтения
        порядок и содержимое не должны нарушаться. Здесь ломаются ошибки
        в арифметике индексов, невидимые на коротких сценариях.
        """
        cap = 4
        buf = FrameRingBuffer(max_frames=cap)
        expected_next = 0
        pushed = 0
        for i in range(cap * 10):
            if buf.try_push(make_frame(i), video_frame_to_pts(i)):
                pushed += 1
            if buf.count >= 2:
                pts, frame = buf.peek_first()
                self.assertEqual(pts, video_frame_to_pts(expected_next))
                self.assertEqual(int(frame[0, 0, 0]), expected_next % 256)
                buf.advance()
                expected_next += 1
        while buf.count:
            self.assertEqual(buf.peek_first()[0], video_frame_to_pts(expected_next))
            buf.advance()
            expected_next += 1
        self.assertEqual(expected_next, pushed, "прочитано столько же, сколько записано")

    def test_peek_all_matches_count(self):
        buf = FrameRingBuffer(max_frames=8)
        for i in range(5):
            buf.try_push(make_frame(i), video_frame_to_pts(i))
        buf.advance()
        entries = buf.peek_all()
        self.assertEqual(len(entries), buf.count)
        self.assertEqual([p for p, _ in entries],
                         [video_frame_to_pts(i) for i in range(1, 5)])

    def test_peek_all_is_non_destructive(self):
        buf = FrameRingBuffer(max_frames=4)
        buf.try_push(make_frame(1), 100)
        before = buf.count
        buf.peek_all()
        self.assertEqual(buf.count, before)

    def test_drop_until(self):
        buf = FrameRingBuffer(max_frames=10)
        for i in range(6):
            buf.try_push(make_frame(i), video_frame_to_pts(i))
        buf.drop_until(video_frame_to_pts(3))
        self.assertEqual(buf.count, 3)
        self.assertEqual(buf.peek_first()[0], video_frame_to_pts(3))

    def test_drop_until_can_empty_buffer(self):
        buf = FrameRingBuffer(max_frames=4)
        for i in range(3):
            buf.try_push(make_frame(i), video_frame_to_pts(i))
        buf.drop_until(video_frame_to_pts(100))
        self.assertEqual(buf.count, 0)
        self.assertIsNone(buf.peek_first())

    def test_drop_until_keeps_all_when_threshold_low(self):
        buf = FrameRingBuffer(max_frames=4)
        for i in range(3):
            buf.try_push(make_frame(i), video_frame_to_pts(i))
        buf.drop_until(0)
        self.assertEqual(buf.count, 3)

    def test_advance_on_empty_is_safe(self):
        buf = FrameRingBuffer(max_frames=4)
        buf.advance()          # не должно бросать
        self.assertEqual(buf.count, 0)

    def test_keep_last_stored_and_read(self):
        """keep_last сохраняется и читается, пока буфер не очищен."""
        buf = FrameRingBuffer(max_frames=4)
        buf.update_keep_last(make_frame(7), video_frame_to_pts(7))
        kept = buf.get_keep_last()
        self.assertIsNotNone(kept)
        self.assertEqual(int(kept[0, 0, 0]), 7)
        self.assertEqual(buf.get_keep_last_pts(), video_frame_to_pts(7))

    def test_clear_also_resets_keep_last(self):
        """
        Фиксирует ФАКТИЧЕСКОЕ поведение: clear() обнуляет и keep_last.

        Это не баг в текущих сценариях — clear() вызывается на неактивных
        буферах (free_buffer при seek, буферы при старте), а keep_last
        выставляется после. Но контракт стоит помнить: очистка АКТИВНОГО
        display-буфера приведёт к чёрному экрану на паузе, поскольку
        SyncManager.get_display_frame() при пустом буфере возвращает
        именно keep_last. Тест зафиксирует, если поведение изменится.
        """
        buf = FrameRingBuffer(max_frames=4)
        buf.update_keep_last(make_frame(7), video_frame_to_pts(7))
        buf.try_push(make_frame(1), video_frame_to_pts(1))
        buf.clear()
        self.assertEqual(buf.count, 0)
        self.assertIsNone(buf.get_keep_last())

    def test_free_slots_and_is_empty(self):
        buf = FrameRingBuffer(max_frames=3)
        self.assertTrue(buf.is_empty)
        self.assertEqual(buf.free_slots, 3)
        buf.try_push(make_frame(0), 0)
        self.assertFalse(buf.is_empty)
        self.assertEqual(buf.free_slots, 2)

    def test_non_contiguous_frame_accepted(self):
        """Срез numpy не C-contiguous; буфер обязан привести его сам."""
        big = np.zeros((8, 8, 3), dtype=np.uint8)
        view = big[::2, ::2]            # не contiguous
        self.assertFalse(view.flags["C_CONTIGUOUS"])
        buf = FrameRingBuffer(max_frames=2)
        self.assertTrue(buf.try_push(view, 100))
        _pts, stored = buf.peek_first()
        self.assertTrue(stored.flags["C_CONTIGUOUS"])

    def test_concurrent_push_and_advance(self):
        """
        Буфер объявлен потокобезопасным и реально используется из разных
        потоков (декодер пишет, рендер читает). Проверяем, что счётчик не
        разъезжается и структура не разрушается под конкурентной нагрузкой.
        """
        buf = FrameRingBuffer(max_frames=32)
        pushed = {"n": 0}
        consumed = {"n": 0}
        stop = threading.Event()
        errors = []

        def producer():
            try:
                for i in range(2000):
                    if buf.try_push(make_frame(i), video_frame_to_pts(i)):
                        pushed["n"] += 1
            except Exception as e:      # noqa: BLE001 - тест обязан сообщить причину
                errors.append(e)
            finally:
                stop.set()

        def consumer():
            try:
                while not stop.is_set() or buf.count:
                    if buf.peek_first() is not None:
                        buf.advance()
                        consumed["n"] += 1
            except Exception as e:      # noqa: BLE001
                errors.append(e)

        t1 = threading.Thread(target=producer)
        t2 = threading.Thread(target=consumer)
        t1.start(); t2.start()
        t1.join(timeout=10); t2.join(timeout=10)

        self.assertFalse(errors, f"исключения в потоках: {errors}")
        self.assertEqual(buf.count, 0)
        self.assertEqual(pushed["n"], consumed["n"],
                         "число записанных и прочитанных кадров должно совпадать")
        self.assertTrue(0 <= buf.count <= buf.max_frames)


# ===========================================================================
# StreamScheduler
# ===========================================================================
class FakeBuffer:
    """Минимальный двойник буфера: планировщику нужны только count/max_frames."""

    def __init__(self, count=0, max_frames=100):
        self.count = count
        self.max_frames = max_frames


class TestStreamScheduler(unittest.TestCase):

    def _sched(self, buf_count=0, max_frames=100):
        s = StreamScheduler()
        s.set_buffer(FakeBuffer(buf_count, max_frames))
        return s

    def test_no_chunks_before_configured(self):
        s = self._sched()
        self.assertIsNone(s.get_next_chunk(), "без окна планировщик ничего не выдаёт")

    def test_normal_sequence(self):
        s = self._sched()
        s.set_normal_mode(0, 5)
        got = []
        for _ in range(5):
            c = s.get_next_chunk()
            if c is None:
                break
            got.append(c)
            s._last_chunk_ts = 0.0     # снимаем ограничение темпа для теста
        self.assertEqual(got, [0, 1, 2, 3, 4])

    def test_exhausted_returns_none(self):
        s = self._sched()
        s.set_normal_mode(0, 2)
        for _ in range(2):
            s.get_next_chunk()
            s._last_chunk_ts = 0.0
        self.assertIsNone(s.get_next_chunk())

    def test_start_offset_respected(self):
        s = self._sched()
        s.set_normal_mode(3, 6)
        self.assertEqual(s.get_next_chunk(), 3)

    def test_hysteresis_stops_at_high_watermark(self):
        """Заполненность >= 95% — загрузка обязана остановиться."""
        s = self._sched(buf_count=96, max_frames=100)
        s.set_normal_mode(0, 10)
        self.assertIsNone(s.get_next_chunk())

    def test_hysteresis_resumes_only_below_low_watermark(self):
        """
        Гистерезис: после остановки на 95% загрузка возобновляется не
        сразу, а лишь при падении до 60%. Между этими порогами состояние
        сохраняется — именно это предотвращает дребезг.
        """
        buf = FakeBuffer(96, 100)
        s = StreamScheduler()
        s.set_buffer(buf)
        s.set_normal_mode(0, 10)
        self.assertIsNone(s.get_next_chunk())      # остановились на 96%

        buf.count = 70                              # ниже 95%, но выше 60%
        self.assertIsNone(s.get_next_chunk(),
                          "в зоне гистерезиса загрузка ещё не должна возобновляться")

        buf.count = 55                              # ниже 60%
        s._last_chunk_ts = 0.0
        self.assertIsNotNone(s.get_next_chunk(), "ниже нижнего порога загрузка идёт")

    def test_mark_chunk_failed_allows_retry(self):
        """Неудачный чанк обязан выдаваться повторно, иначе в видео дыра."""
        s = self._sched()
        s.set_normal_mode(0, 3)
        first = s.get_next_chunk()
        s._last_chunk_ts = 0.0
        s.mark_chunk_failed(first)
        s._current_chunk = first                   # планировщик вернулся к нему
        self.assertEqual(s.get_next_chunk(), first)

    def test_seek_mode_prioritises_target(self):
        s = self._sched()
        s.set_seek_mode(7, 20)
        self.assertEqual(s.get_next_chunk(), 7, "первым читается целевой чанк")

    def test_seek_mode_then_neighbours(self):
        s = self._sched()
        s.set_seek_mode(7, 20)
        got = []
        for _ in range(4):
            c = s.get_next_chunk()
            s._last_chunk_ts = 0.0
            if c is None:
                break
            got.append(c)
        self.assertEqual(got[0], 7)
        self.assertTrue(set(got[1:]).issubset({5, 6, 8, 9}),
                        f"после цели читаются соседние чанки, получено {got}")

    def test_pause_mode_still_prefetches(self):
        """На паузе буфер должен продолжать наполняться."""
        s = self._sched()
        s.set_pause_mode(0, 5)
        self.assertIsNotNone(s.get_next_chunk())

    def test_fast_forward_uses_stride(self):
        s = self._sched()
        s.set_fast_forward_mode(direction=1, speed=4.0,
                                current_local_chunk=0, total_local_chunks=50)
        first = s.get_next_chunk()
        s._last_chunk_ts = 0.0
        second = s.get_next_chunk()
        self.assertIsNotNone(second)
        self.assertGreater(second - first, 1,
                           "при ускоренной перемотке чанки читаются с пропуском")

    def test_extend_total_chunks_keeps_position(self):
        """
        Рост live-файла расширяет границу, но НЕ сбрасывает позицию —
        иначе воспроизведение отматывалось бы назад на каждое обновление
        индекса (раз в 10 секунд).
        """
        s = self._sched()
        s.set_normal_mode(0, 5)
        s.get_next_chunk()
        s._last_chunk_ts = 0.0
        pos_before = s._current_chunk
        loaded_before = s.loaded_count

        s.extend_total_chunks(50)
        self.assertEqual(s._current_chunk, pos_before, "позиция не должна сбрасываться")
        self.assertEqual(s.loaded_count, loaded_before, "загруженные чанки сохраняются")
        self.assertEqual(s._total_chunks, 50)

    def test_extend_never_shrinks(self):
        s = self._sched()
        s.set_normal_mode(0, 40)
        s.extend_total_chunks(10)
        self.assertEqual(s._total_chunks, 40, "граница не должна уменьшаться")

    def test_shift_loaded_remaps_indices(self):
        """
        Сдвиг скользящего окна: локальные индексы пересчитываются на
        chunk_shift, ушедшие за левую границу отбрасываются, режим и темп
        не сбрасываются.
        """
        s = self._sched()
        s.set_normal_mode(0, 100)
        for _ in range(5):
            s.get_next_chunk()
            s._last_chunk_ts = 0.0
        self.assertEqual(sorted(s._loaded_chunks), [0, 1, 2, 3, 4])
        mode_before = s._mode

        s.shift_loaded(chunk_shift=2, new_total_local_chunks=100)

        self.assertEqual(sorted(s._loaded_chunks), [0, 1, 2],
                         "чанки 0 и 1 ушли за границу окна и должны выпасть")
        self.assertEqual(s._current_chunk, 3, "позиция сдвинута на chunk_shift")
        self.assertEqual(s._mode, mode_before, "режим не меняется при сдвиге окна")

    def test_shift_loaded_zero_is_noop_for_position(self):
        s = self._sched()
        s.set_normal_mode(4, 20)
        s.shift_loaded(chunk_shift=0, new_total_local_chunks=30)
        self.assertEqual(s._current_chunk, 4)
        self.assertEqual(s._total_chunks, 30)

    def test_reset_clears_everything(self):
        s = self._sched()
        s.set_normal_mode(0, 10)
        s.get_next_chunk()
        s.reset()
        self.assertEqual(s.loaded_count, 0)
        self.assertEqual(s._total_chunks, 0)
        self.assertEqual(s._mode, PlaybackMode.NORMAL)


if __name__ == "__main__":
    unittest.main(verbosity=2)
