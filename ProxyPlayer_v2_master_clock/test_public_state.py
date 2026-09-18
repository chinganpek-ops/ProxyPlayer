#!/usr/bin/env python3
"""
test_public_state.py – тесты публичных интерфейсов состояния.

ЭТАП 1.2 первой итерации рефакторинга.

ЗАЧЕМ ЭТИ ТЕСТЫ
Публичные методы состояния (get_state, get_playback_state,
get_track_state и другие) добавлены, чтобы телеметрия, инструменты замера
и соседние модули перестали читать приватные поля напрямую. Такие связи
не проверяются контрактным тестом и ломаются молча при переименовании
поля — этот класс ошибок оказался самым дорогим в отладке проекта.

Но сам по себе новый метод проблему не решает: он должен ОТДАВАТЬ РОВНО
ТО, что раньше читалось напрямую. Если состав полей окажется неполным,
потребитель вернётся к приватным полям, и работа будет напрасной.

Поэтому тесты проверяют три вещи:

  1. Метод существует и вызывается без исключений.
  2. Возвращает ожидаемый НАБОР полей — тот, что реально нужен
     потребителям (телеметрии, measure_playback, soak_test).
  3. Значения соответствуют внутреннему состоянию объекта — то есть
     метод не отдаёт заглушку и не расходится с реальностью.

Отдельная группа тестов следит за отсутствием ПОВТОРНЫХ определений
методов. Это не теоретическая опасность: при добавлении публичных
методов дубли реально возникли в восьми модулях сразу. В Python
побеждает последнее определение, поэтому первое становится мёртвым
кодом, который выглядит рабочим — и правка в нём ни на что не влияет.

ЗАПУСК
    python test_public_state.py
    pytest test_public_state.py -v

Тесты работают на заглушках: без файлов, сети, звуковой карты и PyAV.
"""

import ast
import sys
import threading
import types
import unittest
from collections import Counter, deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Заглушки тяжёлых зависимостей
# ---------------------------------------------------------------------------
def _stub_modules():
    """
    Ставит заглушки ТОЛЬКО на то, что мешает: внешние библиотеки и
    модули, тянущие их при импорте.

    Важно: модули, которые тест проверяет (pipeline.stream_scheduler,
    core.master_clock) и их лёгкие зависимости (config.timebase)
    заглушками НЕ подменяются. Ранняя версия клала пустой модуль в
    sys.modules для pipeline.stream_scheduler — и настоящий модуль
    становился недоступен: импорт находил заглушку, падал с ImportError,
    а запасной путь `from stream_scheduler import ...` работал только при
    плоской раскладке файлов. В проекте с пакетами тест не запускался.
    """
    import importlib
    import logging

    def stub(name):
        """Безусловная заглушка — для того, чего в среде может не быть."""
        m = sys.modules.get(name)
        if m is None:
            m = types.ModuleType(name)
            sys.modules[name] = m
        return m

    def stub_if_missing(name):
        """Заглушка только если настоящий модуль не импортируется."""
        if name in sys.modules:
            return sys.modules[name]
        try:
            return importlib.import_module(name)
        except Exception:
            return stub(name)

    # Внешние зависимости и модули, которые их тянут: подменяем всегда,
    # иначе тест нельзя прогнать без звуковой карты и PyAV.
    stub("sounddevice")
    for name in ("buffer", "buffer.frame_buffer", "decode", "decode.decoder",
                 "decode.audio_decoder", "index", "index.lazy_index",
                 "index.moov_builder", "index.idx_cache", "file_io",
                 "file_io.win_sequential_reader", "utils", "utils.utils",
                 "utils.sync_logger"):
        stub(name)

    # Лёгкие зависимости: пробуем настоящие, заглушка только как запасной
    # вариант. Настоящая timebase точнее любой имитации.
    for name in ("config", "config.timebase"):
        stub_if_missing(name)

    # Дозаполняем только отсутствующее: если настоящая timebase
    # загрузилась, её значения не трогаем.
    tb = sys.modules["config.timebase"]
    for attr, value in (("FRAMES_PER_CHUNK", 12), ("SAMPLES_PER_CHUNK", 23040),
                        ("SAMPLES_PER_VIDEO_FRAME", 1920),
                        ("AUDIO_SAMPLE_RATE", 48000)):
        if not hasattr(tb, attr):
            setattr(tb, attr, value)
    if not hasattr(tb, "video_frame_to_pts"):
        tb.video_frame_to_pts = lambda f: f * 1920
    if not hasattr(tb, "pts_to_video_frame"):
        tb.pts_to_video_frame = lambda p: p // 1920

    sys.modules["buffer.frame_buffer"].FrameRingBuffer = FakeBuffer
    sys.modules["decode.decoder"].Decoder = object
    sys.modules["decode.audio_decoder"].AudioDecoder = object
    sys.modules["index.lazy_index"].LazyIndex = object
    sys.modules["index.lazy_index"].IndexWindow = object
    sys.modules["index.moov_builder"].SEGMENT_SIZE = 4294967296
    sys.modules["index.moov_builder"]._abs_offset = lambda r: 0
    sys.modules["file_io.win_sequential_reader"].WinSequentialReader = object
    sys.modules["utils.utils"].get_real_size = lambda p: 0
    sys.modules["utils.sync_logger"].sync_monitor_logger = logging.getLogger("stub")

    class AC:
        def __init__(self, *a, **k):
            pass

        def get_fast_forward_stride(self, speed):
            return max(1, int(speed))

    mod("pipeline.adaptive_chunk").AdaptiveChunkStrategy = AC

    import ctypes
    if not hasattr(ctypes, "windll"):
        class _DLL:
            def __getattr__(self, name):
                f = lambda *a: 0
                f.argtypes = None
                f.restype = None
                return f
        ctypes.windll = types.SimpleNamespace(kernel32=_DLL(), user32=_DLL())


class FakeBuffer:
    """Двойник FrameRingBuffer: нужны только count/max_frames и peek."""

    def __init__(self, max_frames=360):
        self.count = 0
        self.max_frames = max_frames
        self._keep = None

    def clear(self):
        self.count = 0

    def peek_first(self):
        return None

    def update_keep_last(self, frame, pts):
        self._keep = (pts, frame)

    def get_keep_last(self):
        return self._keep[1] if self._keep else None


_stub_modules()

def _import_any(*paths):
    """
    Импортирует модуль, перебирая возможные пути.

    Проект встречается в двух раскладках: пакетами (pipeline/, core/) и
    плоско рядом с main.py. Тест должен работать в обеих, поэтому путь не
    зашивается, а подбирается.
    """
    import importlib
    errors = []
    for path in paths:
        try:
            return importlib.import_module(path)
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    raise ImportError("не удалось импортировать модуль. Попытки:\n  "
                      + "\n  ".join(errors))


_sched_mod = _import_any("pipeline.stream_scheduler", "stream_scheduler")
StreamScheduler = _sched_mod.StreamScheduler
PlaybackMode = _sched_mod.PlaybackMode


# ===========================================================================
# Отсутствие повторных определений
# ===========================================================================
class TestNoDuplicateMethods(unittest.TestCase):
    """
    Повторное определение метода в одном классе — не стилистическая
    придирка, а реальный дефект: побеждает последнее, первое становится
    мёртвым кодом, который выглядит рабочим. При добавлении публичных
    методов состояния такие дубли возникли сразу в восьми модулях.
    """

    MODULES = [
        "stream_scheduler.py", "chunk_pipeline.py", "master_clock.py",
        "playback_engine.py", "lazy_index.py", "seek_engine.py",
        "stream_controller.py", "win_sequential_reader.py",
        "sync_manager.py", "idx_cache.py", "moov_builder.py",
    ]

    @staticmethod
    def _find_file(name):
        """Ищет модуль и в корне, и в подкаталогах проекта."""
        direct = ROOT / name
        if direct.exists():
            return direct
        found = list(ROOT.rglob(name))
        return found[0] if found else None

    def test_no_duplicate_methods(self):
        problems = []
        checked = 0
        for name in self.MODULES:
            path = self._find_file(name)
            if path is None:
                continue
            checked += 1
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
                counts = Counter(
                    i.name for i in cls.body if isinstance(i, ast.FunctionDef))
                for method, cnt in counts.items():
                    if cnt > 1:
                        problems.append(f"{name}: {cls.name}.{method}() определён {cnt} раза")
        self.assertGreater(checked, 0, "не найден ни один модуль проекта")
        self.assertFalse(problems, "Повторные определения:\n  " + "\n  ".join(problems))

    def test_no_duplicate_module_functions(self):
        """То же для функций уровня модуля."""
        problems = []
        for name in self.MODULES:
            path = self._find_file(name)
            if path is None:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            counts = Counter(
                n.name for n in tree.body if isinstance(n, ast.FunctionDef))
            for fn, cnt in counts.items():
                if cnt > 1:
                    problems.append(f"{name}: {fn}() определена {cnt} раза")
        self.assertFalse(problems, "Повторные определения:\n  " + "\n  ".join(problems))


# ===========================================================================
# StreamScheduler
# ===========================================================================
class TestSchedulerPublicState(unittest.TestCase):

    def _sched(self, count=0, max_frames=360):
        s = StreamScheduler()
        buf = FakeBuffer(max_frames)
        buf.count = count
        s.set_buffer(buf)
        return s

    def test_get_state_has_required_fields(self):
        """
        Набор полей продиктован тем, что реально читали потребители:
        телеметрия (режим, позиция, загруженные), measure_playback
        (loading_allowed), тесты (total_chunks).
        """
        s = self._sched()
        s.set_normal_mode(5, 100)
        st = s.get_state()
        for field in ("mode", "current_chunk", "total_chunks", "loaded_count",
                      "loading_allowed", "target_chunk", "direction", "speed"):
            self.assertIn(field, st, f"в get_state() нет поля {field}")

    def test_get_state_matches_internals(self):
        """Метод должен отражать реальное состояние, а не отдавать заглушку."""
        s = self._sched()
        s.set_normal_mode(7, 42)
        st = s.get_state()
        self.assertEqual(st["current_chunk"], 7)
        self.assertEqual(st["total_chunks"], 42)
        self.assertEqual(st["mode"], "NORMAL")

    def test_get_state_reflects_mode_change(self):
        s = self._sched()
        s.set_normal_mode(0, 100)
        self.assertEqual(s.get_state()["mode"], "NORMAL")
        s.set_fast_forward(direction=-1, speed=4.0,
                           current_local_chunk=50, total_local_chunks=100)
        st = s.get_state()
        self.assertEqual(st["mode"], "FAST_FORWARD")
        self.assertEqual(st["direction"], -1)
        self.assertEqual(st["speed"], 4.0)

    def test_get_state_counts_loaded(self):
        s = self._sched()
        s.set_normal_mode(0, 100)
        for _ in range(3):
            s.reset_rate_limit()
            s.get_next_chunk()
        self.assertEqual(s.get_state()["loaded_count"], 3)

    def test_set_fast_forward_equals_legacy_call(self):
        """Публичный псевдоним обязан делать ровно то же, что старый метод."""
        a = self._sched()
        b = self._sched()
        a.set_fast_forward(1, 8.0, 10, 200)
        b.set_fast_forward_mode(1, 8.0, 10, 200)
        self.assertEqual(a.get_state(), b.get_state())

    def test_reset_rate_limit_allows_immediate_chunk(self):
        """
        Заменяет запись в _last_chunk_ts из тестов. Без снятия ограничения
        второй чанк подряд отклоняется темпом.
        """
        s = self._sched()
        s.set_normal_mode(0, 100)
        first = s.get_next_chunk()
        self.assertIsNotNone(first)
        self.assertIsNone(s.get_next_chunk(), "темп должен отклонить второй подряд")
        s.reset_rate_limit()
        self.assertIsNotNone(s.get_next_chunk(), "после сброса чанк должен выдаваться")

    def test_properties_match_state(self):
        s = self._sched()
        s.set_normal_mode(9, 77)
        st = s.get_state()
        self.assertEqual(s.mode_name, st["mode"])
        self.assertEqual(s.total_chunks, st["total_chunks"])
        self.assertEqual(s.current_chunk, st["current_chunk"])

    def test_get_state_is_read_only(self):
        """Снимок не должен менять поведение планировщика."""
        s = self._sched()
        s.set_normal_mode(0, 100)
        before = s.get_state()
        for _ in range(5):
            s.get_state()
        after = s.get_state()
        self.assertEqual(before, after)


# ===========================================================================
# MasterClock
# ===========================================================================
def make_clock():
    MasterClock = _import_any("core.master_clock", "master_clock").MasterClock
    mc = MasterClock.__new__(MasterClock)
    mc.sample_rate = 48000
    mc.buffer_size = 1024
    mc.max_audio_queue_samples = 691200
    mc._samples_played = 0
    mc._clock_lock = threading.Lock()
    mc._queue_lock = threading.Lock()
    mc._queue2, mc._queue3 = deque(), deque()
    mc._track2_enabled = mc._track3_enabled = True
    mc._muted = False
    mc.audio_delay = 0.0
    mc._underruns = 0
    mc._max_queue_len = 0
    mc._stream = None
    mc._active = False
    mc._device_available = True
    # Счётчики качества звука: заполняются в push_audio(). Раньше эти
    # измерения делала телеметрия, подменяя push_audio снаружи.
    mc._track_stats = {
        t: {"blocks": 0, "samples": 0, "dtype": None, "peak": 0.0,
            "clipped": 0, "non_finite": 0, "rms_sum": 0.0, "rms_n": 0,
            "min_block": None, "max_block": 0}
        for t in (2, 3)
    }
    return mc


class TestMasterClockPublicState(unittest.TestCase):

    def test_track_state_fields(self):
        mc = make_clock()
        mc.push_audio(2, np.zeros(2048, dtype=np.float32), pts=0)
        st = mc.get_track_state(2)
        for field in ("track", "enabled", "blocks", "samples"):
            self.assertIn(field, st, f"в get_track_state() нет поля {field}")

    def test_track_state_counts_samples_not_tuples(self):
        """
        Очередь хранит пары (pts, samples). Подсчёт обязан брать длину
        массива, а не длину кортежа: измерительный скрипт уже допускал
        эту ошибку и показывал вдвое большее число блоков вместо сэмплов.
        """
        mc = make_clock()
        for i in range(3):
            mc.push_audio(2, np.zeros(2048, dtype=np.float32), pts=i * 2048)
        st = mc.get_track_state(2)
        self.assertEqual(st["blocks"], 3)
        self.assertEqual(st["samples"], 3 * 2048)

    def test_track_state_is_per_track(self):
        """Раздельный подсчёт — единственный способ увидеть расхождение дорожек."""
        mc = make_clock()
        mc.push_audio(2, np.zeros(2048, dtype=np.float32), pts=0)
        mc.push_audio(2, np.zeros(2048, dtype=np.float32), pts=2048)
        mc.push_audio(3, np.zeros(2048, dtype=np.float32), pts=0)
        self.assertEqual(mc.get_track_state(2)["samples"], 4096)
        self.assertEqual(mc.get_track_state(3)["samples"], 2048)

    def test_track_queue_samples_matches_state(self):
        mc = make_clock()
        mc.push_audio(3, np.zeros(1024, dtype=np.float32), pts=0)
        self.assertEqual(mc.get_track_queue_samples(3),
                         mc.get_track_state(3)["samples"])

    def test_total_equals_sum_of_tracks(self):
        mc = make_clock()
        mc.push_audio(2, np.zeros(2048, dtype=np.float32), pts=0)
        mc.push_audio(3, np.zeros(1024, dtype=np.float32), pts=0)
        self.assertEqual(mc.get_audio_queue_samples(),
                         mc.get_track_state(2)["samples"]
                         + mc.get_track_state(3)["samples"])

    def test_quality_detects_out_of_range(self):
        """
        Амплитуда вне -1..+1 означает, что декодер отдаёт целочисленные
        сэмплы, а поток ожидает float32 — слышно как громкий треск.
        Измерение перенесено внутрь MasterClock, потому что снаружи его
        можно было получить только подменой push_audio.
        """
        mc = make_clock()
        mc.push_audio(3, (np.ones(2048, dtype=np.float32) * 8000.0), pts=0)
        q = mc.get_track_quality(3)
        self.assertGreater(q["peak"], 1.5)
        self.assertEqual(q["clipped"], 2048)

    def test_quality_detects_non_finite(self):
        """NaN/inf от повреждённых пакетов — в звуке это щелчки."""
        mc = make_clock()
        mc.push_audio(2, np.full(512, np.nan, dtype=np.float32), pts=0)
        self.assertEqual(mc.get_track_quality(2)["non_finite"], 512)

    def test_diagnostics_reports_drift(self):
        """Расхождение подачи — главный признак расползания дорожек."""
        mc = make_clock()
        for i in range(4):
            mc.push_audio(2, np.zeros(2048, dtype=np.float32), pts=i * 2048)
        mc.push_audio(3, np.zeros(2048, dtype=np.float32), pts=0)
        d = mc.get_audio_diagnostics()
        self.assertEqual(d["pushed_drift_samples"], 3 * 2048)
        self.assertIn("queue_drift_ms", d)

    def test_diagnostics_has_both_tracks(self):
        mc = make_clock()
        d = mc.get_audio_diagnostics()
        self.assertIn("2", d["tracks"])
        self.assertIn("3", d["tracks"])

    def test_disabled_track_reported(self):
        mc = make_clock()
        mc.set_track_enabled(3, False)
        self.assertFalse(mc.get_track_state(3)["enabled"])


# ===========================================================================
# Полнота набора: методы существуют у всех классов
# ===========================================================================
class TestPublicApiPresence(unittest.TestCase):
    """
    Проверяет наличие методов статически, по исходникам. Импортировать
    все модули нельзя — часть требует PyQt5 и PyAV, которых может не быть
    на машине, где гоняются тесты.
    """

    EXPECTED = {
        "stream_scheduler.py": {
            "StreamScheduler": ["get_state", "set_fast_forward", "reset_rate_limit",
                                "get_loaded_chunks"]},
        "chunk_pipeline.py": {
            "ChunkPipeline": ["get_scheduler_state", "get_queue_sizes",
                              "get_stage_status", "get_reader_state",
                              "set_fast_forward", "set_normal_position"]},
        "master_clock.py": {
            "MasterClock": ["get_track_state", "get_track_queue_samples",
                            "get_track_quality", "get_audio_diagnostics",
                            "get_underruns"]},
        "playback_engine.py": {
            "PlaybackEngine": ["get_playback_state", "get_buffer_state"]},
        "lazy_index.py": {
            "LazyIndex": ["get_index_state", "get_memory_usage"]},
        "seek_engine.py": {
            "SeekEngine": ["get_workers_state", "get_state"]},
        "stream_controller.py": {
            "StreamController": ["is_ready", "wait_ready", "get_init_error",
                                 "get_components_state", "get_display_buffer"]},
        "win_sequential_reader.py": {
            "WinSequentialReader": ["get_state"]},
    }

    def test_expected_methods_present(self):
        missing = []
        checked = 0
        for filename, classes in self.EXPECTED.items():
            path = TestNoDuplicateMethods._find_file(filename)
            if path is None:
                continue
            checked += 1
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            found = {}
            for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
                found[cls.name] = {i.name for i in cls.body
                                   if isinstance(i, ast.FunctionDef)}
            for cls_name, methods in classes.items():
                have = found.get(cls_name, set())
                for m in methods:
                    if m not in have:
                        missing.append(f"{filename}: {cls_name}.{m}()")
        self.assertGreater(checked, 0, "не найден ни один модуль проекта")
        self.assertFalse(missing, "Отсутствуют публичные методы:\n  "
                                  + "\n  ".join(missing))


if __name__ == "__main__":
    unittest.main(verbosity=2)
