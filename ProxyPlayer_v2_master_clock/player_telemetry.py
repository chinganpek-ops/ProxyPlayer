#!/usr/bin/env python3
"""
player_telemetry.py – телеметрия ProxyPlayer, работающая ВНУТРИ процесса.

В отличие от внешнего монитора, который парсил текстовые логи, этот модуль
подключается к живым объектам и снимает их фактическое состояние. Никакого
разбора строк: буферы, счётчики и состояния читаются напрямую.

Что собирается:

1. Состояние компонентов (через StreamController):
   - MasterClock: underruns, сэмплы в очередях, дорожки, mute
   - FrameRingBuffer: заполненность видеобуфера (count/max)
   - StreamScheduler: режим, текущий чанк, загруженные чанки
   - ChunkPipeline: длины очередей raw/video/audio, границы активного окна
   - SeekEngine: состояния воркеров (free/busy/stuck), поколение запросов
   - LazyIndex: total_frames, mdat_end, границы окна
   - PlaybackEngine: позиция, playing/paused, состояние перехода окна
   - WinSequentialReader: размер файла, позиция, статистика чтений

2. Потоки процесса: количество по группам имён (SeekWorker, ReaderStage,
   DemuxerStage и т.д.) — рост числа SeekWorker'ов означает, что воркеры
   зависают и выводятся из пула.

3. Метрики процесса (psutil, если установлен): CPU, RSS, потоки, HANDLE.

4. Логи — не парсингом, а перехватом: на root-логгер вешается handler,
   который получает готовые LogRecord и агрегирует их по (логгер, уровень).
   Тексты не разбираются, кроме короткого списка маркеров-счётчиков.

Выход:
   <outdir>/telemetry_<tag>_<pid>.jsonl   — снимок на каждый интервал
   <outdir>/telemetry_<tag>_<pid>.txt     — итоговый отчёт при завершении

Подключение (main.py):

    from player_telemetry import install_telemetry, attach_controller

    # сразу после setup_logging(), в любом режиме:
    install_telemetry(log_dir, tag="managed")

    # после создания StreamController (там, где сейчас attach_to_controller):
    attach_controller(widget.player)

Модуль намеренно не импортирует ничего из проекта: все обращения к
объектам плеера — через getattr с защитой, поэтому переименование
внутренних полей телеметрию не ломает (поле просто пропадёт из снимка),
и модуль безопасно работает в процессе IndexService, где плеера нет.
"""

import atexit
import json
import logging
import os
import platform
import struct
import sys
import threading
import time
import traceback
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None


DEFAULT_INTERVAL_SEC = 5.0

# Маркеры, которые считаем по тексту сообщения. Это единственное место, где
# телеметрия смотрит на текст, — всё остальное берётся из живых объектов.
# Нужны для событий, не имеющих счётчика в самих компонентах.
MESSAGE_MARKERS = {
    "short_read": "Недочитано",
    "worker_retired": "выведен из пула",
    "watchdog": "watchdog",
    "chunk_failed": "помечаю как неудачный",
    "queue_full": "переполнена",
    "window_slid": "окно сдвинуто",
    "mdat_refreshed": "mdat_end обновлён",
    "filesize_refreshed": "Размер файла обновлён",
    "idx_rescan": "idx просканирован",
    "seek_cancelled": "Seek отменён",
}

# Группы потоков, за которыми следим отдельно.
THREAD_GROUPS = ("SeekWorker", "SeekCoordinator", "ReaderStage", "DemuxerStage",
                 "VideoDecoderStage", "AudioDecoderStage", "Thread-")


def _safe(fn, default=None):
    """Вызывает fn() и возвращает default при любой ошибке."""
    try:
        return fn()
    except Exception:
        return default


class _TelemetryLogHandler(logging.Handler):
    """
    Handler на root-логгере: агрегирует LogRecord'ы вместо записи текста.

    Работает с готовыми объектами записей (record.levelname, record.name,
    record.threadName), поэтому не зависит от формата строк и не ломается
    при изменении форматтера. Тяжёлого форматирования не делает — вызов
    record.getMessage() происходит только для WARNING и выше.
    """

    def __init__(self, error_writer=None):
        """
        error_writer — callable(text), вызывается при каждой записи уровня
        ERROR и выше. Через него полный стектрейс уходит на диск НЕМЕДЛЕННО,
        а не при закрытии плеера: если процесс завершится аварийно, отчёт
        не успеет записаться, а этот файл уже будет на диске.
        """
        super().__init__(level=logging.DEBUG)
        self.error_writer = error_writer
        self.lock_ = threading.Lock()
        self.by_level = Counter()
        self.by_logger_level = Counter()
        self.markers = Counter()
        self.recent_problems = deque(maxlen=100)
        # Счётчики за текущий интервал (сбрасываются при каждом снимке)
        self.interval_levels = Counter()
        self.interval_markers = Counter()

    def emit(self, record):
        try:
            level = record.levelname
            with self.lock_:
                self.by_level[level] += 1
                self.interval_levels[level] += 1
                self.by_logger_level[(record.name, level)] += 1

            if record.levelno >= logging.WARNING:
                msg = record.getMessage()
                with self.lock_:
                    for key, marker in MESSAGE_MARKERS.items():
                        if marker in msg:
                            self.markers[key] += 1
                            self.interval_markers[key] += 1
                    if record.levelno >= logging.ERROR:
                        self.recent_problems.append({
                            "t": datetime.now().strftime("%H:%M:%S"),
                            "logger": record.name,
                            "level": level,
                            "thread": record.threadName,
                            "msg": msg[:300],
                        })
                        if self.error_writer is not None:
                            # Полный стектрейс — целиком, без обрезки.
                            tb = ""
                            if record.exc_info:
                                tb = "".join(traceback.format_exception(*record.exc_info))
                            self.error_writer(
                                record=record, level=level, msg=msg, tb=tb)
            else:
                # INFO/DEBUG: маркеры тоже нужны (например, "окно сдвинуто"
                # логируется на INFO), но getMessage() дорог — вызываем его
                # только если в сыром шаблоне есть хоть один маркер.
                raw = str(record.msg)
                for key, marker in MESSAGE_MARKERS.items():
                    if marker in raw:
                        with self.lock_:
                            self.markers[key] += 1
                            self.interval_markers[key] += 1
                        break
        except Exception:
            pass  # телеметрия не должна ломать логирование

    def take_interval(self):
        with self.lock_:
            levels = dict(self.interval_levels)
            markers = dict(self.interval_markers)
            self.interval_levels.clear()
            self.interval_markers.clear()
        return levels, markers

    def totals(self):
        with self.lock_:
            return dict(self.by_level), dict(self.markers), list(self.recent_problems)


class Telemetry:
    """Сборщик. Один экземпляр на процесс (см. install_telemetry)."""

    def __init__(self, outdir: Path, tag: str, interval: float):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.tag = tag
        self.interval = interval
        self.pid = os.getpid()

        self.jsonl_path = self.outdir / f"telemetry_{tag}_{self.pid}.jsonl"
        self.report_path = self.outdir / f"telemetry_{tag}_{self.pid}.txt"
        # Отдельный файл только под ошибки: полный стектрейс + состояние
        # компонентов на момент сбоя. Пишется сразу, построчно (buffering=1),
        # поэтому переживает даже аварийное завершение процесса.
        self.errors_path = self.outdir / f"telemetry_{tag}_{self.pid}_ERRORS.log"

        self._controller = None
        self._stop = threading.Event()
        self._started_at = time.monotonic()
        self._samples = 0

        # Пиковые/экстремальные значения для итогового отчёта
        self._peaks = defaultdict(float)
        self._mins = {}
        self._frame_info = {}
        self._files_info = {}
        self._audio_summary = {}

        self._last_snapshot = {}
        try:
            self._err_fh = open(self.errors_path, "a", encoding="utf-8", buffering=1)
        except Exception:
            self._err_fh = None

        self.handler = _TelemetryLogHandler(error_writer=self._write_error)
        logging.getLogger().addHandler(self.handler)

        self._proc = psutil.Process(self.pid) if psutil else None

        self._fh = open(self.jsonl_path, "a", encoding="utf-8", buffering=1)
        # Первой строкой — статическое окружение (разрядность процесса и т.п.),
        # чтобы отчёт был самодостаточным при пересылке.
        self._env = self._env_info()
        try:
            self._fh.write(json.dumps({"record": "env", **self._env}, ensure_ascii=False) + "\n")
        except Exception:
            pass
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="TelemetrySampler")
        self._thread.start()
        atexit.register(self.stop)

        logging.getLogger("Telemetry").info(
            "Телеметрия включена: %s (интервал %.1f с); ошибки: %s",
            self.jsonl_path, interval, self.errors_path.name)

    # ------------------------------------------------------------------
    def _write_error(self, record, level, msg, tb):
        """
        Записывает ошибку в отдельный файл: сообщение, полный стектрейс и
        СОСТОЯНИЕ КОМПОНЕНТОВ на момент сбоя.

        Состояние берётся из последнего снимка сборщика, а не собирается
        здесь заново: этот метод вызывается из произвольного потока прямо
        из logging-обработчика, и обращение к живым объектам (например,
        pipeline.get_window_snapshot(), берущий _window_lock) могло бы
        привести к взаимной блокировке, если исключение возникло как раз
        под этим локом. Поэтому — заведомо безопасный, пусть и слегка
        устаревший (до интервала опроса) срез.
        """
        if self._err_fh is None:
            return
        try:
            parts = [
                "=" * 70,
                f"{datetime.now().isoformat(timespec='seconds')}  [{level}]  "
                f"{record.name}  (поток: {record.threadName})",
                "=" * 70,
                msg,
            ]
            if tb:
                parts.append("")
                parts.append(tb.rstrip())

            snap = self._last_snapshot
            if snap:
                age = round(time.monotonic() - snap.get("_taken_at", 0), 1)
                parts.append("")
                parts.append(f"-- Состояние компонентов (снимок {age} с назад) --")
                comp = snap.get("components", {})
                for key in ("controller", "playback", "window", "scheduler",
                            "display_buffer", "seek_engine", "lazy_index",
                            "pipeline", "reader", "master_clock", "files",
                            "index_memory"):
                    if key in comp:
                        parts.append(f"  {key}: "
                                     f"{json.dumps(comp[key], ensure_ascii=False)}")
                for key in ("process", "threads"):
                    if key in snap:
                        parts.append(f"  {key}: "
                                     f"{json.dumps(snap[key], ensure_ascii=False)}")
            parts.append("")
            self._err_fh.write("\n".join(parts) + "\n")
        except Exception:
            pass  # телеметрия не должна ломать логирование

    # ------------------------------------------------------------------
    def attach(self, controller):
        """Привязывает StreamController — с этого момента снимается состояние компонентов."""
        self._controller = controller
        logging.getLogger("Telemetry").info("Телеметрия: контроллер подключён")

    def stop(self):
        if self._stop.is_set():
            return
        self._stop.set()
        try:
            self._thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            self._write_report()
        except Exception:
            logging.getLogger("Telemetry").exception("Не удалось записать отчёт телеметрии")
        try:
            self._fh.close()
        except Exception:
            pass
        try:
            if self._err_fh:
                self._err_fh.close()
        except Exception:
            pass
        try:
            logging.getLogger().removeHandler(self.handler)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _loop(self):
        # Первый вызов cpu_percent задаёт точку отсчёта и всегда возвращает 0.
        if self._proc:
            _safe(lambda: self._proc.cpu_percent(None))

        while not self._stop.wait(self.interval):
            try:
                snap = self._collect()
                snap["_taken_at"] = time.monotonic()
                self._last_snapshot = snap
                self._fh.write(json.dumps(snap, ensure_ascii=False) + "\n")
                self._samples += 1
                self._track_peaks(snap)
            except Exception:
                logging.getLogger("Telemetry").exception("Ошибка сбора телеметрии")

    def _collect(self) -> dict:
        levels, markers = self.handler.take_interval()
        snap = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "uptime_sec": round(time.monotonic() - self._started_at, 1),
            "tag": self.tag,
            "pid": self.pid,
            "log_levels": levels,
            "log_markers": markers,
            "threads": self._threads(),
        }
        if self._proc:
            snap["process"] = self._process_metrics()
        if self._controller is not None:
            snap["components"] = self._components()
        return snap

    # ------------------------------------------------------------------
    def _process_metrics(self) -> dict:
        out = {}
        p = self._proc
        out["cpu_percent"] = _safe(lambda: p.cpu_percent(None), -1)
        out["rss_mb"] = _safe(lambda: round(p.memory_info().rss / 1048576, 1), -1)
        out["num_threads"] = _safe(lambda: p.num_threads(), -1)
        if hasattr(p, "num_handles"):
            out["handles"] = _safe(lambda: p.num_handles(), -1)
        out["read_mb"] = _safe(lambda: round(p.io_counters().read_bytes / 1048576, 1), -1)
        return out

    @staticmethod
    def _threads() -> dict:
        """Количество живых потоков по группам имён."""
        counts = Counter()
        total = 0
        for t in threading.enumerate():
            total += 1
            name = t.name or ""
            for grp in THREAD_GROUPS:
                if name.startswith(grp):
                    counts[grp.rstrip("-")] += 1
                    break
            else:
                counts["other"] += 1
        counts["total"] = total
        return dict(counts)

    # ------------------------------------------------------------------
    def _audio_metrics(self) -> dict:
        """
        Метрики звука: качество подачи и очереди по дорожкам.

        Раньше телеметрия ПОДМЕНЯЛА MasterClock.push_audio(), чтобы
        измерять каждый блок PCM, и читала очереди _queue2/_queue3
        напрямую. Подмена чужого метода работала, но ломалась молча при
        изменении сигнатуры — и однажды сломалась, когда в push_audio
        добавился параметр pts. Чтение приватных очередей ломалось при
        смене их формата на пары (pts, samples).

        Теперь измерение живёт там, где данные: MasterClock считает
        качество в момент подачи и отдаёт через get_audio_diagnostics().
        Телеметрия только запрашивает готовый результат.
        """
        c = self._controller
        mc = getattr(c, "master_clock", None) if c else None
        if mc is None:
            return {"tracks": {}}

        diag = _safe(lambda: mc.get_audio_diagnostics(), None)
        if diag:
            # Совместимость с прежним форматом снимка: имена полей,
            # по которым построены отчёт и внешние скрипты.
            diag.setdefault("track_drift_samples", diag.get("queue_drift_samples"))
            diag.setdefault("track_drift_ms", diag.get("queue_drift_ms"))
            return diag

        # Запасной путь: старая версия MasterClock без диагностики.
        out = {"tracks": {}}
        for track_id in (2, 3):
            st = _safe(lambda t=track_id: mc.get_track_state(t), {}) or {}
            out["tracks"][str(track_id)] = st
        return out


    # ------------------------------------------------------------------
    @staticmethod
    def _env_info() -> dict:
        """
        Статическая информация об окружении. Пишется один раз, первой
        строкой в jsonl. Разрядность процесса критична: 32-битный Python
        упирается в ~2 ГБ адресного пространства, и тогда крупная
        аллокация (пересборка окна) падает с MemoryError даже при
        небольшом файле.
        """
        return {
            "python": sys.version.split()[0],
            "bits": struct.calcsize("P") * 8,
            "platform": platform.platform(),
            "executable": sys.executable,
        }

    def _files(self) -> dict:
        """
        Фактические размеры файлов на диске: исходный MP4, .idx и локальное
        зеркало. Нужно, чтобы соотносить потребление памяти с реальным
        объёмом данных, а не с предположениями.
        """
        c = self._controller
        out = {}
        for label, attr in (("mp4", "mp4_path"), ("idx", "idx_path"), ("ref", "ref_path")):
            path = getattr(c, attr, None)
            if not path:
                continue
            try:
                pth = Path(path)
                out[label] = {
                    "path": str(pth),
                    "size_mb": round(pth.stat().st_size / 1048576, 1) if pth.exists() else None,
                }
            except Exception:
                pass

        mirror = getattr(c, "mirror_path", None)
        if mirror:
            try:
                mp = Path(mirror)
                out["mirror"] = {
                    "path": str(mp),
                    "size_mb": round(mp.stat().st_size / 1048576, 2) if mp.exists() else None,
                }
            except Exception:
                pass
        return out

    @staticmethod
    def _probe_frame(buf) -> dict:
        """
        Снимает РЕАЛЬНЫЕ параметры кадра из буфера: разрешение, dtype и
        число байт. peek_first() не изменяет буфер, поэтому воспроизведению
        не мешает. Отсюда же считается фактический объём буфера — вместо
        оценок «на глазок».
        """
        out = {}
        try:
            entry = buf.peek_first()
            if entry is None:
                return out
            _pts, frame = entry
            out["shape"] = list(getattr(frame, "shape", []) or [])
            out["dtype"] = str(getattr(frame, "dtype", ""))
            nbytes = int(getattr(frame, "nbytes", 0) or 0)
            out["frame_kb"] = round(nbytes / 1024, 1)
            count = int(getattr(buf, "count", 0) or 0)
            maxf = int(getattr(buf, "max_frames", 0) or 0)
            out["used_mb"] = round(nbytes * count / 1048576, 1)
            out["max_possible_mb"] = round(nbytes * maxf / 1048576, 1)
        except Exception:
            pass
        return out

    @staticmethod
    def _index_memory(li) -> dict:
        """Фактический объём numpy-массивов индекса (nbytes, без оценок)."""
        out = {}
        for label, attr in (("all_193", "_all_193"), ("all_c9", "_all_c9"),
                            ("video_records", "_video_records_full"),
                            ("audio_tracks", "_audio_tracks_full")):
            arr = getattr(li, attr, None)
            if arr is None:
                out[label] = None      # None у audio_tracks = сброшен, будет пересчёт
                continue
            try:
                out[label] = {
                    "items": int(len(arr)),
                    "mb": round(int(arr.nbytes) / 1048576, 2),
                }
            except Exception:
                pass
        return out

    # ------------------------------------------------------------------
    def _components(self) -> dict:
        """
        Снимает состояние живых объектов. Каждый блок изолирован: сбой или
        отсутствие поля не рушит остальной снимок — блок просто пропускается.
        """
        c = self._controller
        out = {}

        out["controller"] = _safe(lambda: {
            "playing": getattr(c, "playing", None),
            "paused": getattr(c, "_paused", None),
            "total_frames": getattr(c, "total_frames", None),
            "ready": _safe(lambda: c.is_ready()),
            "closed": getattr(c, "_closed", None),
            "init_error": getattr(c, "_init_error", None),
            "active_tracks": list(getattr(c, "active_tracks", []) or []),
        }, {})

        mc = getattr(c, "master_clock", None)
        if mc is not None:
            out["master_clock"] = _safe(lambda: mc.get_stats(), {})
            out.setdefault("master_clock", {})["audio_clock"] = _safe(
                lambda: mc.get_audio_clock(), -1)

        pb = getattr(c, "_playback", None)
        if pb is not None:
            out["playback"] = _safe(lambda: {
                "playing": getattr(pb, "playing", None),
                "paused": getattr(pb, "_paused", None),
                "current_frame": getattr(pb, "_current_frame_idx", None),
                "audio_clock": getattr(pb, "_audio_clock", None),
                "total_frames": getattr(pb, "total_frames", None),
                "seek_generation": getattr(pb, "_seek_generation", None),
                "transition": str(getattr(pb, "_transition_state", "")),
                "transition_gen": getattr(pb, "_transition_generation", None),
                "seek_speed": getattr(pb, "_seek_speed", None),
                "seek_direction": getattr(pb, "_seek_direction", None),
                "sliding": _safe(lambda: pb._sliding_in_progress.is_set()),
                "growth_refreshing": _safe(lambda: pb._growth_refresh_in_progress.is_set()),
            }, {})

            seen_buffers = {}
            for label, attr in (("display_buffer", "_display_buffer"),
                                ("fill_buffer", "_fill_buffer")):
                buf = getattr(pb, attr, None)
                if buf is not None:
                    info = _safe(lambda b=buf: {
                        "count": getattr(b, "count", None),
                        "max_frames": getattr(b, "max_frames", None),
                        "fill_pct": round(100.0 * b.count / b.max_frames, 1)
                        if getattr(b, "max_frames", 0) else None,
                    }, {})
                    # Реальные размеры кадра и занятая память — измерением,
                    # а не расчётом по предполагаемому разрешению.
                    info.update(self._probe_frame(buf))
                    out[label] = info
                    seen_buffers[id(buf)] = info.get("used_mb")

            # display и fill в норме указывают на ОДИН объект (см.
            # start_playback) — фиксируем это явно, иначе при чтении отчёта
            # легко посчитать их память дважды.
            out["buffers_shared"] = (
                getattr(pb, "_display_buffer", None) is getattr(pb, "_fill_buffer", None)
            )
            free_buf = getattr(pb, "_free_buffer", None)
            if free_buf is not None:
                fb = _safe(lambda: {"count": getattr(free_buf, "count", None),
                                    "max_frames": getattr(free_buf, "max_frames", None)}, {})
                fb.update(self._probe_frame(free_buf))
                out["free_buffer"] = fb

        pipe = getattr(c, "_pipeline", None)
        if pipe is not None:
            # Через публичные методы конвейера: раньше читались
            # _raw_queue/_video_queue/_audio_queue и _stages напрямую.
            out["pipeline"] = _safe(lambda: {
                **(pipe.get_queue_sizes() or {}),
                **(pipe.get_stage_status() or {}),
            }, {})

            win = _safe(lambda: pipe.get_window_snapshot())
            if win is not None:
                out["window"] = _safe(lambda: {
                    "start_frame": win.window_start_frame,
                    "end_frame": win.window_end_frame,
                    "start_chunk": win.window_start_chunk,
                    "total_chunks": win.total_chunks,
                }, {})

            # Планировщик — через конвейер: телеметрия не должна знать,
            # что он вообще существует как отдельный объект.
            out["scheduler"] = _safe(lambda: pipe.get_scheduler_state(), {})

            # Состояние ридера — через конвейер. Раньше телеметрия сама
            # обходила _stages, находила _reader и читала его приватные
            # поля: связь через два уровня чужой реализации.
            out["reader"] = _safe(lambda: pipe.get_reader_state(), {})

        se = getattr(c, "_seek_engine", None)
        if se is not None:
            # Через публичный метод движка вместо чтения _generation,
            # workers[].state и приватных очередей команд/результатов.
            out["seek_engine"] = _safe(lambda: se.get_state(), {})
            # Суммарная память буферов seek-воркеров: у каждого свой
            # FrameRingBuffer, и при трёх воркерах это заметная величина,
            # которую легко упустить, глядя только на видеобуфер плеера.
            total_mb = 0.0
            for w in getattr(se, "workers", []):
                pf = self._probe_frame(getattr(w, "buffer", None))
                if pf.get("used_mb"):
                    total_mb += pf["used_mb"]
            out["seek_engine"]["workers_mem_mb"] = round(total_mb, 1)

        li = getattr(c, "_lazy_index", None)
        if li is not None:
            # Публичные методы индекса вместо чтения _next_scan_element
            # и прямого доступа к numpy-массивам.
            out["lazy_index"] = _safe(lambda: li.get_index_state(), {})
            out["index_memory"] = _safe(lambda: li.get_memory_usage(),
                                        self._index_memory(li))

        out["files"] = self._files()

        # Аудио: измерение живёт внутри MasterClock, здесь только запрос.
        out["audio"] = self._audio_metrics()
        return out

    # ------------------------------------------------------------------
    def _track_peaks(self, snap: dict):
        proc = snap.get("process", {})
        for k in ("rss_mb", "num_threads", "handles"):
            v = proc.get(k)
            if isinstance(v, (int, float)) and v > self._peaks[k]:
                self._peaks[k] = v

        thr = snap.get("threads", {})
        for k in ("SeekWorker", "total"):
            v = thr.get(k)
            if isinstance(v, (int, float)) and v > self._peaks[f"thr_{k}"]:
                self._peaks[f"thr_{k}"] = v

        comp = snap.get("components", {})

        # Фактические размеры кадра/буферов — для сводки в отчёте.
        db = comp.get("display_buffer", {})
        if db.get("frame_kb"):
            self._frame_info = {
                "shape": db.get("shape"),
                "dtype": db.get("dtype"),
                "frame_kb": db.get("frame_kb"),
                "max_possible_mb": db.get("max_possible_mb"),
            }
        for key, dst in (("used_mb", "video_buf_mb"),):
            v = db.get(key)
            if isinstance(v, (int, float)) and v > self._peaks[dst]:
                self._peaks[dst] = v
        wm = comp.get("seek_engine", {}).get("workers_mem_mb")
        if isinstance(wm, (int, float)) and wm > self._peaks["seek_buf_mb"]:
            self._peaks["seek_buf_mb"] = wm
        im = comp.get("index_memory", {})
        idx_mb = sum(v["mb"] for v in im.values() if isinstance(v, dict) and "mb" in v)
        if idx_mb > self._peaks["index_mb"]:
            self._peaks["index_mb"] = idx_mb
        self._files_info = comp.get("files", self._files_info)
        self._audio_summary = comp.get("audio", self._audio_summary)

        mc = comp.get("master_clock", {})
        if isinstance(mc.get("underruns"), int):
            self._peaks["underruns"] = max(self._peaks["underruns"], mc["underruns"])

        db = comp.get("display_buffer", {})
        if isinstance(db.get("fill_pct"), (int, float)):
            cur = db["fill_pct"]
            self._peaks["buffer_fill_max"] = max(self._peaks["buffer_fill_max"], cur)
            prev = self._mins.get("buffer_fill_min")
            self._mins["buffer_fill_min"] = cur if prev is None else min(prev, cur)

    # ------------------------------------------------------------------
    def _write_report(self):
        levels, markers, problems = self.handler.totals()
        dur = time.monotonic() - self._started_at

        L = []
        add = L.append
        add("=" * 68)
        add(f"ТЕЛЕМЕТРИЯ ProxyPlayer — {self.tag} (pid {self.pid})")
        add("=" * 68)
        add(f"Длительность: {dur/60:.1f} мин, снимков: {self._samples}")
        add(f"Данные:       {self.jsonl_path.name}")
        if self.errors_path.exists() and self.errors_path.stat().st_size > 0:
            add(f"ОШИБКИ:       {self.errors_path.name}  <-- стектрейсы здесь")
        add("")

        add("-- Окружение " + "-" * 53)
        for k, v in self._env.items():
            add(f"  {k:12s} {v}")
        add("")

        if self._files_info:
            add("-- Файлы " + "-" * 57)
            for label, info in self._files_info.items():
                sz = info.get("size_mb")
                add(f"  {label:8s} {sz if sz is not None else '?':>8} МБ  {info.get('path','')}")
            add("")

        add("-- Память: измеренная, не расчётная " + "-" * 30)
        if self._frame_info:
            fi = self._frame_info
            add(f"  кадр: {fi.get('shape')} {fi.get('dtype')} = {fi.get('frame_kb')} КБ")
            add(f"  видеобуфер при полном заполнении: {fi.get('max_possible_mb')} МБ")
        add(f"  видеобуфер пик:        {self._peaks.get('video_buf_mb', 0):.1f} МБ")
        add(f"  буферы seek-воркеров:  {self._peaks.get('seek_buf_mb', 0):.1f} МБ")
        add(f"  массивы индекса:       {self._peaks.get('index_mb', 0):.1f} МБ")
        rss = self._peaks.get("rss_mb", 0)
        accounted = (self._peaks.get('video_buf_mb', 0)
                     + self._peaks.get('seek_buf_mb', 0)
                     + self._peaks.get('index_mb', 0))
        add(f"  RSS процесса пик:      {rss:.0f} МБ")
        if rss:
            pct = 100.0 * accounted / rss
            add(f"  из них объяснено:      {accounted:.0f} МБ ({pct:.0f}%)")
            if pct > 110:
                add("  (>100% — нормально: пики буферов достигнуты в разные моменты,")
                add("   либо кадры разделяют память; сравнивать надо по одному снимку)")
        if rss and accounted < rss * 0.5:
            add("  ВНИМАНИЕ: больше половины памяти не объясняется буферами и")
            add("  индексом — искать в декодере (PyAV) или очередях конвейера.")
        add("")

        if self._audio_summary:
            a = self._audio_summary
            add("-- Аудио " + "-" * 57)
            for tid in ("2", "3"):
                t = a.get("tracks", {}).get(tid, {})
                if not t.get("blocks"):
                    continue
                add(f"  трек {tid}: блоков={t['blocks']}, сэмплов={t['samples']}, "
                    f"dtype={t.get('dtype')}")
                add(f"          пик={t.get('peak')}, rms={t.get('rms_avg')}, "
                    f"клиппинг={t.get('clipped')}, NaN/inf={t.get('non_finite')}")
                add(f"          блок мин/макс={t.get('min_block')}/{t.get('max_block')}")
            if a.get("pushed_drift_samples") is not None:
                d = a["pushed_drift_samples"]
                add(f"  расхождение дорожек (подано): {d} сэмплов "
                    f"({d/48.0:.0f} мс)")
            if a.get("track_drift_samples") is not None:
                d = a["track_drift_samples"]
                add(f"  расхождение очередей:         {d} сэмплов "
                    f"({a.get('track_drift_ms')} мс)")
            add("")

        add("-- Пики " + "-" * 58)
        if self._peaks:
            for k in sorted(self._peaks):
                add(f"  {k:22s} {self._peaks[k]:.0f}")
        for k, v in self._mins.items():
            add(f"  {k:22s} {v:.0f}")
        add("")

        add("-- Уровни логов " + "-" * 50)
        for lvl in ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"):
            if levels.get(lvl):
                add(f"  {lvl:10s} {levels[lvl]:>8d}")
        add("")

        if markers:
            add("-- События " + "-" * 55)
            for k, v in sorted(markers.items(), key=lambda x: -x[1]):
                add(f"  {k:22s} {v:>8d}")
            add("")

        add("-- Диагностика " + "-" * 51)
        notes = []
        if self._peaks.get("underruns"):
            notes.append(f"Провалы звука (underruns): {self._peaks['underruns']:.0f}. "
                         "Сверить в jsonl с buffer_fill и scheduler.loading_allowed.")
        if markers.get("short_read"):
            notes.append(f"Частичные чтения SMB: {markers['short_read']}.")
        if markers.get("worker_retired") or markers.get("watchdog"):
            notes.append("Зависали SeekWorker'ы — смотреть thr_SeekWorker: "
                         "если пик растёт и не падает, потоки утекают.")
        if self._peaks.get("thr_SeekWorker", 0) > 6:
            notes.append(f"Пик потоков SeekWorker: {self._peaks['thr_SeekWorker']:.0f} "
                         "(норма — до 3-4 при пуле из 3 воркеров).")
        if not markers.get("filesize_refreshed") and not markers.get("mdat_refreshed"):
            notes.append("Размер файла и mdat_end ни разу не обновлялись — "
                         "для растущего файла это значит, что рост не отслеживался.")
        a = self._audio_summary or {}
        for tid in ("2", "3"):
            t = a.get("tracks", {}).get(tid, {})
            if not t.get("blocks"):
                continue
            dt = (t.get("dtype") or "")
            peak = t.get("peak") or 0
            if peak > 1.5:
                notes.append(
                    f"Трек {tid}: пик амплитуды {peak} — сигнал ВНЕ диапазона "
                    "-1..+1, ожидаемого потоком float32. Похоже, декодер отдаёт "
                    "целочисленные сэмплы (int16), и astype(float32) в "
                    "push_audio() не масштабирует их. Это и есть треск/шум.")
            elif 0 < peak < 0.02:
                notes.append(f"Трек {tid}: пик {peak} — сигнал почти нулевой "
                             "(тишина или неверный масштаб).")
            if t.get("non_finite"):
                notes.append(f"Трек {tid}: {t['non_finite']} NaN/inf сэмплов — "
                             "повреждённые кадры (вероятно, обрезанные пакеты).")
            if t.get("clipped"):
                notes.append(f"Трек {tid}: {t['clipped']} сэмплов с клиппингом.")
            if dt and "float32" not in dt:
                notes.append(f"Трек {tid}: dtype={dt} — поток ожидает float32.")
        drift = a.get("pushed_drift_samples")
        if isinstance(drift, int) and abs(drift) > 4800:
            notes.append(f"Дорожки разъехались на {drift} сэмплов "
                         f"({drift/48.0:.0f} мс) — левое и правое ухо не синхронны.")

        if self._mins.get("buffer_fill_min") == 0:
            notes.append("Видеобуфер опустошался до нуля — были моменты без кадров.")
        if notes:
            for n in notes:
                add(f"  * {n}")
        else:
            add("  Аномалий не обнаружено.")
        add("")

        if problems:
            add("-- Последние ошибки " + "-" * 46)
            for p in problems[-30:]:
                add(f"  {p['t']} [{p['level']}] {p['logger']} ({p['thread']}): {p['msg'][:150]}")

        self.report_path.write_text("\n".join(L), encoding="utf-8")
        logging.getLogger("Telemetry").info("Отчёт телеметрии: %s", self.report_path)


# ---------------------------------------------------------------------------
# Публичный API
# ---------------------------------------------------------------------------
_instance = None


def install_telemetry(outdir, tag: str = "player",
                      interval: float = DEFAULT_INTERVAL_SEC) -> "Telemetry":
    """
    Включает телеметрию в текущем процессе. Вызывать сразу после
    setup_logging(). Повторный вызов возвращает уже созданный экземпляр.
    """
    global _instance
    if _instance is not None:
        return _instance
    _instance = Telemetry(Path(outdir) / "telemetry", tag, interval)
    return _instance


def attach_controller(controller) -> None:
    """Привязывает StreamController после его создания."""
    if _instance is not None and controller is not None:
        _instance.attach(controller)


def get_telemetry():
    return _instance


def stop_telemetry() -> None:
    if _instance is not None:
        _instance.stop()
