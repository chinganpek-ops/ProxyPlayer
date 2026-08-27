#!/usr/bin/env python3
"""
soak_test.py – многочасовой тест ProxyPlayer на реальном растущем файле.

ЗАЧЕМ
Проблемы, ради которых он написан, не проявляются за 10 минут ручной
проверки: утечки памяти и HANDLE, накопление потоков от зависших
SeekWorker'ов, рассинхрон окна на растущем файле, деградация звука спустя
часы, «вечная заглушка» после неудачной перемотки. Тест запускается на
ночь и утром оставляет готовый отчёт.

ЧТО ДЕЛАЕТ
1. Открывает реальный файл через StreamController — тот же путь, что и
   плеер (зеркало .idx, LazyIndex, конвейер, MasterClock).
2. Крутит рендер-цикл на нужном fps: без него буферы не расходуются и
   поведение отличается от реального воспроизведения.
3. Раз в N секунд выполняет один сценарий из репертуара оператора:
   пауза/воспроизведение, перемотка внутри окна, перемотка далеко за
   окно (назад и вперёд), возврат в live, JKL-перемотка, переключение
   дорожек, две перемотки подряд (проверка отмены), прыжок в начало.
4. Непрерывно следит за здоровьем: не встали ли часы при playing,
   растут ли RSS/потоки/HANDLE, появляются ли underrun'ы, отвечает ли
   перемотка, не сыпятся ли ошибки.
5. Пишет отчёт КАЖДЫЕ 10 минут (а не только в конце) — если процесс
   упадёт ночью, утром всё равно будет что читать.

ЗАПУСК
    python soak_test.py "D:\\media\\1808854_2026-08-24T10-00-00.000.mp4"
    python soak_test.py 1808854                 # по ID, homedir из конфига
    python soak_test.py <файл> --hours 12 --action-interval 60
    python soak_test.py <файл> --hours 8 --no-audio    # без звуковой карты

ОТЧЁТ
    soak_report_<ts>.txt   — сводка, инциденты, вердикт (читать утром)
    soak_events_<ts>.jsonl — таймсерия для графиков
    плюс файлы player_telemetry, если модуль доступен

ОСТАНОВКА
    Ctrl+C — корректно завершает и дописывает отчёт.
"""

import argparse
import json
import logging
import os
import random
import signal
import sys
import threading
import time
import traceback
from collections import Counter, deque
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger("SoakTest")

# ---------------------------------------------------------------------------
# Пороги, при которых тест фиксирует инцидент
# ---------------------------------------------------------------------------
STALL_SECONDS = 15.0          # часы не двигаются при playing — залипание
SEEK_TIMEOUT_SEC = 30.0       # перемотка не ответила
RSS_GROWTH_ALERT_MB = 500     # рост RSS сверх стартового + буферы
THREAD_GROWTH_ALERT = 15      # рост числа потоков относительно старта
REPORT_EVERY_SEC = 600        # переписывать отчёт раз в 10 минут


class Incident:
    __slots__ = ("t", "kind", "detail")

    def __init__(self, kind, detail):
        self.t = datetime.now()
        self.kind = kind
        self.detail = detail


class SoakTest:
    def __init__(self, args):
        self.args = args
        self.stop_flag = threading.Event()
        self.controller = None

        self.started = datetime.now()
        self.t0 = time.monotonic()

        self.actions = Counter()
        self.action_fail = Counter()
        self.incidents = []
        self.seek_latencies = deque(maxlen=500)
        self.samples = []

        self.baseline = {}
        self.peaks = Counter()
        self._last_clock = None
        self._last_clock_change = time.monotonic()
        self._frames_rendered = 0
        self._render_errors = 0

        ts = self.started.strftime("%Y%m%d_%H%M%S")
        outdir = Path(args.outdir) if args.outdir else (ROOT / "soak")
        outdir.mkdir(parents=True, exist_ok=True)
        self.report_path = outdir / f"soak_report_{ts}.txt"
        self.events_path = outdir / f"soak_events_{ts}.jsonl"
        self._events_fh = open(self.events_path, "a", encoding="utf-8", buffering=1)

        self.proc = psutil.Process(os.getpid()) if psutil else None

    # ------------------------------------------------------------------
    # Подготовка
    # ------------------------------------------------------------------
    def resolve_media(self):
        """Находит MP4 по пути или по числовому ID (как это делает main.py)."""
        from PyQt5.QtCore import QStandardPaths
        from config.config import load_config

        cfg_path = Path(QStandardPaths.writableLocation(
            QStandardPaths.AppConfigLocation)) / "player_config.json"
        config = load_config(cfg_path)

        raw = self.args.media
        path = Path(raw)
        if path.exists():
            return path, config

        # Числовой ID — ищем в homedir, как это делает плеер.
        homedir = config.get("homedir", "")
        if raw.isdigit() and homedir:
            try:
                from utils.utils import resolve_id_to_mp4
            except ImportError:
                from utils import resolve_id_to_mp4
            found = resolve_id_to_mp4(raw, homedir)
            if found:
                return found, config

        raise FileNotFoundError(f"Не найден медиафайл: {raw} (homedir={homedir!r})")

    @staticmethod
    def find_idx(mp4_path: Path) -> Path:
        idx = mp4_path.parent / "idx" / "mp4" / f"{mp4_path.stem}.idx"
        if not idx.exists():
            idx = mp4_path.parent / f"{mp4_path.stem}.idx"
        return idx

    @staticmethod
    def find_ref(mp4_path: Path) -> Path:
        ref = mp4_path.parent / f"{mp4_path.stem}.mp4.ref"
        if not ref.exists():
            ref = mp4_path.parent / f"{mp4_path.stem}.ref"
        return ref

    def setup(self):
        mp4_path, config = self.resolve_media()
        idx_path = self.find_idx(mp4_path)
        ref_path = self.find_ref(mp4_path)
        if not idx_path.exists():
            raise FileNotFoundError(f"Не найден индекс: {idx_path}")

        logger.info("Файл:   %s", mp4_path)
        logger.info("Индекс: %s", idx_path)

        # Зеркало готовит IndexService. В тесте эмулируем его фоновым
        # потоком: prepare_mirror() — ровно то, что сервис делает по
        # таймеру. Так тест не зависит от запущенного сервиса и при этом
        # воспроизводит рост зеркала.
        from index.idx_cache import prepare_mirror
        mirror = prepare_mirror(idx_path)
        logger.info("Зеркало: %s", mirror)

        self._mirror_thread = threading.Thread(
            target=self._mirror_loop, args=(idx_path,), daemon=True,
            name="SoakMirrorSync")
        self._mirror_thread.start()

        from core.stream_controller import StreamController
        self.controller = StreamController(
            ref_path=ref_path,
            idx_path=idx_path,
            mp4_path=mp4_path,
            fps=config.get("fps", 25.0),
            buffer_size=config.get("buffer_size", 360),
            audio_delay_ms=config.get("audio_delay_ms", 0),
            start_from_live=True,
            mirror_path=str(mirror),
        )

        if not self.controller._ready.wait(timeout=180):
            raise RuntimeError("StreamController не инициализировался за 180 с")
        if self.controller._init_error:
            raise RuntimeError(f"Ошибка инициализации: {self.controller._init_error}")

        # Телеметрия — необязательна, но с ней отчёт заметно информативнее.
        try:
            from player_telemetry import install_telemetry, attach_controller
            install_telemetry(self.report_path.parent, tag="soak",
                              interval=self.args.sample_interval)
            attach_controller(self.controller)
            logger.info("Телеметрия подключена")
        except Exception as e:
            logger.warning("Телеметрия недоступна: %s", e)

        self.controller.start_playback()
        time.sleep(2.0)
        self.controller.resume()

        self.baseline = self._process_metrics()
        self.baseline["total_frames"] = self.controller.total_frames
        logger.info("Старт: total_frames=%s, RSS=%s МБ",
                    self.controller.total_frames, self.baseline.get("rss_mb"))

    def _mirror_loop(self, idx_path):
        """Эмулирует IndexService: периодически докачивает зеркало."""
        from index.idx_cache import prepare_mirror
        while not self.stop_flag.wait(self.args.mirror_interval):
            try:
                prepare_mirror(idx_path)
            except Exception as e:
                self.incident("mirror_sync_error", str(e))

    # ------------------------------------------------------------------
    # Служебное
    # ------------------------------------------------------------------
    def incident(self, kind, detail):
        inc = Incident(kind, detail)
        self.incidents.append(inc)
        logger.warning("ИНЦИДЕНТ [%s] %s", kind, detail)
        self._write_event({"type": "incident", "kind": kind, "detail": str(detail)[:500]})

    def _write_event(self, obj):
        try:
            obj["ts"] = datetime.now().isoformat(timespec="seconds")
            obj["uptime_min"] = round((time.monotonic() - self.t0) / 60, 1)
            self._events_fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _process_metrics(self):
        if not self.proc:
            return {}
        out = {}
        try:
            out["rss_mb"] = round(self.proc.memory_info().rss / 1048576, 1)
            out["threads"] = self.proc.num_threads()
            if hasattr(self.proc, "num_handles"):
                out["handles"] = self.proc.num_handles()
            out["cpu"] = self.proc.cpu_percent(None)
        except Exception:
            pass
        return out

    # ------------------------------------------------------------------
    # Рендер-цикл: без него буферы не расходуются
    # ------------------------------------------------------------------
    def _render_loop(self):
        fps = self.controller.fps or 25.0
        period = 1.0 / fps
        while not self.stop_flag.is_set():
            start = time.monotonic()
            try:
                frame = self.controller.get_display_frame()
                if frame is not None:
                    self._frames_rendered += 1
            except Exception as e:
                self._render_errors += 1
                if self._render_errors in (1, 10, 100):
                    self.incident("render_error", f"{e}\n{traceback.format_exc()[:400]}")
            elapsed = time.monotonic() - start
            time.sleep(max(0.0, period - elapsed))

    # ------------------------------------------------------------------
    # Сценарии оператора
    # ------------------------------------------------------------------
    def _current_frame(self):
        try:
            from config.timebase import pts_to_video_frame
            return pts_to_video_frame(self.controller.audio_clock)
        except Exception:
            return 0

    def _do_seek(self, target, label):
        """Перемотка с ожиданием ответа и замером задержки."""
        total = max(1, self.controller.total_frames)
        target = max(0, min(int(target), total - 1))
        done = threading.Event()
        result = {}

        def on_ok():
            result["ok"] = True
            done.set()

        def on_err(msg):
            result["error"] = str(msg)
            done.set()

        t0 = time.monotonic()
        self.controller.seek_absolute(target, callback=on_ok, on_error=on_err)
        answered = done.wait(SEEK_TIMEOUT_SEC)
        latency = time.monotonic() - t0

        if not answered:
            self.action_fail[label] += 1
            self.incident("seek_timeout",
                          f"{label}: нет ответа за {SEEK_TIMEOUT_SEC} с (кадр {target})")
            return False
        if "error" in result:
            self.action_fail[label] += 1
            self.incident("seek_error", f"{label}: {result['error']} (кадр {target})")
            return False

        self.seek_latencies.append(latency)
        if latency > 5.0:
            self.incident("seek_slow", f"{label}: {latency:.1f} с (кадр {target})")
        return True

    def act_pause_resume(self):
        self.controller.pause()
        time.sleep(random.uniform(2, 8))
        self.controller.resume()

    def act_seek_in_window(self):
        cur = self._current_frame()
        delta = random.randint(-25 * 20, 25 * 20)      # ±20 секунд
        if self._do_seek(cur + delta, "seek_in_window"):
            self.controller.resume()

    def act_seek_back_far(self):
        """Далеко назад — гарантированный выход за пределы окна."""
        cur = self._current_frame()
        delta = random.randint(25 * 60 * 5, 25 * 60 * 20)   # 5-20 минут назад
        if self._do_seek(cur - delta, "seek_back_far"):
            self.controller.resume()

    def act_seek_forward_far(self):
        cur = self._current_frame()
        delta = random.randint(25 * 60 * 3, 25 * 60 * 10)
        if self._do_seek(cur + delta, "seek_forward_far"):
            self.controller.resume()

    def act_go_live(self):
        live = max(0, self.controller.total_frames - 1600)
        if self._do_seek(live, "go_live"):
            self.controller.resume()

    def act_seek_start(self):
        if self._do_seek(0, "seek_start"):
            self.controller.resume()

    def act_double_seek(self):
        """
        Две перемотки подряд: вторая обязана отменить первую. Проверяет
        механизм поколений и отсутствие «залипания» после отмены.
        """
        cur = self._current_frame()
        self.controller.seek_absolute(max(0, cur - 25 * 120))
        time.sleep(0.3)
        if self._do_seek(cur + 25 * 60, "double_seek"):
            self.controller.resume()

    def act_jkl(self):
        direction = random.choice((-1, 1))
        self.controller.set_seek_speed(direction)
        time.sleep(random.uniform(2, 6))
        if random.random() < 0.5:
            self.controller.set_seek_speed(direction)   # ускоряем ещё
            time.sleep(random.uniform(2, 4))
        self.controller.reset_seek_speed()
        self.controller.resume()

    def act_tracks(self):
        combo = random.choice(([2, 3], [2], [3], [2, 3]))
        self.controller.set_active_tracks(combo)

    def act_idle(self):
        """Просто смотреть — самый частый режим оператора."""
        time.sleep(random.uniform(10, 30))

    def scenarios(self):
        # Веса примерно отражают реальное поведение: чаще смотрят, реже дёргают.
        return [
            (self.act_idle, 4),
            (self.act_seek_in_window, 3),
            (self.act_pause_resume, 2),
            (self.act_seek_back_far, 2),
            (self.act_go_live, 2),
            (self.act_jkl, 2),
            (self.act_seek_forward_far, 1),
            (self.act_double_seek, 1),
            (self.act_tracks, 1),
            (self.act_seek_start, 1),
        ]

    # ------------------------------------------------------------------
    # Контроль здоровья
    # ------------------------------------------------------------------
    def health_check(self):
        m = self._process_metrics()
        clock = self.controller.audio_clock
        playing = bool(getattr(self.controller, "playing", False))

        # Часы не двигаются при воспроизведении — залипание конвейера.
        if clock != self._last_clock:
            self._last_clock = clock
            self._last_clock_change = time.monotonic()
        elif playing and (time.monotonic() - self._last_clock_change) > STALL_SECONDS:
            self.incident("playback_stall",
                          f"audio_clock не менялся {STALL_SECONDS:.0f} с при playing=True")
            self._last_clock_change = time.monotonic()   # не спамим каждую секунду

        for key in ("rss_mb", "threads", "handles"):
            if key in m and m[key] > self.peaks[key]:
                self.peaks[key] = m[key]

        base_rss = self.baseline.get("rss_mb", 0)
        if base_rss and m.get("rss_mb", 0) - base_rss > RSS_GROWTH_ALERT_MB:
            self.incident("memory_growth",
                          f"RSS вырос с {base_rss} до {m['rss_mb']} МБ")
            self.baseline["rss_mb"] = m["rss_mb"]        # порог сдвигаем

        base_thr = self.baseline.get("threads", 0)
        if base_thr and m.get("threads", 0) - base_thr > THREAD_GROWTH_ALERT:
            self.incident("thread_growth",
                          f"потоков было {base_thr}, стало {m['threads']} — "
                          "вероятно, воркеры не завершаются")
            self.baseline["threads"] = m["threads"]

        sample = {"type": "sample", "clock": clock, "playing": playing,
                  "total_frames": self.controller.total_frames,
                  "frames_rendered": self._frames_rendered, **m}
        self.samples.append(sample)
        self._write_event(sample)

    # ------------------------------------------------------------------
    # Основной цикл
    # ------------------------------------------------------------------
    def run(self):
        deadline = self.t0 + self.args.hours * 3600
        render = threading.Thread(target=self._render_loop, daemon=True,
                                  name="SoakRender")
        render.start()

        scenarios = self.scenarios()
        population = [s for s, w in scenarios for _ in range(w)]

        next_action = time.monotonic() + self.args.action_interval
        next_health = time.monotonic() + self.args.sample_interval
        next_report = time.monotonic() + REPORT_EVERY_SEC

        logger.info("Тест запущен на %.1f ч, действие раз в %d с",
                    self.args.hours, self.args.action_interval)

        while not self.stop_flag.is_set() and time.monotonic() < deadline:
            now = time.monotonic()

            if now >= next_health:
                next_health = now + self.args.sample_interval
                try:
                    self.health_check()
                except Exception as e:
                    self.incident("health_check_error", str(e))

            if now >= next_action:
                next_action = now + self.args.action_interval
                action = random.choice(population)
                name = action.__name__
                self.actions[name] += 1
                logger.info("[%s] действие: %s (кадр %s)",
                            self._uptime_str(), name, self._current_frame())
                self._write_event({"type": "action", "name": name})
                try:
                    action()
                except Exception as e:
                    self.action_fail[name] += 1
                    self.incident("action_error",
                                  f"{name}: {e}\n{traceback.format_exc()[:500]}")

            if now >= next_report:
                next_report = now + REPORT_EVERY_SEC
                self.write_report(final=False)

            self.stop_flag.wait(0.5)

        logger.info("Завершение теста...")
        self.stop_flag.set()
        try:
            self.controller.close()
        except Exception as e:
            self.incident("close_error", str(e))
        self.write_report(final=True)

    def _uptime_str(self):
        return str(timedelta(seconds=int(time.monotonic() - self.t0)))

    # ------------------------------------------------------------------
    # Отчёт
    # ------------------------------------------------------------------
    def write_report(self, final: bool):
        dur_h = (time.monotonic() - self.t0) / 3600
        L = []
        add = L.append

        add("=" * 72)
        add(f"SOAK-ТЕСТ ProxyPlayer — {'ЗАВЕРШЁН' if final else 'В ПРОЦЕССЕ'}")
        add("=" * 72)
        add(f"Начало:        {self.started.strftime('%Y-%m-%d %H:%M:%S')}")
        add(f"Длительность:  {dur_h:.2f} ч из запланированных {self.args.hours}")
        add(f"Файл:          {self.args.media}")
        add("")

        add("-- Итог " + "-" * 62)
        critical = [i for i in self.incidents
                    if i.kind in ("playback_stall", "seek_timeout", "seek_error",
                                  "action_error", "render_error", "thread_growth")]
        if not self.incidents:
            add("  ЗАМЕЧАНИЙ НЕТ — тест пройден чисто.")
        elif not critical:
            add(f"  Некритичные замечания: {len(self.incidents)} (см. ниже).")
        else:
            add(f"  ОБНАРУЖЕНЫ ПРОБЛЕМЫ: {len(critical)} критичных "
                f"из {len(self.incidents)} инцидентов.")
        add("")

        add("-- Действия оператора " + "-" * 48)
        total_actions = sum(self.actions.values())
        add(f"  Всего: {total_actions}")
        for name, cnt in sorted(self.actions.items(), key=lambda x: -x[1]):
            fails = self.action_fail.get(name, 0)
            mark = "  ОШИБОК: %d" % fails if fails else ""
            add(f"    {name:22s} {cnt:>5d}{mark}")
        add("")

        if self.seek_latencies:
            lat = sorted(self.seek_latencies)
            add("-- Отклик перемотки " + "-" * 50)
            add(f"  замеров: {len(lat)}")
            add(f"  медиана: {lat[len(lat)//2]:.2f} с")
            add(f"  95-й перцентиль: {lat[int(len(lat)*0.95)]:.2f} с")
            add(f"  максимум: {lat[-1]:.2f} с")
            add("")

        add("-- Ресурсы " + "-" * 59)
        if psutil is None:
            add("  psutil не установлен — метрики памяти и потоков недоступны.")
            add("  Для полноценного отчёта: pip install psutil")
        else:
            add(f"  RSS:     старт {self.baseline.get('rss_mb', '?')} МБ, "
                f"пик {self.peaks.get('rss_mb', 0)} МБ")
            add(f"  Потоки:  старт {self.baseline.get('threads', '?')}, "
                f"пик {self.peaks.get('threads', 0)}")
            if self.peaks.get("handles"):
                add(f"  HANDLE:  пик {self.peaks['handles']}")
        add(f"  Кадров отрисовано: {self._frames_rendered}")
        add(f"  Ошибок рендера:    {self._render_errors}")
        add("")

        add("-- Рост файла " + "-" * 56)
        add(f"  total_frames: старт {self.baseline.get('total_frames', '?')}, "
            f"сейчас {getattr(self.controller, 'total_frames', '?')}")
        if self.samples:
            first, last = self.samples[0], self.samples[-1]
            grew = last.get("total_frames", 0) - first.get("total_frames", 0)
            add(f"  прирост за тест: {grew} кадров "
                f"({grew / 25 / 60:.1f} мин видео)")
            if grew <= 0:
                add("  ВНИМАНИЕ: файл не рос — либо запись остановлена,")
                add("  либо не работает отслеживание роста индекса.")
        add("")

        if self.incidents:
            add("-- Инциденты " + "-" * 57)
            by_kind = Counter(i.kind for i in self.incidents)
            for kind, cnt in by_kind.most_common():
                add(f"  {kind:22s} {cnt:>5d}")
            add("")
            add("  Последние 40:")
            for inc in self.incidents[-40:]:
                add(f"    {inc.t.strftime('%H:%M:%S')} [{inc.kind}] "
                    f"{str(inc.detail)[:160]}")
            add("")

        add("-- Что смотреть дальше " + "-" * 47)
        hints = []
        if any(i.kind == "playback_stall" for i in self.incidents):
            hints.append("Были залипания воспроизведения — сверить время инцидента "
                         "с jsonl телеметрии (scheduler.loading_allowed, buffer fill).")
        if any(i.kind in ("seek_timeout", "seek_error") for i in self.incidents):
            hints.append("Перемотка отказывала — смотреть telemetry_*_ERRORS.log.")
        if self.peaks.get("threads", 0) - self.baseline.get("threads", 0) > 5:
            hints.append("Число потоков выросло — вероятно, SeekWorker'ы зависают "
                         "и выводятся из пула (искать 'выведен из пула').")
        if self.peaks.get("rss_mb", 0) > self.baseline.get("rss_mb", 0) * 2:
            hints.append("RSS вырос более чем вдвое — проверить buffer_size и "
                         "рост массивов индекса (index_memory в телеметрии).")
        if hints:
            for h in hints:
                add(f"  * {h}")
        else:
            add("  Аномалий, требующих разбора, не выявлено.")

        self.report_path.write_text("\n".join(L), encoding="utf-8")
        if final:
            logger.info("Отчёт записан: %s", self.report_path)


def main():
    ap = argparse.ArgumentParser(
        description="Многочасовой soak-тест ProxyPlayer на реальном файле")
    ap.add_argument("media", help="путь к MP4 или числовой ID (как в main.py)")
    ap.add_argument("--hours", type=float, default=8.0, help="длительность (по умолчанию 8)")
    ap.add_argument("--action-interval", type=float, default=60.0,
                    help="секунд между действиями оператора (по умолчанию 60)")
    ap.add_argument("--sample-interval", type=float, default=5.0,
                    help="секунд между замерами здоровья")
    ap.add_argument("--mirror-interval", type=float, default=10.0,
                    help="секунд между докачками зеркала (эмуляция IndexService)")
    ap.add_argument("--outdir", default=None, help="куда писать отчёты")
    ap.add_argument("--seed", type=int, default=None,
                    help="фиксировать последовательность действий для воспроизводимости")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    test = SoakTest(args)

    def _stop(signum, frame):
        logger.info("Получен сигнал остановки, завершаю...")
        test.stop_flag.set()

    signal.signal(signal.SIGINT, _stop)
    try:
        signal.signal(signal.SIGTERM, _stop)
    except (ValueError, AttributeError):
        pass

    try:
        test.setup()
    except Exception as e:
        logger.exception("Не удалось запустить тест")
        test.incident("setup_failed", str(e))
        test.write_report(final=True)
        return 2

    try:
        test.run()
    except Exception:
        logger.exception("Аварийное завершение теста")
        test.incident("fatal", traceback.format_exc()[:1000])
        test.write_report(final=True)
        return 1

    crit = [i for i in test.incidents
            if i.kind in ("playback_stall", "seek_timeout", "seek_error",
                          "action_error", "render_error", "thread_growth")]
    print(f"\nОтчёт: {test.report_path}")
    return 1 if crit else 0


if __name__ == "__main__":
    sys.exit(main())
