#!/usr/bin/env python3
"""
load_test.py – нагрузочный тест: последовательный запуск нескольких окон
плеера на длинных файлах со снятием метрик после каждого запуска.

ЗАЧЕМ
На коротких файлах (20 минут) шесть окон укладываются в разумную память,
а на многочасовых — нет. Причина в том, что масштабируется линейно только
одна величина: массивы индекса, которые КАЖДЫЙ процесс держит целиком и
независимо от остальных. Видеобуфер, буферы seek-воркеров и накладные
расходы Qt/PyAV от длины файла не зависят.

Тест измеряет это прямо: запускает окна по одному, после каждого даёт
буферам наполниться и снимает срез. В отчёте видно, сколько памяти
добавляет каждое следующее окно и на каком количестве окон упирается
доступная память.

ЧТО ИЗМЕРЯЕТСЯ на каждом шаге
  - RSS, потоки, HANDLE каждого процесса плеера по отдельности
  - суммарное потребление всех окон
  - свободная память системы
  - прирост относительно предыдущего шага (цена одного окна)
  - вес индекса, если окно успело записать телеметрию

ЗАПУСК
    python load_test.py file1.mp4 file2.mp4 file3.mp4 file4.mp4 file5.mp4 file6.mp4
    python load_test.py 1979587 1979597 --settle 60
    python load_test.py <файл> --repeat 6          # один файл шесть раз
    python load_test.py <файлы...> --hold 600      # подержать окна 10 минут в конце

Аргументы — пути к MP4 или числовые ID (как в main.py).

ОТЧЁТ
    load_report_<ts>.txt   — таблица по шагам и выводы
    load_metrics_<ts>.csv  — сырые метрики для графиков
"""

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    import psutil
except ImportError:
    print("Требуется psutil:  pip install psutil", file=sys.stderr)
    sys.exit(1)


def resolve_media(raw: str) -> Path:
    """Путь или числовой ID — как в main.py."""
    p = Path(raw)
    if p.exists():
        return p
    try:
        from PyQt5.QtCore import QStandardPaths
        from config.config import load_config
        cfg = Path(QStandardPaths.writableLocation(
            QStandardPaths.AppConfigLocation)) / "player_config.json"
        config = load_config(cfg)
        homedir = config.get("homedir", "")
        if raw.isdigit() and homedir:
            try:
                from utils.utils import resolve_id_to_mp4
            except ImportError:
                from utils import resolve_id_to_mp4
            found = resolve_id_to_mp4(raw, homedir)
            if found:
                return found
    except Exception as e:
        print(f"(конфиг недоступен: {e})")
    raise FileNotFoundError(f"Не найден файл: {raw}")


def find_idx(mp4: Path) -> Path:
    idx = mp4.parent / "idx" / "mp4" / f"{mp4.stem}.idx"
    return idx if idx.exists() else mp4.parent / f"{mp4.stem}.idx"


class Window:
    """Один запущенный процесс окна плеера."""

    __slots__ = ("index", "mp4", "idx", "proc", "started_at")

    def __init__(self, index, mp4, idx, proc):
        self.index = index
        self.mp4 = mp4
        self.idx = idx
        self.proc = proc
        self.started_at = time.monotonic()

    @property
    def pid(self):
        return self.proc.pid

    def alive(self):
        return self.proc.poll() is None


class LoadTest:
    def __init__(self, args):
        self.args = args
        self.windows = []
        self.steps = []
        self.started = datetime.now()

        ts = self.started.strftime("%Y%m%d_%H%M%S")
        outdir = Path(args.outdir) if args.outdir else (ROOT / "loadtest")
        outdir.mkdir(parents=True, exist_ok=True)
        self.report_path = outdir / f"load_report_{ts}.txt"
        self.csv_path = outdir / f"load_metrics_{ts}.csv"

        self._csv = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._wr = csv.writer(self._csv)
        self._wr.writerow(["step", "windows", "pid", "file", "rss_mb",
                           "threads", "handles", "total_rss_mb",
                           "system_avail_mb"])

    # ------------------------------------------------------------------
    def prepare(self, raw_list):
        """Резолвит файлы и готовит зеркала индексов ДО запуска окон."""
        media = []
        for raw in raw_list:
            mp4 = resolve_media(raw)
            idx = find_idx(mp4)
            if not idx.exists():
                raise FileNotFoundError(f"Не найден индекс для {mp4}: {idx}")
            media.append((mp4, idx))

        print("Подготовка зеркал индексов (это может занять время на "
              "длинных файлах и медленной сети)...")
        from index.idx_cache import prepare_mirror
        prepared = []
        for mp4, idx in media:
            t0 = time.monotonic()
            mirror = prepare_mirror(idx)
            dt = time.monotonic() - t0
            size_mb = idx.stat().st_size / 1048576
            print(f"  {mp4.name}: индекс {size_mb:.1f} МБ, зеркало за {dt:.1f} с")
            prepared.append((mp4, idx, mirror))
        print()
        return prepared

    # ------------------------------------------------------------------
    def spawn_window(self, index, mp4, idx, mirror):
        """
        Запускает окно тем же способом, что и ManagerWindow: отдельным
        процессом main.py с флагом --managed. Так измеряется реальная
        конфигурация, а не упрощённая модель в одном процессе.
        """
        exe = sys.executable
        if getattr(sys, "frozen", False):
            cmd = [exe, str(mp4), "--managed", "--mirror", str(mirror)]
        else:
            cmd = [exe, str(ROOT / "main.py"), str(mp4), "--managed",
                   "--mirror", str(mirror)]

        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        w = Window(index, mp4, idx, proc)
        self.windows.append(w)
        return w

    # ------------------------------------------------------------------
    def measure(self, step_no):
        """Снимает срез по всем живым окнам."""
        rows = []
        total_rss = 0.0
        for w in self.windows:
            if not w.alive():
                rows.append({"pid": w.pid, "file": w.mp4.name, "dead": True})
                continue
            try:
                p = psutil.Process(w.pid)
                with p.oneshot():
                    rss = p.memory_info().rss / 1048576
                    thr = p.num_threads()
                    hnd = p.num_handles() if hasattr(p, "num_handles") else -1
                total_rss += rss
                rows.append({"pid": w.pid, "file": w.mp4.name, "dead": False,
                             "rss_mb": round(rss, 1), "threads": thr,
                             "handles": hnd})
            except psutil.NoSuchProcess:
                rows.append({"pid": w.pid, "file": w.mp4.name, "dead": True})

        vm = psutil.virtual_memory()
        step = {
            "step": step_no,
            "windows": sum(1 for r in rows if not r.get("dead")),
            "rows": rows,
            "total_rss_mb": round(total_rss, 1),
            "system_avail_mb": round(vm.available / 1048576, 1),
            "system_percent": vm.percent,
        }
        self.steps.append(step)

        for r in rows:
            self._wr.writerow([step_no, step["windows"], r["pid"], r["file"],
                               r.get("rss_mb", ""), r.get("threads", ""),
                               r.get("handles", ""), step["total_rss_mb"],
                               step["system_avail_mb"]])
        self._csv.flush()
        return step

    @staticmethod
    def print_step(step, prev_total):
        print(f"\n--- после запуска окна {step['windows']} ---")
        for r in step["rows"]:
            if r.get("dead"):
                print(f"  pid {r['pid']}: ПРОЦЕСС ЗАВЕРШИЛСЯ ({r['file']})")
            else:
                h = f", handles {r['handles']}" if r["handles"] >= 0 else ""
                print(f"  pid {r['pid']}: {r['rss_mb']:.0f} МБ, "
                      f"потоков {r['threads']}{h}  [{r['file']}]")
        delta = step["total_rss_mb"] - prev_total
        print(f"  ИТОГО: {step['total_rss_mb']:.0f} МБ "
              f"(+{delta:.0f} МБ за это окно)")
        print(f"  свободно в системе: {step['system_avail_mb']:.0f} МБ "
              f"(занято {step['system_percent']:.0f}%)")

    # ------------------------------------------------------------------
    def run(self, prepared):
        print(f"Базовый замер до запуска окон...")
        base = self.measure(0)
        print(f"  свободно в системе: {base['system_avail_mb']:.0f} МБ")

        prev_total = 0.0
        for i, (mp4, idx, mirror) in enumerate(prepared, start=1):
            print(f"\n[{i}/{len(prepared)}] Запуск окна: {mp4.name}")
            self.spawn_window(i, mp4, idx, mirror)

            # Даём окну инициализироваться и наполнить буферы: без паузы
            # замер поймает процесс на середине старта и покажет заниженную
            # цифру.
            for remaining in range(int(self.args.settle), 0, -5):
                time.sleep(min(5, remaining))
                sys.stdout.write(f"\r  ожидание стабилизации: {remaining} с   ")
                sys.stdout.flush()
            print("\r" + " " * 40 + "\r", end="")

            step = self.measure(i)
            self.print_step(step, prev_total)
            prev_total = step["total_rss_mb"]

            if step["system_avail_mb"] < self.args.min_free:
                print(f"\n  ОСТАНОВКА: свободной памяти меньше "
                      f"{self.args.min_free} МБ — дальше запускать опасно.")
                break

        if self.args.hold:
            print(f"\nУдержание окон {self.args.hold} с (проверка на утечку)...")
            end = time.monotonic() + self.args.hold
            while time.monotonic() < end:
                time.sleep(min(30, max(1, end - time.monotonic())))
                s = self.measure(len(self.steps))
                print(f"  {int(end - time.monotonic()):>5} с до конца: "
                      f"{s['total_rss_mb']:.0f} МБ")

    # ------------------------------------------------------------------
    def stop_all(self):
        print("\nЗакрытие окон...")
        for w in self.windows:
            if w.alive():
                try:
                    w.proc.terminate()
                except Exception:
                    pass
        time.sleep(2)
        for w in self.windows:
            if w.alive():
                try:
                    w.proc.kill()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    def write_report(self, prepared):
        L = []
        add = L.append
        add("=" * 74)
        add("НАГРУЗОЧНЫЙ ТЕСТ: несколько окон плеера одновременно")
        add("=" * 74)
        add(f"Начало:  {self.started.strftime('%Y-%m-%d %H:%M:%S')}")
        add(f"Окон запланировано: {len(prepared)}")
        add(f"Пауза на стабилизацию: {self.args.settle} с")
        add("")

        add("-- Файлы " + "-" * 63)
        for mp4, idx, _m in prepared:
            add(f"  {mp4.name}")
            add(f"    mp4 {mp4.stat().st_size / 1048576:8.1f} МБ   "
                f"idx {idx.stat().st_size / 1048576:7.1f} МБ")
        add("")

        # Только шаги запуска окон: шаги фазы удержания (там step
        # повторяется) в расчёт цены окна попадать не должны — иначе
        # нулевые приросты занижают среднее и создают ложный «разброс».
        real = []
        seen_steps = set()
        for s in self.steps:
            if s["step"] <= 0 or s["step"] in seen_steps:
                continue
            seen_steps.add(s["step"])
            real.append(s)
        add("-- Потребление по шагам " + "-" * 48)
        add(f"  {'окон':>5} {'суммарно МБ':>12} {'прирост МБ':>11} "
            f"{'свободно МБ':>12}")
        prev = 0.0
        per_window = []
        for s in real:
            delta = s["total_rss_mb"] - prev
            if s["windows"] > 0:
                per_window.append(delta)
            add(f"  {s['windows']:>5} {s['total_rss_mb']:>12.0f} "
                f"{delta:>11.0f} {s['system_avail_mb']:>12.0f}")
            prev = s["total_rss_mb"]
        add("")

        if per_window:
            avg = sum(per_window) / len(per_window)
            add("-- Анализ " + "-" * 62)
            add(f"  средняя цена одного окна: {avg:.0f} МБ")
            add(f"  минимум/максимум:         {min(per_window):.0f} / "
                f"{max(per_window):.0f} МБ")

            # Прогноз: сколько окон поместится в оставшуюся память.
            if real:
                free = real[-1]["system_avail_mb"]
                if avg > 0:
                    fit = int(free / avg)
                    add(f"  в оставшуюся память ({free:.0f} МБ) поместится "
                        f"ещё примерно {fit} окон")
            add("")

            # Разброс цены окна показывает, зависит ли она от файла.
            spread = max(per_window) - min(per_window)
            if spread > avg * 0.4:
                add("  ВНИМАНИЕ: цена окна сильно зависит от файла "
                    f"(разброс {spread:.0f} МБ).")
                add("  Это ожидаемо, если файлы разной длины: массивы индекса")
                add("  каждый процесс держит целиком, и их вес растёт линейно")
                add("  с длительностью записи.")
            else:
                add("  Цена окна примерно одинакова для всех файлов.")
            add("")

        # Утечка: сравниваем последний шаг с первым замером удержания.
        # Фаза удержания: повторные замеры с тем же номером шага.
        last_step = max((s["step"] for s in self.steps), default=0)
        hold_steps = [s for s in self.steps if s["step"] == last_step]
        if len(hold_steps) > 2:
            growth = hold_steps[-1]["total_rss_mb"] - hold_steps[0]["total_rss_mb"]
            add("-- Проверка на утечку (удержание) " + "-" * 38)
            add(f"  за {self.args.hold} с потребление изменилось на {growth:+.0f} МБ")
            if growth > 200:
                add("  ВНИМАНИЕ: память продолжает расти при бездействии —")
                add("  похоже на утечку, а не на заполнение буферов.")
            else:
                add("  Роста нет: потребление вышло на плато.")
            add("")

        dead = [r for s in self.steps for r in s["rows"] if r.get("dead")]
        if dead:
            add("-- Аварийно завершившиеся окна " + "-" * 41)
            seen = set()
            for r in dead:
                if r["pid"] in seen:
                    continue
                seen.add(r["pid"])
                add(f"  pid {r['pid']}: {r['file']}")
            add("  Смотреть crash_managed_<pid>.log и telemetry_*_ERRORS.log")
            add("")

        add("-- Что смотреть дальше " + "-" * 49)
        add("  Отчёты телеметрии каждого окна (telemetry_managed_<pid>.txt)")
        add("  содержат разбивку памяти: индекс, видеобуфер, seek-воркеры.")
        add("  Если суммарный вес растёт в основном за счёт индекса —")
        add("  это ожидаемо: каждый процесс держит СВОЮ полную копию")
        add("  массивов .idx, и на многочасовых файлах они доминируют.")

        self.report_path.write_text("\n".join(L), encoding="utf-8")
        print(f"\nОтчёт: {self.report_path}")
        print(f"CSV:   {self.csv_path}")

    def close(self):
        try:
            self._csv.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(
        description="Нагрузочный тест ProxyPlayer: несколько окон одновременно")
    ap.add_argument("media", nargs="+", help="пути к MP4 или числовые ID")
    ap.add_argument("--repeat", type=int, default=0,
                    help="повторить указанный файл N раз (если задан один файл)")
    ap.add_argument("--settle", type=float, default=45.0,
                    help="секунд ожидания стабилизации после запуска окна")
    ap.add_argument("--hold", type=float, default=0,
                    help="подержать все окна N секунд в конце (проверка утечки)")
    ap.add_argument("--min-free", type=float, default=1024,
                    help="остановиться, если свободной памяти меньше N МБ")
    ap.add_argument("--outdir", default=None, help="куда писать отчёты")
    args = ap.parse_args()

    raw = list(args.media)
    if args.repeat and len(raw) == 1:
        raw = raw * args.repeat

    test = LoadTest(args)
    stopping = {"flag": False}

    def _stop(signum, frame):
        stopping["flag"] = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _stop)

    prepared = []
    try:
        prepared = test.prepare(raw)
        test.run(prepared)
    except KeyboardInterrupt:
        print("\nПрервано пользователем")
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        test.stop_all()
        try:
            test.write_report(prepared)
        except Exception:
            import traceback
            traceback.print_exc()
        test.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
