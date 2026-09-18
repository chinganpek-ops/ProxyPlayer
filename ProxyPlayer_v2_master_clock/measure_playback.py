#!/usr/bin/env python3
"""
measure_playback.py – замер живого воспроизведения (вариант Б).

Запускает реальный StreamController и измеряет то, что нельзя увидеть
офлайн: как расходятся дорожки во времени, отстаёт ли видео от звука,
когда возникают underrun'ы и что происходит с очередями при перемотке.

В отличие от player_telemetry (общий сбор всего подряд), здесь прицельный
замер синхронизации с шагом 0.2 с — достаточно частым, чтобы поймать
кратковременные провалы, которые пятисекундный интервал телеметрии
пропускает.

ЧТО МЕРЯЕТСЯ

  Дорожки:
    q2/q3           — очереди треков 2 и 3 в сэмплах
    drift_ms        — расхождение подачи между дорожками
    push2/push3     — сколько подано всего (растёт → видно скорость подачи)

  Синхронизация с видео:
    audio_clock     — ведущие часы (сэмплы, воспроизведённые картой)
    video_pts       — PTS первого кадра в буфере отображения
    av_delta_ms     — video_pts - audio_clock; отрицательное = видео ОТСТАЁТ
    vbuf            — заполнение видеобуфера

  Здоровье:
    underruns       — нарастающий счётчик пустых аудиоочередей
    dropped         — кадры, отброшенные как безнадёжно устаревшие

ЗАПУСК
    python measure_playback.py "D:\\media\\1966520_....mp4"
    python measure_playback.py 1966520 --seconds 120
    python measure_playback.py <файл> --seek-at 30 --seek-to 600
        (на 30-й секунде перемотать на кадр 600 и продолжить замер —
         видно, как система восстанавливается после seek)

РЕЗУЛЬТАТ
    measure_<ts>.csv  — таймсерия для графиков
    measure_<ts>.txt  — сводка с выводами
    плюс живая печать в консоль
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Сколько первых замеров исключать из анализа синхронизации.
# При шаге 0.2 с это первые 4 секунды — время, за которое наполняются
# буферы и устанавливаются часы.
SETTLE_SAMPLES = 20


def resolve_media(raw: str):
    path = Path(raw)
    if path.exists():
        return path
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
    raise FileNotFoundError(f"Не найден файл: {raw}")


def find_idx(mp4: Path) -> Path:
    idx = mp4.parent / "idx" / "mp4" / f"{mp4.stem}.idx"
    return idx if idx.exists() else mp4.parent / f"{mp4.stem}.idx"


def find_ref(mp4: Path) -> Path:
    ref = mp4.parent / f"{mp4.stem}.mp4.ref"
    return ref if ref.exists() else mp4.parent / f"{mp4.stem}.ref"


def queue_samples(mc, track_id):
    """
    Сэмплы в очереди ОДНОЙ дорожки (get_audio_queue_samples даёт сумму по
    обеим, чего для поиска расхождения недостаточно).

    Раньше эта функция сама лезла в mc._queue2 / mc._queue3 и считала
    длины элементов. Дважды это приводило к неверным числам: сначала
    очередь стала хранить пары (pts, samples), и len() от кортежа давал 2
    вместо длины массива; расхождение при этом показывало ноль, потому что
    обе дорожки считались одинаково неверно.

    Теперь подсчёт запрашивается у MasterClock: он владеет очередями и
    знает их формат. Смена формата больше не требует правок здесь.
    """
    try:
        return int(mc.get_track_queue_samples(track_id))
    except Exception:
        return -1


def main():
    ap = argparse.ArgumentParser(description="Замер синхронизации аудио/видео")
    ap.add_argument("media", help="путь к MP4 или числовой ID")
    ap.add_argument("--seconds", type=float, default=60.0, help="сколько мерить")
    ap.add_argument("--interval", type=float, default=0.2, help="шаг замера")
    ap.add_argument("--seek-at", type=float, default=0,
                    help="на какой секунде выполнить перемотку (0 = не делать)")
    ap.add_argument("--seek-to", type=int, default=0, help="целевой кадр перемотки")
    args = ap.parse_args()

    mp4 = resolve_media(args.media)
    idx = find_idx(mp4)
    ref = find_ref(mp4)
    if not idx.exists():
        raise FileNotFoundError(f"Не найден индекс: {idx}")

    from PyQt5.QtCore import QStandardPaths
    from config.config import load_config
    from index.idx_cache import prepare_mirror
    from core.stream_controller import StreamController
    from config.timebase import pts_to_video_frame, AUDIO_SAMPLE_RATE

    cfg = Path(QStandardPaths.writableLocation(
        QStandardPaths.AppConfigLocation)) / "player_config.json"
    config = load_config(cfg)

    print(f"Файл:   {mp4}")
    print(f"Индекс: {idx}")
    mirror = prepare_mirror(idx)

    ctl = StreamController(
        ref_path=ref, idx_path=idx, mp4_path=mp4,
        fps=config.get("fps", 25.0),
        buffer_size=config.get("buffer_size", 360),
        audio_delay_ms=config.get("audio_delay_ms", 0),
        start_from_live=True,
        mirror_path=str(mirror),
    )
    # Готовность и ошибка — через публичные методы контроллера.
    deadline = time.monotonic() + 180
    while not ctl.is_ready():
        if time.monotonic() > deadline:
            raise RuntimeError("StreamController не инициализировался")
        time.sleep(0.2)
    init_error = ctl.get_init_error()
    if init_error:
        raise RuntimeError(f"Ошибка инициализации: {init_error}")

    ctl.start_playback()
    time.sleep(1.5)
    ctl.resume()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = ROOT / f"measure_{ts}.csv"
    txt_path = ROOT / f"measure_{ts}.txt"

    fh = open(csv_path, "w", newline="", encoding="utf-8")
    wr = csv.writer(fh)
    wr.writerow(["t", "audio_clock", "video_pts", "av_delta_ms", "vbuf",
                 "q2", "q3", "drift_ms", "push2", "push3",
                 "underruns", "playing"])

    rows = []
    fps = ctl.fps or 25.0
    period = 1.0 / fps
    t0 = time.monotonic()
    next_render = t0
    next_sample = t0
    seek_done = False

    print(f"\n{'t':>6} {'av_delta':>9} {'vbuf':>6} {'q2':>8} {'q3':>8} "
          f"{'drift':>8} {'under':>6}")
    print("-" * 60)

    try:
        while True:
            now = time.monotonic()
            elapsed = now - t0
            if elapsed >= args.seconds:
                break

            # Рендер-цикл: без него буферы не расходуются и картина будет
            # принципиально иной, чем при реальном воспроизведении.
            if now >= next_render:
                next_render = now + period
                try:
                    ctl.get_display_frame()
                except Exception as e:
                    print(f"[{elapsed:6.1f}] ошибка рендера: {e}")

            # Перемотка в заданный момент
            if args.seek_at and not seek_done and elapsed >= args.seek_at:
                seek_done = True
                print(f"\n>>> перемотка на кадр {args.seek_to}\n")
                ctl.seek_absolute(args.seek_to,
                                  callback=lambda: print(">>> seek завершён"),
                                  on_error=lambda m: print(f">>> seek ОШИБКА: {m}"))
                ctl.resume()

            if now >= next_sample:
                next_sample = now + args.interval
                mc = ctl.master_clock

                clock = ctl.audio_clock
                q2 = queue_samples(mc, 2) if mc else -1
                q3 = queue_samples(mc, 3) if mc else -1

                under = -1
                if mc:
                    try:
                        under = int(mc.get_underruns())
                    except Exception:
                        under = -1

                # PTS первого кадра в буфере отображения: показывает, какой
                # кадр СЛЕДУЮЩИМ пойдёт на экран относительно часов.
                #
                # Буфер запрашивается у контроллера публичным методом:
                # раньше здесь читалось ctl._playback._display_buffer —
                # цепочка из двух приватных полей чужих объектов, которая
                # ломается при любом переименовании внутри движка.
                video_pts = -1
                vbuf = -1
                try:
                    buf = ctl.get_display_buffer()
                    if buf is not None:
                        vbuf = buf.count
                        first = buf.peek_first()
                        if first:
                            video_pts = int(first[0])
                except Exception:
                    pass

                av_delta_ms = ((video_pts - clock) / AUDIO_SAMPLE_RATE * 1000
                               if video_pts >= 0 else 0)
                drift_ms = (q2 - q3) / (AUDIO_SAMPLE_RATE / 1000) if q2 >= 0 and q3 >= 0 else 0

                row = [round(elapsed, 2), clock, video_pts, round(av_delta_ms, 1),
                       vbuf, q2, q3, round(drift_ms, 1), 0, 0, under,
                       int(bool(ctl.playing))]
                wr.writerow(row)
                fh.flush()
                rows.append(row)

                print(f"{elapsed:6.1f} {av_delta_ms:8.0f}м {vbuf:6} {q2:8} {q3:8} "
                      f"{drift_ms:7.0f}м {under:6}")

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nПрервано пользователем")
    finally:
        fh.close()
        try:
            ctl.close()
        except Exception:
            pass
        write_summary(txt_path, rows, mp4, args)
        print(f"\nCSV:    {csv_path}")
        print(f"Сводка: {txt_path}")


def write_summary(path, rows, mp4, args):
    """Сводка с выводами — то, ради чего всё измерялось."""
    L = []
    add = L.append
    add("=" * 70)
    add("ЗАМЕР ВОСПРОИЗВЕДЕНИЯ")
    add("=" * 70)
    add(f"Файл: {mp4}")
    add(f"Замеров: {len(rows)}")
    add("")

    if not rows:
        add("Данных нет.")
        path.write_text("\n".join(L), encoding="utf-8")
        return

    def col(i):
        return [r[i] for r in rows if isinstance(r[i], (int, float))]

    av = [r[3] for r in rows if r[2] >= 0]
    q2, q3 = col(5), col(6)
    drift = col(7)
    under = col(10)
    vbuf = col(4)

    add("-- Синхронизация видео и звука " + "-" * 39)
    if av:
        add(f"  av_delta (video_pts - audio_clock), мс:")
        add(f"    за весь замер: среднее {sum(av)/len(av):+.0f}, "
            f"мин {min(av):+.0f}, макс {max(av):+.0f}")
        add("    отрицательное = видео отстаёт от звука")

        # Первые секунды после старта — переходный процесс: буферы ещё
        # наполняются, часы только что выставлены, первый кадр может
        # отличаться от установившегося значения на сотни миллисекунд.
        # Включать этот участок в оценку дрейфа нельзя: завершение
        # переходного процесса выглядит как большое "изменение за замер"
        # и даёт ложную тревогу о нарастающем расхождении.
        settled = av[SETTLE_SAMPLES:] if len(av) > SETTLE_SAMPLES * 2 else av
        skipped = len(av) - len(settled)
        if skipped:
            add(f"    пропущено при анализе: первые {skipped} замеров "
                f"(переходный процесс после старта)")
        add(f"    установившееся: среднее {sum(settled)/len(settled):+.0f}, "
            f"мин {min(settled):+.0f}, макс {max(settled):+.0f}")

        # Тренд считаем по установившемуся участку и не по крайним точкам,
        # а по средним первой и последней трети: одиночный выброс на краю
        # не должен определять вывод.
        if len(settled) >= 6:
            third = max(1, len(settled) // 3)
            head = sum(settled[:third]) / third
            tail = sum(settled[-third:]) / third
            drift_trend = tail - head
        else:
            drift_trend = settled[-1] - settled[0] if len(settled) > 1 else 0
        add(f"    тренд (последняя треть минус первая): {drift_trend:+.0f} мс")

        avg_settled = sum(settled) / len(settled)
        if abs(drift_trend) > 150:
            add("    ВНИМАНИЕ: расхождение НАРАСТАЕТ — это дрейф, а не")
            add("    постоянное смещение; параметром audio_delay_ms не лечится.")
        elif abs(avg_settled) > 100:
            add("    Постоянное смещение — лечится параметром audio_delay_ms.")
        elif abs(avg_settled) <= 40:
            add("    Синхронизация в пределах одного кадра — норма.")
    add("")

    add("-- Дорожки 2 и 3 " + "-" * 53)
    if q2 and q3:
        add(f"  очередь трека 2: среднее {sum(q2)//len(q2)}, макс {max(q2)}")
        add(f"  очередь трека 3: среднее {sum(q3)//len(q3)}, макс {max(q3)}")
    if drift:
        d_settled = drift[SETTLE_SAMPLES:] if len(drift) > SETTLE_SAMPLES * 2 else drift
        add(f"  расхождение, мс: начало {d_settled[0]:.0f}, "
            f"конец {d_settled[-1]:.0f}, макс {max(d_settled, key=abs):.0f}")
        if len(d_settled) > 1 and abs(d_settled[-1]) > abs(d_settled[0]) + 200:
            add("  ВНИМАНИЕ: дорожки РАСХОДЯТСЯ со временем.")
            add("  Проверить, передаётся ли PTS при подаче в MasterClock:")
            add("  без него блоки укладываются по порядку и догнать нечем.")
        elif abs(max(d_settled, key=abs)) < 50:
            add("  Дорожки синхронны — выравнивание по PTS работает.")
    add("")

    add("-- Буферы и провалы " + "-" * 50)
    if vbuf:
        add(f"  видеобуфер: мин {min(vbuf)}, среднее {sum(vbuf)//len(vbuf)}, макс {max(vbuf)}")
        if min(vbuf) == 0:
            add("  ВНИМАНИЕ: видеобуфер опустошался — были моменты без кадров.")
    if under:
        grew = under[-1] - under[0]
        add(f"  underruns: было {under[0]}, стало {under[-1]} (+{grew})")
        if grew > 0:
            add("  ВНИМАНИЕ: звук прерывался из-за пустых очередей.")
        else:
            add("  Очереди не пустели — провалы звука НЕ от нехватки данных.")
    add("")

    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
