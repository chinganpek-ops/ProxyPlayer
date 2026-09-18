#!/usr/bin/env python3
"""
verify_idx_fields.py – проверка гипотез о полях .idx на реальных данных.

ЗАЧЕМ
Статистический анализ показал, что формат построен на 16.16 fixed-point:
f11 = номер_кадра × 65536, f16 у видео = длительность в кадрах,
f8 = тип << 16 | номер_дорожки. Арифметика сходится точно, но проверена
только на одном файле и только по статистике.

Менять рабочий код по такой гипотезе нельзя. Этот скрипт проверяет её
ФАКТАМИ: декодирует кадры, сравнивает вычисленные длины с заявленными,
сверяет номера дорожек.

ЧТО ПРОВЕРЯЕТСЯ

  1. f7 — какие значения действительно являются опорными кадрами.
     Каждое значение f7 проверяется декодированием БЕЗ предшествующих
     кадров: если картинка получилась — кадр самодостаточен, то есть
     опорный. Это прямая проверка, а не рассуждение о номерах NAL.
     Текущий код считает опорными 27..30 — если найдутся другие,
     перемотка сейчас иногда стартует не с той точки.

  2. f16 — совпадает ли заявленная длительность с фактической.
     Сравнивается f16/65536 с расстоянием до следующего сэмпла.
     Если сходится, длину сэмпла можно брать из поля, а не вычислять —
     тогда размер последнего сэмпла окна перестанет зависеть от
     mdat_end, который приходится обновлять при росте файла.

  3. f8 — верна ли трактовка составного поля.
     Проверяется, что старшие 16 бит постоянны и равны сигнатуре типа,
     а младшие дают тот же номер дорожки, что текущий f8 & 0xFF.

ЗАПУСК
    python verify_idx_fields.py <файл.mp4>
    python verify_idx_fields.py <файл.idx>
    python verify_idx_fields.py <файл> --samples 40    # кадров на значение f7
    python verify_idx_fields.py <файл> --no-decode     # без чтения MP4

Скрипт ничего не меняет: только читает и печатает выводы.
"""

import argparse
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

FIXED_ONE = 65536          # 16.16: единица
VIDEO_SIG = 0x193
AUDIO_SIG = 0xC9

# Диапазон, который текущий код считает опорными кадрами.
CURRENT_IDR_RANGE = range(27, 31)


class Report:
    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text)
        self.lines.append(text)

    def save(self, path):
        path.write_text("\n".join(self.lines), encoding="utf-8")
        print(f"\nОтчёт: {path}")


def resolve(raw: str):
    """Возвращает (mp4, idx) по любому из двух путей."""
    p = Path(raw).resolve()
    if p.suffix.lower() == ".idx":
        stem = p.stem
        for cand in (p.parent.parent.parent / f"{stem}.mp4",
                     p.parent / f"{stem}.mp4"):
            if cand.exists():
                return cand, p
        return None, p
    idx = p.parent / "idx" / "mp4" / f"{p.stem}.idx"
    if not idx.exists():
        idx = p.parent / f"{p.stem}.idx"
    return p, idx


def load_records(idx_path: Path, signature: int, limit: int = 0):
    """
    Читает записи заданного типа с привязкой к сетке.

    Сетка обязательна: значение сигнатуры встречается и как обычные
    данные (номер кадра 403 совпадает с 0x193), и без привязки все поля
    читались бы со смещением.
    """
    words = np.fromfile(idx_path, dtype="<u4")
    hits = np.concatenate([np.where(words == VIDEO_SIG)[0],
                           np.where(words == AUDIO_SIG)[0]])
    if not len(hits):
        return np.empty((0, 17), dtype=np.uint32), words
    grid = Counter(int(h) % 17 for h in hits).most_common(1)[0][0]

    pos = np.where(words == signature)[0]
    pos = pos[(pos % 17 == grid) & (pos + 17 <= len(words))]
    if limit and len(pos) > limit:
        pos = pos[:limit]
    if not len(pos):
        return np.empty((0, 17), dtype=np.uint32), words
    recs = words[pos[:, None] + np.arange(17)]
    return recs, words


def abs_offset(rec_col1, rec_col2):
    """Абсолютное смещение: f1 + f2 × 4ГБ − 4 (как в moov_builder)."""
    return rec_col1.astype(np.int64) + rec_col2.astype(np.int64) * 4294967296 - 4


# ===========================================================================
# 1. Какие значения f7 являются опорными кадрами
# ===========================================================================
def _decode_run(mp4, avcc, offsets, start_idx, count, Decoder, reader):
    """
    Декодирует последовательность кадров, начиная с указанного.

    Возвращает число полученных кадров. Декодер создаётся заново на
    каждый прогон: проверяется именно то, можно ли НАЧАТЬ декодирование
    с этого кадра, а не продолжить уже идущее.

    Подаётся несколько кадров подряд, а не один. Причина: декодер
    буферизует и на первый пакет обычно не возвращает ничего — кадр
    выходит со следующими. Ранняя версия проверки подавала один пакет и
    получала ноль кадров ДЛЯ ВСЕХ значений поля, включая заведомо
    опорные; вывод «опорных кадров нет» был следствием этой ошибки, а не
    свойством данных. Реальная перемотка тоже декодирует от опорного
    кадра вперёд, так что последовательность — ещё и более честная
    модель происходящего.
    """
    produced = 0
    dec = Decoder(avcc, mp4, thread_type="AUTO", thread_count=1,
                  skip_frame=False, gpu_mode="off")
    try:
        for k in range(count):
            i = start_idx + k
            if i + 1 >= len(offsets):
                break
            begin = int(offsets[i])
            size = int(offsets[i + 1]) - begin
            if size <= 0 or size > 4_000_000:
                break
            reader.seek(begin)
            data = reader.read(size)
            if not data:
                break
            filtered = dec.filter_avcc(data)
            if not filtered:
                continue
            produced += len(dec.decode_sample(filtered))
    except Exception:
        pass
    finally:
        try:
            dec.close()
        except Exception:
            pass
    return produced


def check_idr(out, mp4, recs, samples_per_value, seq_len=6):
    out("")
    out("=" * 76)
    out("1. ОПОРНЫЕ КАДРЫ: с каких значений f7 можно НАЧАТЬ декодирование")
    out("=" * 76)

    if mp4 is None or not mp4.exists():
        out("  MP4 недоступен — проверка декодированием невозможна.")
        return

    try:
        from decode.decoder import Decoder
        from file_io.ref_parser import extract_ftyp_avcc
        from file_io.win_sequential_reader import WinSequentialReader
    except ImportError as e:
        out(f"  Модули плеера недоступны ({e}) — пропущено.")
        return

    ref = mp4.parent / f"{mp4.stem}.mp4.ref"
    if not ref.exists():
        ref = mp4.parent / f"{mp4.stem}.ref"
    try:
        _, avcc = extract_ftyp_avcc(ref)
    except Exception as e:
        out(f"  Не удалось прочитать .ref ({e}) — пропущено.")
        return

    f7 = recs[:, 7]
    offsets = abs_offset(recs[:, 1], recs[:, 2])
    values = sorted(Counter(int(v) for v in f7).items(), key=lambda x: x[0])

    out(f"  Значений f7: {len(values)}. Проверяется до {samples_per_value} точек")
    out(f"  на значение, с каждой декодируется {seq_len} кадров подряд.")
    out("")
    out(f"  {'f7':>4} {'всего':>8} {'точек':>7} {'кадров/точку':>13} "
        f"{'вывод':<20} сейчас в коде")
    out("  " + "-" * 82)

    reader = WinSequentialReader(mp4)
    results = {}
    per_value_rate = {}
    try:
        for value, total in values:
            idx = np.where(f7 == value)[0]
            step = max(1, len(idx) // samples_per_value)
            chosen = [int(x) for x in idx[::step][:samples_per_value]]

            counts = []
            for i in chosen:
                counts.append(_decode_run(mp4, avcc, offsets, i, seq_len,
                                          Decoder, reader))
            if not counts:
                results[value] = "не проверено"
                out(f"  {value:>4} {total:>8} {0:>7} {'-':>13} "
                    f"{'не проверено':<20}")
                continue

            avg = sum(counts) / len(counts)
            per_value_rate[value] = avg
            if avg >= seq_len * 0.6:
                verdict = "ОПОРНЫЙ"
            elif avg == 0:
                verdict = "зависимый"
            else:
                verdict = f"частично ({avg:.1f})"
            results[value] = verdict

            in_code = "считается опорным" if value in CURRENT_IDR_RANGE else ""
            out(f"  {value:>4} {total:>8} {len(counts):>7} {avg:>13.1f} "
                f"{verdict:<20} {in_code}")

        # --- САМОПРОВЕРКА МЕТОДА ---
        #
        # Если декодирование успешно начинается С ЛЮБОГО кадра, проверка
        # ничего не различает и её выводам нельзя верить. Берём точки,
        # заведомо находящиеся в середине группы (сразу после кандидата),
        # и смотрим, отличается ли результат.
        out("")
        out("  -- Самопроверка метода " + "-" * 50)
        best = max(per_value_rate, key=per_value_rate.get) if per_value_rate else None
        if best is None:
            out("  Нет данных для самопроверки.")
        else:
            idx = np.where(f7 == best)[0]
            step = max(1, len(idx) // samples_per_value)
            mid_counts = []
            for i in idx[::step][:samples_per_value]:
                # +2 кадра от опорного — гарантированно середина группы
                mid_counts.append(_decode_run(mp4, avcc, offsets, int(i) + 2,
                                              seq_len, Decoder, reader))
            mid_avg = sum(mid_counts) / len(mid_counts) if mid_counts else 0
            out(f"  Лучшее значение f7={best}: {per_value_rate[best]:.1f} кадров")
            out(f"  Старт из середины группы (+2 кадра): {mid_avg:.1f} кадров")
            if mid_avg >= per_value_rate[best] * 0.8:
                out("")
                out("  ВНИМАНИЕ: декодирование одинаково успешно начинается")
                out("  и с опорного кадра, и из середины группы. Значит метод")
                out("  НЕ РАЗЛИЧАЕТ их, и таблица выше ничего не доказывает.")
                out("  Вероятно, декодер восстанавливается по SPS/PPS в потоке.")
                out("  Выводы делать нельзя — менять код по ним тем более.")
                results = {}
            else:
                out("  Метод различает опорные кадры и середину группы:")
                out("  таблице выше можно доверять.")
    finally:
        reader.close()

    if not results:
        return

    out("")
    missed = [v for v, r in results.items()
              if r == "ОПОРНЫЙ" and v not in CURRENT_IDR_RANGE]
    wrong = [v for v, r in results.items()
             if r == "зависимый" and v in CURRENT_IDR_RANGE]

    if missed:
        out("  НАЙДЕНО: опорные кадры, которые код НЕ считает опорными:")
        out(f"    f7 = {', '.join(map(str, missed))}")
        out("    Перемотка начинает декодирование дальше, чем могла бы.")
    if wrong:
        out("  НАЙДЕНО: код считает опорными значения, которые таковыми не являются:")
        out(f"    f7 = {', '.join(map(str, wrong))}")
        out("    Перемотка может стартовать с недекодируемого кадра.")
    if not missed and not wrong:
        out("  Диапазон 27..30 в коде соответствует данным. Менять не нужно.")


# ===========================================================================
# 2. f16 против фактической длины
# ===========================================================================
def check_duration(out, recs):
    out("")
    out("=" * 76)
    out("2. ДЛИТЕЛЬНОСТЬ: совпадает ли f16 с фактическим расстоянием")
    out("=" * 76)

    if len(recs) < 3:
        out("  Недостаточно записей.")
        return

    f16 = recs[:, 16].astype(np.int64)
    f3 = recs[:, 3].astype(np.int64)

    declared = f16 / FIXED_ONE          # длительность в кадрах
    actual = np.diff(f3)                # фактический шаг номера кадра

    out(f"  f16/65536: мин {declared.min():.6f}, макс {declared.max():.6f}")
    cnt = Counter(round(float(x), 6) for x in declared[:100000])
    out("  частые значения: " + ", ".join(
        f"{v}×{c}" for v, c in cnt.most_common(4)))
    out("")

    same = int((np.abs(declared[:-1] - actual) < 0.001).sum())
    total = len(actual)
    out(f"  Совпадает с фактическим шагом: {same} из {total} "
        f"({100.0 * same / total:.2f}%)")

    if same / total > 0.99:
        out("  ВЫВОД: f16 достоверно описывает длительность сэмпла.")
        out("    Длину можно брать из поля, а не вычислять как расстояние")
        out("    до следующей записи. Тогда размер последнего сэмпла окна")
        out("    перестанет зависеть от mdat_end, который приходится")
        out("    обновлять при росте файла.")
    else:
        out("  ВЫВОД: f16 НЕ совпадает с фактическим шагом.")
        out("    Оставить текущий расчёт через расстояние до следующей записи.")

    # --- проверка f11 = номер кадра × 65536 ---
    #
    # Сравнение ведётся ПО МОДУЛЮ 2^32. Поле 32-битное, а произведение
    # номера кадра на 65536 выходит за этот предел уже на кадре 65536 —
    # это 43 минуты при 25 к/с. Ранняя версия сравнивала без учёта
    # переполнения и на длинном файле давала ровно 65536 совпадений из
    # всех, то есть "не подтверждено" — хотя формат как раз подтверждался,
    # просто поле переполнялось.
    out("")
    f11 = recs[:, 11].astype(np.int64)
    expected = (f3 * FIXED_ONE) % (2 ** 32)
    match = int((f11 == expected).sum())
    share = match / len(f3)
    out(f"  f11 == (f3 × 65536) mod 2^32: {match} из {len(f3)} "
        f"({100.0 * share:.2f}%)")

    overflowed = int((f3 >= 2 ** 32 // FIXED_ONE).sum())
    if overflowed:
        out(f"    Записей с переполнением поля: {overflowed} "
            f"(кадры от {2 ** 32 // FIXED_ONE} и дальше)")

    if share > 0.99:
        out("    Подтверждено: формат использует 16.16 fixed-point.")
        if overflowed:
            out("    Но как источник номера кадра поле НЕПРИГОДНО: после")
            out(f"    кадра {2 ** 32 // FIXED_ONE} (около 43 минут при 25 к/с)")
            out("    оно начинает отсчёт заново. Использовать f3.")
    else:
        out("    Не подтверждено даже с учётом переполнения —")
        out("    трактовка 16.16 для этого поля под вопросом.")


# ===========================================================================
# 3. f8 как составное поле
# ===========================================================================
def check_track_field(out, recs, signature):
    out("")
    out("=" * 76)
    out("3. НОМЕР ДОРОЖКИ: трактовка составного поля f8")
    out("=" * 76)

    if not len(recs):
        out("  Записей нет.")
        return

    f8 = recs[:, 8].astype(np.int64)
    high = f8 >> 16
    low16 = f8 & 0xFFFF
    low8 = f8 & 0xFF

    uniq_high = Counter(int(x) for x in high)
    out(f"  Старшие 16 бит: {dict(list(uniq_high.items())[:6])}")
    if len(uniq_high) == 1:
        v = next(iter(uniq_high))
        out(f"    Постоянны и равны {v}" +
            (f" — совпадает с сигнатурой типа 0x{signature:X}"
             if v == signature else ""))
    else:
        out("    Не постоянны — возможно, кодируют подтип записи.")

    tracks16 = sorted(set(int(x) for x in low16))
    tracks8 = sorted(set(int(x) for x in low8))
    out("")
    out(f"  Дорожки по f8 & 0xFFFF: {tracks16}")
    out(f"  Дорожки по f8 & 0xFF:   {tracks8}   (так делает код сейчас)")

    if tracks16 == tracks8:
        out("")
        out("  ВЫВОД: оба способа дают одно и то же — номер укладывается")
        out("    в младший байт. Текущий код работает верно, но по")
        out("    случайности: при номере дорожки больше 255 он сломается.")
        out("    Смена на & 0xFFFF безопасна и делает намерение явным.")
    else:
        out("")
        out("  ВЫВОД: способы РАСХОДЯТСЯ — текущий код теряет часть дорожек.")
        out(f"    Пропускаются: {sorted(set(tracks16) - set(tracks8))}")

    # Сколько записей приходится на каждую дорожку
    out("")
    per_track = Counter(int(x) for x in low16)
    out("  Записей по дорожкам:")
    for t, c in sorted(per_track.items())[:20]:
        used = "  <- используется плеером" if t in (2, 3) else ""
        out(f"    дорожка {t:>3}: {c:>8}{used}")


def main():
    ap = argparse.ArgumentParser(
        description="Проверка гипотез о полях .idx на реальных данных")
    ap.add_argument("path", help="путь к .mp4 или .idx")
    ap.add_argument("--samples", type=int, default=25,
                    help="кадров на каждое значение f7 (по умолчанию 25)")
    ap.add_argument("--limit", type=int, default=100000,
                    help="сколько записей читать")
    ap.add_argument("--seq", type=int, default=6,
                    help="сколько кадров подряд декодировать с каждой точки")
    ap.add_argument("--no-decode", action="store_true",
                    help="без декодирования: только проверки полей")
    args = ap.parse_args()

    mp4, idx = resolve(args.path)
    if not idx.exists():
        print(f"Не найден индекс: {idx}", file=sys.stderr)
        return 2

    out = Report()
    out("=" * 76)
    out("ПРОВЕРКА ГИПОТЕЗ О ПОЛЯХ ИНДЕКСА")
    out("=" * 76)
    out(f"MP4: {mp4 if mp4 else '(недоступен)'}")
    out(f"IDX: {idx}")

    video, _ = load_records(idx, VIDEO_SIG, args.limit)
    audio, _ = load_records(idx, AUDIO_SIG, args.limit)
    out(f"Записей: видео {len(video)}, аудио {len(audio)}")

    if len(video):
        if not args.no_decode:
            check_idr(out, mp4, video, args.samples, args.seq)
        else:
            out("\n(декодирование пропущено: --no-decode)")
        check_duration(out, video)
    if len(audio):
        check_track_field(out, audio, AUDIO_SIG)

    out("")
    out("=" * 76)
    out("ЧТО ДЕЛАТЬ ПО РЕЗУЛЬТАТАМ")
    out("  Раздел 1 -> moov_builder.get_idr_indices_from_mmap(),")
    out("              chunk_pipeline.DemuxerStage (пометка is_idr)")
    out("  Раздел 2 -> moov_builder (длина сэмпла), lazy_index (mdat_end)")
    out("  Раздел 3 -> moov_builder.build_audio_tracks()")
    out("  Менять код стоит только там, где проверка дала однозначный ответ.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out.save(ROOT / f"verify_idx_{idx.stem[:30]}_{ts}.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
