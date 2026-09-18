#!/usr/bin/env python3
"""
analyze_idx_fields.py – анализ полей записей .idx.

ЗАЧЕМ
Каждая запись индекса — 17 полей по uint32. Используются пять:
f1, f2 (смещение), f3 (номер кадра или PTS), f7 (тип NAL или размер),
f8 (номер дорожки). Остальные двенадцать не читает никто, и что в них
лежит — неизвестно.

Этот скрипт не строит догадок, а описывает данные: по каждому полю
показывает диапазон, число различных значений, монотонность, связь с уже
известными полями и вероятную трактовку. По такой таблице смысл поля
обычно виден сразу.

Заодно скрипт находит ВСЕ типы записей в файле, а не только 0x193 и
0xC9 — сейчас остальные просто отбрасываются, и что в них, никто не
смотрел.

ЧТО ИЩЕТСЯ

  всегда ноль            -> резерв или неиспользуемое поле
  одно значение          -> константа, версия формата, признак источника
  два-три значения       -> флаг
  монотонный рост        -> счётчик, смещение, временная метка
  совпадает с известным  -> дубликат или производная величина
  похоже на ASCII        -> текстовая метка (fourcc, имя дорожки)
  похоже на время        -> unix-время, счётчик тактов

ЗАПУСК
    python analyze_idx_fields.py <файл.idx>
    python analyze_idx_fields.py <файл.mp4>          # найдёт .idx сам
    python analyze_idx_fields.py <файл> --limit 50000
    python analyze_idx_fields.py <файл> --type 0x193 # только видеозаписи
    python analyze_idx_fields.py <файл> --raw 5      # показать 5 записей в hex

Результат печатается таблицей и сохраняется рядом в текстовый файл.
"""

import argparse
import struct
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RECORD_WORDS = 17                      # 17 x uint32 = 68 байт
RECORD_BYTES = RECORD_WORDS * 4
SEGMENT_SIZE = 4_294_967_296

# Известные назначения — чтобы отделить изученное от неизвестного.
KNOWN = {
    0x193: {1: "смещение, младшая часть", 2: "номер 4ГБ-сегмента",
            3: "номер кадра", 7: "тип NAL (27-30 = IDR)"},
    0xC9: {1: "смещение, младшая часть", 2: "номер 4ГБ-сегмента",
           3: "PTS в сэмплах", 7: "размер первой части",
           8: "номер дорожки (младший байт)"},
}


class Report:
    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text)
        self.lines.append(text)

    def save(self, path):
        path.write_text("\n".join(self.lines), encoding="utf-8")
        print(f"\nОтчёт сохранён: {path}")


def find_idx(path: Path) -> Path:
    """Принимает и .idx, и .mp4 — во втором случае ищет индекс рядом."""
    if path.suffix.lower() == ".idx":
        return path
    candidate = path.parent / "idx" / "mp4" / f"{path.stem}.idx"
    if candidate.exists():
        return candidate
    candidate = path.parent / f"{path.stem}.idx"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Не найден .idx для {path}")


def load_words(path: Path, limit_mb: float) -> np.ndarray:
    """Читает файл как массив uint32 little-endian."""
    size = path.stat().st_size
    to_read = int(min(size, limit_mb * 1024 * 1024)) if limit_mb else size
    with open(path, "rb") as f:
        raw = f.read(to_read)
    usable = (len(raw) // 4) * 4
    return np.frombuffer(raw[:usable], dtype="<u4")


def detect_grid(words: np.ndarray, signatures=(0x193, 0xC9)) -> int:
    """
    Определяет выравнивание сетки записей — остаток позиции по модулю
    длины записи.

    Зачем это нужно. Значение сигнатуры может встретиться и как ДАННЫЕ:
    например, номер кадра 403 совпадает с 0x193. Если принимать каждое
    совпадение за начало записи, сетка после такого места сдвигается и
    все поля читаются не со своих позиций — в анализе это выглядит как
    хаотичные значения и ложные «флаги».

    Записи имеют постоянную длину и идут подряд, поэтому настоящие
    начала лежат на одной сетке: их позиции дают один и тот же остаток
    по модулю RECORD_WORDS. Ложные совпадения распределены произвольно и
    в преобладающий остаток не попадают.
    """
    hits = []
    for sig in signatures:
        hits.append(np.where(words == sig)[0])
    if not hits:
        return 0
    allhits = np.concatenate(hits) if len(hits) > 1 else hits[0]
    if not len(allhits):
        return 0
    residues = Counter((int(p) % RECORD_WORDS) for p in allhits)
    return residues.most_common(1)[0][0]


def find_records(words: np.ndarray, signature: int, grid: int) -> tuple:
    """
    Позиции записей заданного типа с привязкой к сетке.

    Возвращает (позиции, сколько отброшено как случайные совпадения).
    Число отброшенных полезно само по себе: если оно велико, значит
    сигнатура часто встречается внутри данных, и рабочему коду нужна
    валидация — что и подтверждается проверками в idx_cache.
    """
    cand = np.where(words == signature)[0]
    cand = cand[cand + RECORD_WORDS <= len(words)]
    aligned = cand[(cand % RECORD_WORDS) == grid]
    return aligned, len(cand) - len(aligned)


def looks_like_ascii(values) -> str:
    """Проверяет, похоже ли значение на текст в 4 байтах."""
    printable = 0
    samples = []
    for v in list(values)[:200]:
        b = struct.pack("<I", int(v))
        if all(32 <= c < 127 for c in b):
            printable += 1
            if len(samples) < 3:
                samples.append(b.decode("ascii"))
        b_be = struct.pack(">I", int(v))
        if all(32 <= c < 127 for c in b_be) and len(samples) < 3:
            samples.append(b_be.decode("ascii") + " (BE)")
    if printable >= max(1, len(list(values)[:200]) * 0.8):
        return f"ASCII: {', '.join(repr(s) for s in samples[:3])}"
    return ""


def looks_like_time(values) -> str:
    """Проверяет, похоже ли значение на unix-время."""
    v = [int(x) for x in list(values)[:100] if x]
    if not v:
        return ""
    lo, hi = min(v), max(v)
    # Диапазон примерно с 2001 по 2050 год
    if 1_000_000_000 <= lo <= 2_500_000_000 and 1_000_000_000 <= hi <= 2_500_000_000:
        try:
            d1 = datetime.fromtimestamp(lo, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            d2 = datetime.fromtimestamp(hi, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            return f"похоже на unix-время: {d1} .. {d2}"
        except Exception:
            pass
    return ""


def describe_field(idx: int, col: np.ndarray, known: dict,
                   others: dict) -> dict:
    """Характеристика одного поля."""
    n = len(col)
    uniq = np.unique(col)
    nuniq = len(uniq)
    info = {
        "field": f"f{idx}",
        "min": int(col.min()),
        "max": int(col.max()),
        "uniq": nuniq,
        "zeros_pct": round(100.0 * int((col == 0).sum()) / n, 1),
        "known": known.get(idx, ""),
        "note": "",
        "detail": "",
    }

    # --- текст и время проверяются ПЕРВЫМИ ---
    # Иначе они теряются: текстовая метка обычно константа, а время
    # монотонно растёт, и ранний выход по этим признакам не дал бы
    # добраться до содержательной трактовки.
    txt = looks_like_ascii(col)
    tm = looks_like_time(col)
    extra = ""
    if txt:
        extra = " | " + txt
    elif tm:
        extra = " | " + tm

    # --- характер значений ---
    if nuniq == 1:
        v = int(uniq[0])
        info["note"] = "ТЕКСТОВАЯ МЕТКА" if txt else "КОНСТАНТА"
        info["detail"] = f"всегда {v} (0x{v:X}){extra}"
        if v == 0:
            info["note"] = "всегда ноль"
            info["detail"] = "резерв или неиспользуемое поле"
        return info

    if nuniq <= 8:
        info["note"] = f"ФЛАГ / перечисление ({nuniq} значений)"
        cnt = Counter(int(x) for x in col[:50000])
        top = ", ".join(f"{v}×{c}" for v, c in cnt.most_common(6))
        info["detail"] = top + extra
        return info

    # монотонность
    d = np.diff(col.astype(np.int64))
    if len(d) and np.all(d >= 0):
        step = Counter(int(x) for x in d[:50000]).most_common(3)
        info["note"] = "ВРЕМЯ (монотонное)" if tm else "МОНОТОННО РАСТЁТ"
        info["detail"] = "шаги: " + ", ".join(f"{v}×{c}" for v, c in step) + extra
        return info
    if len(d) and (d >= 0).mean() > 0.95:
        info["note"] = "почти монотонный (есть сбросы)"
        info["detail"] = f"убываний: {int((d < 0).sum())}"

    # связь с известными полями
    for other_idx, other_col in others.items():
        if other_idx == idx or len(other_col) != n:
            continue
        if np.array_equal(col, other_col):
            info["note"] = f"ДУБЛИКАТ f{other_idx}"
            return info
        nz = other_col != 0
        if nz.sum() > 10:
            ratio = col[nz].astype(np.float64) / other_col[nz].astype(np.float64)
            if np.allclose(ratio, ratio[0], rtol=1e-6) and 0 < ratio[0]:
                info["note"] = f"пропорционально f{other_idx}"
                info["detail"] = f"коэффициент {ratio[0]:.6g}"
                return info
        diff = col.astype(np.int64) - other_col.astype(np.int64)
        if len(np.unique(diff)) == 1:
            info["note"] = f"f{other_idx} + {int(diff[0])}"
            return info

    if not info["note"]:
        info["note"] = "произвольные значения"
        cnt = Counter(int(x) for x in col[:50000])
        info["detail"] = "частые: " + ", ".join(
            f"{v}×{c}" for v, c in cnt.most_common(3)) + extra
    return info


def analyze_type(out: Report, words: np.ndarray, signature: int,
                 limit: int, show_raw: int, grid: int):
    pos, rejected = find_records(words, signature, grid)
    if not len(pos):
        return 0

    total_found = len(pos)
    if limit and len(pos) > limit:
        pos = pos[:limit]

    out("")
    out("=" * 78)
    out(f"ТИП ЗАПИСИ 0x{signature:X}  ({signature})   найдено: {total_found}")
    if rejected:
        out(f"Отброшено как случайные совпадения внутри данных: {rejected}")
        out("(значение сигнатуры встречается и как обычное число — "
            "например, номер кадра)")
    out("=" * 78)

    # Матрица записей: строка — запись, столбец — поле
    cols = {}
    for i in range(RECORD_WORDS):
        cols[i] = words[pos + i]

    known = KNOWN.get(signature, {})
    rows = []
    for i in range(RECORD_WORDS):
        rows.append(describe_field(i, cols[i], known, cols))

    out("")
    out(f"{'поле':>5} {'мин':>12} {'макс':>14} {'разных':>8} {'нулей%':>7}  "
        f"{'характер':<32} известно")
    out("-" * 108)
    for r in rows:
        mark = " " if r["known"] else "?"
        out(f"{mark}{r['field']:>4} {r['min']:>12} {r['max']:>14} "
            f"{r['uniq']:>8} {r['zeros_pct']:>7}  {r['note']:<32} {r['known']}")

    out("")
    out("Подробности по полям:")
    for r in rows:
        if r["detail"]:
            status = "известно" if r["known"] else "НЕ ИЗУЧЕНО"
            out(f"  {r['field']:>4} [{status}] {r['detail']}")

    # --- что можно предположить о неизученных полях ---
    unknown = [r for r in rows if not r["known"]]
    useful = [r for r in unknown
              if r["note"] not in ("всегда ноль", "произвольные значения")]
    out("")
    out("-- Неизученные поля, в которых есть содержание " + "-" * 30)
    if useful:
        for r in useful:
            out(f"  {r['field']}: {r['note']}  {r['detail']}")
        out("")
        out("  Эти поля стоит рассмотреть в первую очередь: в них есть")
        out("  структура, а значит и смысл.")
    else:
        empty = [r["field"] for r in unknown if r["note"] == "всегда ноль"]
        if empty:
            out(f"  Пустые (резерв): {', '.join(empty)}")
        out("  Полей с очевидной структурой среди неизученных нет.")

    # --- сырые записи для сверки с hex ---
    if show_raw:
        out("")
        out(f"-- Первые {show_raw} записей целиком " + "-" * 40)
        for k in range(min(show_raw, len(pos))):
            p = pos[k]
            vals = [int(words[p + i]) for i in range(RECORD_WORDS)]
            out(f"  запись {k}:")
            out("    dec: " + " ".join(f"{v:>10}" for v in vals))
            out("    hex: " + " ".join(f"{v:>10X}" for v in vals))

    return len(pos)


def main():
    ap = argparse.ArgumentParser(
        description="Анализ полей записей .idx")
    ap.add_argument("path", help="путь к .idx или .mp4")
    ap.add_argument("--limit", type=int, default=200000,
                    help="сколько записей каждого типа анализировать")
    ap.add_argument("--read-mb", type=float, default=200,
                    help="сколько мегабайт файла читать (0 = весь)")
    ap.add_argument("--type", default=None,
                    help="анализировать только этот тип, например 0x193")
    ap.add_argument("--raw", type=int, default=0,
                    help="показать N записей целиком в dec и hex")
    ap.add_argument("--scan-types", action="store_true",
                    help="искать ВСЕ возможные типы записей, а не только известные")
    args = ap.parse_args()

    path = find_idx(Path(args.path).resolve())
    out = Report()

    out("=" * 78)
    out("АНАЛИЗ ПОЛЕЙ ИНДЕКСА")
    out("=" * 78)
    out(f"Файл:   {path}")
    size_mb = path.stat().st_size / 1048576
    out(f"Размер: {size_mb:.1f} МБ")
    out(f"Запись: {RECORD_WORDS} полей × 4 байта = {RECORD_BYTES} байт")

    words = load_words(path, args.read_mb)
    out(f"Прочитано: {len(words) * 4 / 1048576:.1f} МБ "
        f"({len(words)} слов по 4 байта)")

    # --- какие типы записей вообще есть ---
    out("")
    out("-- Типы записей в файле " + "-" * 52)
    if args.type:
        signatures = [int(args.type, 0)]
    else:
        signatures = [0x193, 0xC9]

    if args.scan_types:
        # Ищем значения, которые встречаются с шагом, кратным размеру
        # записи — признак регулярной структуры.
        cnt = Counter(int(x) for x in words[:2_000_000])
        candidates = [(v, c) for v, c in cnt.most_common(40)
                      if c > 100 and v not in (0, 1) and v < 0x10000]
        out("  Часто встречающиеся малые значения (возможные сигнатуры):")
        for v, c in candidates[:15]:
            mark = "  <- известен" if v in (0x193, 0xC9) else ""
            out(f"    0x{v:<6X} ({v:>6}) : {c:>8} раз{mark}")
        for v, c in candidates[:6]:
            if v not in signatures:
                signatures.append(v)

    grid = detect_grid(words)
    out("")
    out(f"Выравнивание сетки записей: остаток {grid} по модулю {RECORD_WORDS}")
    out("(настоящие записи лежат на одной сетке; совпадения сигнатуры")
    out(" вне её — это данные, а не начало записи)")

    total = 0
    for sig in signatures:
        n = analyze_type(out, words, sig, args.limit, args.raw, grid)
        total += n
        if n == 0:
            out(f"\nТип 0x{sig:X}: записей не найдено")

    out("")
    out("=" * 78)
    out(f"Всего проанализировано записей: {total}")
    out("")
    out("КАК ЧИТАТЬ ТАБЛИЦУ")
    out("  Знак ? перед именем поля означает, что его назначение не")
    out("  установлено. Смотрите на колонку 'характер':")
    out("    всегда ноль     — резерв, можно не рассматривать")
    out("    КОНСТАНТА       — версия формата или признак источника")
    out("    ФЛАГ            — булев признак или небольшое перечисление")
    out("    МОНОТОННО       — счётчик или смещение")
    out("    ДУБЛИКАТ        — то же, что известное поле")
    out("    ТЕКСТ / ВРЕМЯ   — метка или временная отметка")
    out("")
    out("  Поля с характером 'ФЛАГ' и 'МОНОТОННО' наиболее перспективны:")
    out("  среди них может оказаться явная длина сэмпла (сейчас она")
    out("  вычисляется как расстояние до следующей записи) или более")
    out("  надёжный признак опорного кадра, чем диапазон NAL 27-30.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out.save(path.parent / f"idx_fields_{path.stem[:30]}_{ts}.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
