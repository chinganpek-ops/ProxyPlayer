#!/usr/bin/env python3
"""
analyze_media.py – офлайн-анализ реального файла и его индекса (вариант А).

Ничего не воспроизводит, звуковая карта не нужна. Читает .idx и MP4
напрямую и отвечает на вопросы, которые сейчас блокируют правку
синхронизации звука:

1. Сколько записей в индексе у трека 2 и у трека 3?
   Если у трека 3 их изначально меньше — чинить надо индексатор или
   демуксер, а не микшер MasterClock. Если поровну — значит пакеты
   теряются при декодировании, и это другая правка.

2. Непрерывны ли PTS по каждому треку?
   Разрывы = провалы звука, наложения = треск, дубли = дребезг.

3. Сколько сэмплов реально отдаёт декодер на пакет?
   Ожидается 2048 (два AAC-фрейма по 1024). Отклонения означают
   обрезанные пакеты.

4. Совпадают ли аудио- и видео-таймлайны?
   Есть ли аудио на каждый видеочанк и с какого места начинается каждый.

5. Каков интервал между IDR-кадрами?
   Определяет реально нужный размер seek-буфера (BUFFER_MAX_FRAMES).

ЗАПУСК
    python analyze_media.py "D:\\media\\1966520_....mp4"
    python analyze_media.py 1966520                 # по ID, homedir из конфига
    python analyze_media.py <файл> --decode 200     # декодировать 200 аудиопакетов
    python analyze_media.py <файл> --no-decode      # только индекс, без чтения MP4

Результат печатается в консоль и сохраняется рядом:
    analyze_<имя>_<ts>.txt
"""

import argparse
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def resolve_media(raw: str):
    """Находит MP4 по пути или числовому ID — как это делает main.py."""
    path = Path(raw)
    if path.exists():
        return path
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


def find_ref(mp4: Path) -> Path:
    ref = mp4.parent / f"{mp4.stem}.mp4.ref"
    return ref if ref.exists() else mp4.parent / f"{mp4.stem}.ref"


class Report:
    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text)
        self.lines.append(text)

    def save(self, path: Path):
        path.write_text("\n".join(self.lines), encoding="utf-8")
        print(f"\nОтчёт сохранён: {path}")


def analyze(args):
    out = Report()
    mp4 = resolve_media(args.media)
    idx = find_idx(mp4)
    if not idx.exists():
        raise FileNotFoundError(f"Не найден индекс: {idx}")

    out("=" * 72)
    out("АНАЛИЗ МЕДИАФАЙЛА И ИНДЕКСА")
    out("=" * 72)
    out(f"MP4:    {mp4}")
    out(f"размер: {mp4.stat().st_size / 1048576:.1f} МБ")
    out(f"IDX:    {idx}")
    out(f"размер: {idx.stat().st_size / 1048576:.2f} МБ")
    out("")

    # ------------------------------------------------------------------
    # Чтение индекса
    # ------------------------------------------------------------------
    from index.idx_cache import prepare_mirror, remap_idx_incremental
    from index.moov_builder import (
        _filter_normal_records, _abs_offset, build_audio_tracks,
        get_idr_indices_from_mmap, DEFAULT_TRACK_FILTER,
    )
    from config.timebase import (
        SAMPLES_PER_VIDEO_FRAME, FRAMES_PER_CHUNK, SAMPLES_PER_CHUNK,
        AUDIO_SAMPLE_RATE,
    )

    mirror = prepare_mirror(idx)
    all_193, all_c9, _next = remap_idx_incremental(mirror, 0)
    video = _filter_normal_records(all_193)

    out("-- Индекс " + "-" * 61)
    out(f"  записей 0x193 (видео):  {len(all_193)}")
    out(f"  после фильтрации:       {len(video)}")
    out(f"  записей 0xC9  (аудио):  {len(all_c9)}")
    if len(video):
        dur_min = len(video) / 25 / 60
        out(f"  длительность видео:     {len(video)} кадров ≈ {dur_min:.1f} мин")
    out("")

    # ------------------------------------------------------------------
    # Аудиодорожки: главный вопрос — поровну ли записей
    # ------------------------------------------------------------------
    audio = build_audio_tracks(all_c9, track_filter=DEFAULT_TRACK_FILTER)
    out("-- Аудиодорожки в индексе " + "-" * 45)
    out(f"  всего записей после фильтра {DEFAULT_TRACK_FILTER}: {len(audio)}")

    per_track = {}
    for tid in DEFAULT_TRACK_FILTER:
        sel = audio[audio["track"] == tid]
        per_track[tid] = sel
        out(f"  трек {tid}: {len(sel)} записей")

    t2, t3 = per_track.get(2), per_track.get(3)
    if t2 is not None and t3 is not None and len(t2) and len(t3):
        diff = len(t2) - len(t3)
        out("")
        out(f"  РАЗНИЦА: {diff} записей "
            f"({abs(diff) / max(len(t2), len(t3)) * 100:.2f}%)")
        if diff == 0:
            out("  => В ИНДЕКСЕ ДОРОЖКИ РАВНЫ.")
            out("     Значит расхождение возникает ПОЗЖЕ — при демуксе или")
            out("     декодировании. Искать в AudioDecoderStage.")
        else:
            out("  => ДОРОЖКИ НЕРАВНЫ УЖЕ В ИНДЕКСЕ.")
            out("     Микшер MasterClock ни при чём: одному каналу физически")
            out("     нечего играть. Искать в индексаторе/источнике записи.")
    out("")

    # ------------------------------------------------------------------
    # Непрерывность PTS
    # ------------------------------------------------------------------
    out("-- Непрерывность PTS по дорожкам " + "-" * 38)
    for tid, sel in per_track.items():
        if sel is None or len(sel) < 2:
            continue
        pts = np.sort(sel["pts"].astype(np.int64))
        d = np.diff(pts)
        uniq = Counter(d.tolist())
        common = uniq.most_common(3)
        out(f"  трек {tid}:")
        out(f"    PTS от {pts[0]} до {pts[-1]}")
        out(f"    шаг между записями (топ-3): "
            f"{', '.join(f'{v}×{c}' for v, c in common)}")
        if common:
            step = common[0][0]
            gaps = np.where(d > step)[0]
            overlaps = np.where(d < step)[0]
            dups = int((d == 0).sum())
            out(f"    РАЗРЫВОВ (шаг больше обычного): {len(gaps)}")
            out(f"    НАЛОЖЕНИЙ (шаг меньше):        {len(overlaps)}")
            out(f"    ДУБЛЕЙ (шаг ноль):             {dups}")
            if len(gaps):
                worst = int(d[gaps].max())
                out(f"    самый большой разрыв: {worst} сэмплов "
                    f"({worst / AUDIO_SAMPLE_RATE * 1000:.0f} мс)")
                # Показываем, где именно — по времени от начала
                for i in gaps[:5]:
                    t_sec = int(pts[i]) / AUDIO_SAMPLE_RATE
                    out(f"      на {t_sec/60:.1f} мин: пропуск {int(d[i])} сэмплов")
    out("")

    # ------------------------------------------------------------------
    # Сопоставление аудио и видео по времени
    # ------------------------------------------------------------------
    out("-- Согласованность аудио и видео " + "-" * 38)
    if len(video) and len(audio):
        video_end_pts = len(video) * SAMPLES_PER_VIDEO_FRAME
        audio_end_pts = int(audio["pts"].max())
        out(f"  видео заканчивается на PTS {video_end_pts}")
        out(f"  аудио заканчивается на PTS {audio_end_pts}")
        delta = audio_end_pts - video_end_pts
        out(f"  разница: {delta} сэмплов ({delta / AUDIO_SAMPLE_RATE:.2f} с)")
        if abs(delta) > AUDIO_SAMPLE_RATE:
            out("  ВНИМАНИЕ: дорожки расходятся более чем на секунду —")
            out("  одна из них обрывается раньше другой.")

        # Сколько чанков вообще без аудио
        total_chunks = (len(video) + FRAMES_PER_CHUNK - 1) // FRAMES_PER_CHUNK
        chunk_of = (audio["pts"].astype(np.int64) // SAMPLES_PER_CHUNK)
        per_chunk = Counter(chunk_of.tolist())
        empty = [c for c in range(total_chunks) if per_chunk.get(c, 0) == 0]
        out(f"  чанков всего: {total_chunks}, без аудио: {len(empty)}")
        if empty[:5]:
            out(f"    первые без аудио: {empty[:5]}")
        counts = Counter(per_chunk.values())
        out(f"  записей аудио на чанк (топ-3): "
            f"{', '.join(f'{k}→{v} чанков' for k, v in counts.most_common(3))}")
    out("")

    # ------------------------------------------------------------------
    # IDR
    # ------------------------------------------------------------------
    out("-- IDR-кадры " + "-" * 58)
    idr = get_idr_indices_from_mmap(video)
    out(f"  найдено IDR: {len(idr)}")
    if len(idr) > 1:
        d = np.diff(idr)
        out(f"  интервал: мин {int(d.min())}, макс {int(d.max())}, "
            f"медиана {int(np.median(d))}")
        out(f"  чанк = {FRAMES_PER_CHUNK} кадров")
        need = int(d.max()) + 12
        out(f"  => для seek достаточно буфера {need} кадров "
            f"(сейчас BUFFER_MAX_FRAMES=300)")
    out("")

    # ------------------------------------------------------------------
    # Реальное декодирование
    # ------------------------------------------------------------------
    if args.no_decode:
        out("(декодирование пропущено: --no-decode)")
    else:
        decode_audio(out, mp4, idx, audio, per_track, args)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out.save(ROOT / f"analyze_{mp4.stem[:30]}_{ts}.txt")


def decode_audio(out, mp4: Path, idx: Path, audio, per_track, args):
    """Декодирует N пакетов каждой дорожки и меряет реальный выход."""
    from decode.audio_decoder import AudioDecoder
    from file_io.win_sequential_reader import WinSequentialReader
    from file_io.ref_parser import extract_asc

    out("-- Реальное декодирование аудио " + "-" * 39)

    ref = find_ref(mp4)
    asc = b"\x11\x88"
    if ref.exists():
        try:
            asc = extract_asc(ref)
            out(f"  ASC из .ref: {asc.hex()}")
        except Exception as e:
            out(f"  ASC по умолчанию (ошибка чтения .ref: {e})")
    else:
        out(f"  ASC по умолчанию (нет .ref)")

    reader = WinSequentialReader(mp4)
    try:
        for tid, sel in per_track.items():
            if sel is None or not len(sel):
                continue
            n = min(args.decode, len(sel))
            sel = np.sort(sel, order="pts")[:n]
            dec = AudioDecoder(asc)
            sizes = Counter()
            errors = 0
            empty_d2 = 0
            total_samples = 0
            peak = 0.0
            dtypes = set()

            for entry in sel:
                try:
                    off = int(entry["abs_offset"])
                    s1 = int(entry["size1"])
                    s2 = int(entry["size2"])
                    reader.seek(off)
                    raw = reader.read(s1 + s2)
                    d1 = raw[:s1]
                    d2 = raw[s1:s1 + s2] if s2 > 0 else b""
                    if not s2:
                        empty_d2 += 1
                    pcm1 = dec.decode(d1)
                    pcm2 = dec.decode(d2) if d2 else np.array([], dtype=np.float32)
                    block = np.concatenate([pcm1, pcm2])
                    sizes[len(block)] += 1
                    total_samples += len(block)
                    dtypes.add(str(block.dtype))
                    if len(block):
                        peak = max(peak, float(np.abs(block).max()))
                except Exception:
                    errors += 1

            out(f"  трек {tid}: обработано {n} записей")
            out(f"    ошибок декодирования: {errors}")
            out(f"    записей без второй части (size2=0): {empty_d2}")
            out(f"    dtype: {', '.join(dtypes) or '—'}")
            out(f"    пик амплитуды: {peak:.3f}")
            out(f"    размеры блоков: "
                f"{', '.join(f'{k}×{v}' for k, v in sizes.most_common(4))}")
            if sizes and len(sizes) > 1:
                out("    ВНИМАНИЕ: размеры блоков неодинаковы — часть пакетов")
                out("    декодируется не полностью (обрезанные данные).")
            if errors:
                out(f"    ВНИМАНИЕ: {errors} из {n} пакетов не декодировались.")
                out("    Это прямая причина недостачи сэмплов у дорожки.")
    finally:
        reader.close()
    out("")


def main():
    ap = argparse.ArgumentParser(
        description="Офлайн-анализ медиафайла и индекса ProxyPlayer")
    ap.add_argument("media", help="путь к MP4 или числовой ID")
    ap.add_argument("--decode", type=int, default=300,
                    help="сколько аудиозаписей каждой дорожки декодировать (по умолчанию 300)")
    ap.add_argument("--no-decode", action="store_true",
                    help="только анализ индекса, без чтения MP4")
    args = ap.parse_args()

    try:
        analyze(args)
    except Exception:
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
