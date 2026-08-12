"""
test_real_file.py – тест на реальном файле Dalet (путь встроен для вашего файла).
Запуск из Git Bash:
  pytest tests_v2/test_real_file.py -v
Если нужно указать другой файл, задайте переменную DALET_MP4:
  export DALET_MP4='//10.20.49.171/DaletProxy/LSU/LR_2689/1920178_2026-08-11T15-45-25768.mp4'
"""

import os
import pytest
from pathlib import Path
import numpy as np

from index.lazy_index import LazyIndex
from seek.seek_engine import SeekEngine
from decode.decoder import Decoder
from file_io.win_sequential_reader import WinSequentialReader
from file_io.ref_parser import extract_ftyp_avcc
from utils.utils import get_real_size

# ----- Файл по умолчанию (если переменная DALET_MP4 не задана) -----
DEFAULT_MP4 = Path("//10.20.49.171/DaletProxy/LSU/LR_2689/1920178_2026-08-11T15-45-25768.mp4")

def _find_idx(mp4_path: Path) -> Path:
    """Ищет .idx в стандартной структуре Dalet."""
    idx = mp4_path.parent / "idx" / "mp4" / f"{mp4_path.stem}.idx"
    if idx.exists():
        return idx
    idx = mp4_path.parent / f"{mp4_path.stem}.idx"
    if idx.exists():
        return idx
    return None

def _find_ref(mp4_path: Path) -> Path:
    """Ищет .ref (сначала .mp4.ref, потом .ref)."""
    ref = mp4_path.parent / f"{mp4_path.stem}.mp4.ref"
    if ref.exists():
        return ref
    ref = mp4_path.parent / f"{mp4_path.stem}.ref"
    if ref.exists():
        return ref
    return None


class TestRealFile:
    @pytest.fixture(scope="class")
    @classmethod
    def dalet_files(cls):
        mp4_str = os.environ.get("DALET_MP4")
        mp4 = Path(mp4_str) if mp4_str else DEFAULT_MP4
        if not mp4.exists():
            pytest.fail(f"MP4-файл не найден: {mp4}")
        idx = _find_idx(mp4)
        ref = _find_ref(mp4)
        if not idx:
            pytest.fail(f"Не найден .idx для {mp4}")
        if not ref:
            pytest.fail(f"Не найден .ref для {mp4}")
        return {"mp4": mp4, "idx": idx, "ref": ref}

    def test_seek_engine_on_real_file(self, dalet_files):
        """Проверяет seek на реальном файле: IDR, PTS, кадры."""
        mdat_end = get_real_size(str(dalet_files["mp4"]))
        assert mdat_end > 0

        li = LazyIndex(dalet_files["idx"], dalet_files["mp4"], mdat_end)
        _, avcc = extract_ftyp_avcc(dalet_files["ref"])
        decoder = Decoder(avcc, dalet_files["mp4"])
        reader = WinSequentialReader(dalet_files["mp4"])
        engine = SeekEngine(li, decoder, reader)

        # Прыгаем на кадр 100 (проверьте, что он существует)
        buf = engine.seek_sync(100)
        assert buf is not None and buf.count > 0, "Seek не вернул кадры"

        pts = [entry[0] for entry in buf.peek_all()]
        assert pts == sorted(pts), "PTS не по порядку"

        reader.close()
        decoder.close()
        li.close()