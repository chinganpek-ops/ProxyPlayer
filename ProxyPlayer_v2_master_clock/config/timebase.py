"""
timebase.py – единая временна́я шкала для ProxyPlayer v1.
Все метки в аудиосэмплах 48 кГц.
"""

from typing import Tuple

# ----- Основные константы -----
AUDIO_SAMPLE_RATE = 48000            # Гц
VIDEO_FPS = 25.0                     # Кадров в секунду
SAMPLES_PER_AAC_FRAME = 1024         # Сэмплов в одном AAC-фрейме
SAMPLES_PER_VIDEO_FRAME = AUDIO_SAMPLE_RATE // int(VIDEO_FPS)   # 1920
FRAMES_PER_CHUNK = 12                # Кадров в одном чанке
SAMPLES_PER_CHUNK = SAMPLES_PER_VIDEO_FRAME * FRAMES_PER_CHUNK  # 23040

# ----- Тип временно́й метки -----
Pts = int

# ----- Функции преобразования -----
def video_frame_to_pts(frame_idx: int) -> Pts:
    """Переводит индекс видеокадра в PTS (аудиосэмплы)."""
    return frame_idx * SAMPLES_PER_VIDEO_FRAME

def pts_to_video_frame(pts: Pts) -> int:
    """Переводит PTS в индекс видеокадра."""
    return pts // SAMPLES_PER_VIDEO_FRAME

def aac_frames_to_pts(aac_frame_count: int) -> Pts:
    """Переводит количество AAC-фреймов в PTS."""
    return aac_frame_count * SAMPLES_PER_AAC_FRAME

def samples_to_seconds(samples: int) -> float:
    """Сэмплы → секунды."""
    return samples / AUDIO_SAMPLE_RATE

def seconds_to_samples(seconds: float) -> Pts:
    """Секунды → сэмплы (округление)."""
    return round(seconds * AUDIO_SAMPLE_RATE)

def frame_duration_samples() -> int:
    """Длительность видеокадра в сэмплах."""
    return SAMPLES_PER_VIDEO_FRAME

def aac_frame_duration_samples() -> int:
    """Длительность AAC-фрейма в сэмплах."""
    return SAMPLES_PER_AAC_FRAME

def compare_pts(pts1: Pts, pts2: Pts) -> int:
    """Сравнивает два PTS."""
    return pts1 - pts2

def pts_delta(pts1: Pts, pts2: Pts) -> int:
    """Абсолютная разница между PTS."""
    return abs(pts1 - pts2)

def timecode_to_frame(timecode_str: str, fps: float = 25.0) -> int:
    """
    Преобразует таймкод в индекс кадра.
    Форматы: "HH:MM:SS;FF", "MM:SS;FF", "SS;FF", или просто число.
    """
    timecode_str = timecode_str.strip()
    if not timecode_str:
        raise ValueError("Пустая строка таймкода")
    try:
        return int(timecode_str)
    except ValueError:
        pass

    tc = timecode_str.replace(';', ':')
    parts = tc.split(':')

    try:
        if len(parts) == 4:
            h, m, s, f = map(int, parts)
        elif len(parts) == 3:
            h, m, s = 0, *map(int, parts)
            f = 0
        elif len(parts) == 2:
            h, m = 0, 0
            s, f = map(int, parts)
        else:
            raise ValueError(f"Неверный формат таймкода: {timecode_str}")
    except ValueError:
        raise ValueError(f"Некорректные числа в таймкоде: {timecode_str}")

    if not (0 <= h <= 23 and 0 <= m <= 59 and 0 <= s <= 59):
        raise ValueError(f"Таймкод вне диапазона: {timecode_str}")
    if f < 0 or f >= fps:
        raise ValueError(f"Кадры вне диапазона 0-{int(fps)-1}: {timecode_str}")

    total_seconds = h * 3600 + m * 60 + s + f / fps
    return int(total_seconds * fps)

# Lambda для совместимости
SECONDS_TO_SAMPLES = lambda sec: int(sec * AUDIO_SAMPLE_RATE)