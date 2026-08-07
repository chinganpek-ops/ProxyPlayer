"""
timebase.py – единая целочисленная временна́я шкала для Dalet Proxy Player (production).
Все временны́е метки хранятся в аудиосэмплах (48000 Гц).
Константы и функции преобразования. Потокобезопасно (чистые функции).
"""

from typing import Tuple

# ----- Основные константы -----
AUDIO_SAMPLE_RATE = 48000            # Гц
VIDEO_FPS_NUM = 25                   # числитель частоты видео
VIDEO_FPS_DEN = 1                    # знаменатель (25/1 = 25 fps)
SAMPLES_PER_AAC_FRAME = 1024         # сэмплов в одном AAC-фрейме
SAMPLES_PER_AAC = SAMPLES_PER_AAC_FRAME  # совместимость с moov_parser

# Длительность одного видеокадра в аудиосэмплах (целое число)
SAMPLES_PER_VIDEO_FRAME = AUDIO_SAMPLE_RATE * VIDEO_FPS_DEN // VIDEO_FPS_NUM   # 1920

# ----- Тип временно́й метки -----
Pts = int  # все метки – целые числа (сэмплы)

# ----- Функции преобразования -----
def video_frame_to_pts(frame_idx: int) -> Pts:
    """Переводит индекс видеокадра в PTS (аудиосэмплы)."""
    return frame_idx * SAMPLES_PER_VIDEO_FRAME

def pts_to_video_frame(pts: Pts) -> int:
    """Переводит PTS в индекс видеокадра (целочисленное деление)."""
    return pts // SAMPLES_PER_VIDEO_FRAME

def aac_frames_to_pts(aac_frame_count: int) -> Pts:
    """Переводит количество AAC-фреймов в PTS."""
    return aac_frame_count * SAMPLES_PER_AAC_FRAME

def samples_to_seconds(samples: int) -> float:
    """Преобразует сэмплы в секунды (float)."""
    return samples / AUDIO_SAMPLE_RATE

def seconds_to_samples(seconds: float) -> Pts:
    """Преобразует секунды в целое число сэмплов (округление)."""
    return round(seconds * AUDIO_SAMPLE_RATE)

def frame_duration_samples() -> int:
    """Возвращает длительность видеокадра в сэмплах."""
    return SAMPLES_PER_VIDEO_FRAME

def aac_frame_duration_samples() -> int:
    """Возвращает длительность AAC-фрейма в сэмплах."""
    return SAMPLES_PER_AAC_FRAME

def compare_pts(pts1: Pts, pts2: Pts) -> int:
    """Сравнивает два PTS: отрицательное если pts1 < pts2, 0 если равны, положительное иначе."""
    return pts1 - pts2

def pts_delta(pts1: Pts, pts2: Pts) -> int:
    """Возвращает абсолютную разницу между двумя PTS."""
    return abs(pts1 - pts2)

def timecode_to_frame(timecode_str: str, fps: float = 25.0) -> int:
    """
    Преобразует строку таймкода в индекс кадра.
    
    Поддерживаемые форматы:
      - "HH:MM:SS;FF" или "HH:MM:SS:FF" (часы:минуты:секунды;кадры)
      - "MM:SS;FF" или "MM:SS:FF" (минуты:секунды;кадры, часы = 0)
      - "SS;FF" или "SS:FF" (секунды;кадры)
      - просто число (трактуется как индекс кадра)
    
    Args:
        timecode_str: строка таймкода
        fps: частота кадров (по умолчанию 25.0)
    
    Returns:
        Индекс кадра (целое число)
    
    Raises:
        ValueError: если строка не соответствует ни одному формату
    """
    timecode_str = timecode_str.strip()
    
    # Пустая строка
    if not timecode_str:
        raise ValueError("Пустая строка таймкода")
    
    # Пробуем как простое число (индекс кадра)
    try:
        return int(timecode_str)
    except ValueError:
        pass
    
    # Заменяем ; на : для единообразия
    tc = timecode_str.replace(';', ':')
    parts = tc.split(':')
    
    try:
        if len(parts) == 4:
            # HH:MM:SS:FF
            h, m, s, f = map(int, parts)
        elif len(parts) == 3:
            # MM:SS:FF
            h, m, s = 0, *map(int, parts)
            f = 0
        elif len(parts) == 2:
            # SS:FF
            h, m = 0, 0
            s, f = map(int, parts)
        else:
            raise ValueError(f"Неверный формат таймкода: {timecode_str}")
    except ValueError:
        raise ValueError(f"Некорректные числа в таймкоде: {timecode_str}")
    
    # Проверка диапазонов
    if not (0 <= h <= 23 and 0 <= m <= 59 and 0 <= s <= 59):
        raise ValueError(f"Таймкод вне диапазона: {timecode_str}")
    if f < 0 or f >= fps:
        raise ValueError(f"Кадры вне диапазона 0-{int(fps)-1}: {timecode_str}")
    
    total_seconds = h * 3600 + m * 60 + s + f / fps
    return int(total_seconds * fps)

# Дополнительная удобная lambda (оставлена для обратной совместимости)
SECONDS_TO_SAMPLES = lambda sec: int(sec * AUDIO_SAMPLE_RATE)