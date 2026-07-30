"""
tcd_parser.py – парсинг файла .tcd (TimeCode Data) Dalet.
Формат: "PAL,<номер>/<ч>:<м>:<с>:<к>," (завершающая запятая опциональна).
Возвращает начальный таймкод как (hours, minutes, seconds, frames).
Версия production: детальное логирование, строгая проверка входных данных.
"""

import re
import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

# Допустимые диапазоны для PAL (25 fps)
MAX_HOURS = 23
MAX_MINUTES = 59
MAX_SECONDS = 59
MAX_FRAMES = 24  # в PAL 0-24 (иногда 0-24, кадр 25 = 0 следующей секунды)


def parse_tcd(tcd_path: Path) -> Tuple[int, int, int, int]:
    """
    Извлекает стартовый таймкод из .tcd файла.

    Аргументы:
        tcd_path: путь к .tcd файлу.

    Возвращает:
        (hours, minutes, seconds, frames) – целочисленный таймкод.
        При любой ошибке возвращает (0, 0, 0, 0) и логирует предупреждение.
    """
    if not tcd_path.exists():
        logger.warning(f".tcd файл не найден: {tcd_path}")
        return 0, 0, 0, 0

    logger.info(f"Парсинг .tcd: {tcd_path}")

    # Чтение файла
    try:
        text = tcd_path.read_text(encoding='ascii', errors='ignore').strip()
        logger.debug(f"Содержимое .tcd: {text!r}")
    except Exception as e:
        logger.error(f"Ошибка чтения {tcd_path}: {e}")
        return 0, 0, 0, 0

    if not text:
        logger.warning("Файл .tcd пуст")
        return 0, 0, 0, 0

    # Убираем завершающую запятую, если есть
    if text.endswith(','):
        text = text[:-1].strip()

    # Разбор составных частей: "PAL,<номер>/<timecode>" или просто "<timecode>"
    # Обычно формат: "<header>,<frame_number>/<timecode>" или без header
    timecode_part = text
    if ',' in text:
        parts = text.split(',', 1)
        if len(parts) == 2:
            header, tail = parts[0].strip(), parts[1].strip()
            logger.debug(f"Заголовок: '{header}', хвост: '{tail}'")
            timecode_part = tail
        else:
            logger.debug("Запятая есть, но структура нестандартная, используем всю строку")

    # Убираем номер кадра перед '/' если есть (например "12345/01:02:03:04")
    if '/' in timecode_part:
        _, tc_part = timecode_part.split('/', 1)
        timecode_part = tc_part.strip()
        logger.debug(f"После удаления номера кадра: {timecode_part!r}")

    # Заменяем возможные ';' на ':' (некоторые системы используют ; как разделитель кадров)
    tc_clean = timecode_part.replace(';', ':').strip()

    # Регулярное выражение для часов:минут:секунд:кадров
    match = re.match(r'(\d{1,2}):(\d{1,2}):(\d{1,2}):(\d{1,2})', tc_clean)
    if not match:
        logger.warning(f"Неверный формат таймкода: {timecode_part!r}")
        return 0, 0, 0, 0

    try:
        h, m, s, f = map(int, match.groups())
    except ValueError as e:
        logger.warning(f"Некорректные числа в таймкоде: {match.groups()} ({e})")
        return 0, 0, 0, 0

    # Проверка диапазонов
    if not (0 <= h <= MAX_HOURS and 0 <= m <= MAX_MINUTES and 0 <= s <= MAX_SECONDS and 0 <= f <= MAX_FRAMES):
        logger.warning(
            f"Значения таймкода вне допустимых диапазонов: {h:02d}:{m:02d}:{s:02d}:{f:02d} "
            f"(ожидалось h≤{MAX_HOURS}, m≤{MAX_MINUTES}, s≤{MAX_SECONDS}, f≤{MAX_FRAMES})"
        )
        return 0, 0, 0, 0

    logger.info(f"Начальный таймкод: {h:02d}:{m:02d}:{s:02d};{f:02d}")
    return h, m, s, f