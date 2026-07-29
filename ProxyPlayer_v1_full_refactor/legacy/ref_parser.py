"""
ref_parser.py – извлечение атомов ftyp, avcC и AudioSpecificConfig (ASC) из .ref файла Dalet.
Версия production: детальное логирование, строгие проверки, отсутствие скрытых fallback-путей.
"""

import struct
import logging
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def find_atom(data: bytes, fourcc: bytes) -> Optional[Tuple[int, int]]:
    """
    Ищет атом (box) по четырёхсимвольному коду в MP4-подобной структуре.
    Возвращает (offset, size) или None, если атом не найден или повреждён.
    """
    pos = data.find(fourcc)
    if pos < 4:
        return None
    size = struct.unpack_from('>I', data, pos - 4)[0]
    if size < 8 or pos - 4 + size > len(data):
        logger.warning(f"Атом {fourcc!r} найден, но размер {size} некорректен (pos={pos})")
        return None
    return (pos - 4, size)


def extract_ftyp_avcc(ref_path: Path) -> Tuple[bytes, bytes]:
    """
    Извлекает ftyp и avcC из .ref файла Dalet.
    Возвращает (ftyp_data, avcc_data).
    Если обязательные атомы не найдены, выбрасывает исключение.
    """
    if not ref_path.exists():
        raise FileNotFoundError(f".ref не найден: {ref_path}")

    data = ref_path.read_bytes()
    logger.info(f"Разбор .ref: {ref_path} ({len(data)} байт)")

    # Ищем ftyp
    ftyp_atom = find_atom(data, b'ftyp')
    if not ftyp_atom:
        raise ValueError("Атом ftyp не найден в .ref")
    ftyp_data = data[ftyp_atom[0]:ftyp_atom[0] + ftyp_atom[1]]
    logger.debug(f"Извлечён ftyp: {len(ftyp_data)} байт")

    # Ищем avcC
    avcc_atom = find_atom(data, b'avcC')
    if not avcc_atom:
        raise ValueError("Атом avcC не найден в .ref")
    # avcC данные начинаются после поля size (4 байта) и fourcc (4 байта), т.е. с 8-го байта атома
    avcc_data = data[avcc_atom[0] + 8:avcc_atom[0] + avcc_atom[1]]
    logger.debug(f"Извлечён avcC: {len(avcc_data)} байт")
    return ftyp_data, avcc_data


def extract_asc(ref_path: Path) -> bytes:
    """
    Извлекает AudioSpecificConfig из .ref файла.
    Возвращает ASC (обычно 2 байта) или fallback b'\x11\x88' в случае неудачи,
    при этом логируется предупреждение.
    """
    if not ref_path.exists():
        logger.warning(f".ref не найден ({ref_path}), использую ASC по умолчанию")
        return b'\x11\x88'

    data = ref_path.read_bytes()
    logger.debug(f"Поиск ASC в {ref_path} ({len(data)} байт)")

    # Сначала ищем esds глобально
    esds_atom = find_atom(data, b'esds')
    if esds_atom:
        esds_data = data[esds_atom[0]:esds_atom[0] + esds_atom[1]]
    else:
        # Пробуем найти внутри mp4a
        mp4a = find_atom(data, b'mp4a')
        if mp4a:
            mp4a_data = data[mp4a[0]:mp4a[0] + mp4a[1]]
            esds_atom = find_atom(mp4a_data, b'esds')
            if esds_atom:
                # Корректируем смещение относительно начала файла
                esds_atom = (esds_atom[0] + mp4a[0], esds_atom[1])
                esds_data = data[esds_atom[0]:esds_atom[0] + esds_atom[1]]
                logger.debug("esds найден внутри mp4a")
            else:
                logger.warning("Атом esds не найден ни глобально, ни внутри mp4a")
                return b'\x11\x88'
        else:
            logger.warning("Атом mp4a не найден, поиск esds невозможен")
            return b'\x11\x88'

    # --- Метод 1: прямой поиск сигнатуры 05 02 11 88 ---
    asc_marker = b'\x05\x02\x11\x88'
    pos = esds_data.find(asc_marker)
    if pos >= 0:
        # Пропускаем тег (1 байт) и длину (1 байт)
        asc = esds_data[pos + 2:pos + 4]
        logger.info(f"ASC извлечён прямым поиском маркера: {asc.hex()}")
        return asc

    # --- Метод 2: полноценный парсинг ES_Descriptor ---
    logger.debug("Прямой поиск ASC не удался, пробую структурный парсинг esds")
    try:
        # Пропускаем заголовок: size (4 байта) + 'esds' (4 байта) -> offset = 8
        offset = 8
        if offset + 4 > len(esds_data):
            logger.warning("esds слишком короткий")
            return b'\x11\x88'

        # Version/Flags (4 байта)
        offset += 4
        if esds_data[offset] != 0x03:  # ES_Descriptor tag
            logger.warning("Ожидался тег 0x03 (ES_Descriptor)")
            return b'\x11\x88'
        offset += 1
        # Длина дескриптора (varint)
        length = 0
        for _ in range(4):
            b = esds_data[offset]
            offset += 1
            length = (length << 7) | (b & 0x7F)
            if not (b & 0x80):
                break
        # Пропускаем ES_ID (2 байта) и флаги (1 байт)
        offset += 3

        # Теперь ищем DecoderConfigDescriptor (тег 0x04)
        while offset < len(esds_data):
            tag = esds_data[offset]
            offset += 1
            tag_len = 0
            for _ in range(4):
                b = esds_data[offset]
                offset += 1
                tag_len = (tag_len << 7) | (b & 0x7F)
                if not (b & 0x80):
                    break
            tag_end = offset + tag_len

            if tag == 0x04:  # DecoderConfigDescriptor
                # objectTypeIndication (1), streamType (1), bufferSizeDB (3), maxBitrate (4), avgBitrate (4)
                offset += 1 + 1 + 3 + 4 + 4
                # Ищем DecoderSpecificInfo (тег 0x05)
                while offset < tag_end:
                    inner_tag = esds_data[offset]
                    offset += 1
                    inner_len = 0
                    for _ in range(4):
                        b = esds_data[offset]
                        offset += 1
                        inner_len = (inner_len << 7) | (b & 0x7F)
                        if not (b & 0x80):
                            break
                    inner_end = offset + inner_len
                    if inner_tag == 0x05:
                        asc = esds_data[offset:inner_end]
                        logger.info(f"ASC извлечён структурным парсингом: {asc.hex()}")
                        return asc
                    offset = inner_end
                break
            else:
                offset = tag_end
    except (IndexError, ValueError) as e:
        logger.warning(f"Ошибка при парсинге esds: {e}")

    logger.warning("Не удалось извлечь ASC, использую стандартный 0x1188")
    return b'\x11\x88'