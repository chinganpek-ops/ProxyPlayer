"""
config.py – загрузка и сохранение конфигурации плеера (production).
Строгая проверка типов, логирование всех операций, сохранение целостности.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

from PyQt5.QtCore import QStandardPaths

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: Dict[str, Any] = {
    "buffer_size": 800,
    "free_slots_required": 25,
    "drop_window_backward": 300,
    "drop_window_forward": 300,
    "group_chunks": 30,
    "max_retries": 3,
    "normal_playback_speed_factor": 1.25,
    "live_speed_factor": 1.15,
    "live_delay": 1.0,
    "auto_refresh_interval": 10,
    "default_rate_limit": 0,
    "fps": 25.0,
    "thread_type": "AUTO",
    "thread_count": 0,
    "skip_frame": False,
    "use_gpu_decoder": "off",
    "default_avcc": "014d001fffe1002e674d401f9652816824dff80200016a50101014000003000400000300cb8180009600000301e848fc6383b428532c01000568e9093520",
    "audio_enabled": True,
    "audio_buffer_sec": 30.0,
    "volume": 0.8,
    "audio_delay_ms": 0,          # НОВЫЙ ПАРАМЕТР
    "audio_device": "default",
    "active_tracks": [2, 3],
    "show_settings_button": False,
    "homedir": "",
}

KEY_TYPES = {
    "buffer_size": int,
    "free_slots_required": int,
    "drop_window_backward": int,
    "drop_window_forward": int,
    "group_chunks": int,
    "max_retries": int,
    "normal_playback_speed_factor": float,
    "live_speed_factor": float,
    "live_delay": float,
    "auto_refresh_interval": int,
    "default_rate_limit": int,
    "fps": float,
    "thread_type": str,
    "thread_count": int,
    "skip_frame": bool,
    "use_gpu_decoder": str,
    "default_avcc": str,
    "audio_enabled": bool,
    "audio_buffer_sec": float,
    "volume": float,
    "audio_delay_ms": int,        # НОВЫЙ ТИП
    "audio_device": str,
    "active_tracks": list,
    "show_settings_button": bool,
    "homedir": str,
}

_HEX_PATTERN = re.compile(r'^[0-9a-fA-F]*$')


def _validate_and_fix(config: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {}
    for key, default_value in DEFAULT_CONFIG.items():
        if key in config:
            try:
                expected_type = KEY_TYPES.get(key)
                if expected_type:
                    if expected_type is bool:
                        cleaned[key] = bool(config[key])
                    elif expected_type is list:
                        cleaned[key] = list(config[key])
                    else:
                        cleaned[key] = expected_type(config[key])
                else:
                    cleaned[key] = config[key]
                logger.debug(f"Параметр {key} = {cleaned[key]!r}")
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Некорректное значение параметра '{key}': {config[key]!r}. "
                    f"Использую значение по умолчанию {default_value!r}. Ошибка: {e}"
                )
                cleaned[key] = default_value
        else:
            logger.debug(f"Параметр '{key}' отсутствует в файле, используется значение по умолчанию.")
            cleaned[key] = default_value

    # Проверка диапазона audio_delay_ms
    delay = cleaned.get("audio_delay_ms", 0)
    if not (-15000 <= delay <= 15000):
        logger.warning(f"audio_delay_ms {delay} вне [-15000, 15000], сброшен в 0")
        cleaned["audio_delay_ms"] = 0

    # Проверка avcc
    avcc_val = cleaned.get("default_avcc", "")
    if avcc_val:
        if len(avcc_val) % 2 != 0 or not _HEX_PATTERN.match(avcc_val):
            logger.warning("default_avcc не является корректной hex-строкой, сбрасываю на значение по умолчанию")
            cleaned["default_avcc"] = DEFAULT_CONFIG["default_avcc"]
    return cleaned


def load_config(custom_path: Optional[Path] = None) -> Dict[str, Any]:
    path = custom_path or _config_path()
    logger.info(f"Загрузка конфигурации из {path}")

    if not path.exists():
        logger.info("Файл конфигурации не найден. Создаю новый с настройками по умолчанию.")
        config = DEFAULT_CONFIG.copy()
        save_config(config, path)
        return config

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.debug(f"Сырые данные из файла конфигурации: {data}")
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка парсинга JSON в файле конфигурации: {e}. Перезаписываю настройками по умолчанию.")
        config = DEFAULT_CONFIG.copy()
        save_config(config, path)
        return config
    except OSError as e:
        logger.error(f"Ошибка чтения файла конфигурации: {e}. Использую настройки по умолчанию в памяти.")
        return DEFAULT_CONFIG.copy()

    config = DEFAULT_CONFIG.copy()
    config.update(data)
    config = _validate_and_fix(config)

    # Проверка диапазонов
    if config["volume"] < 0.0 or config["volume"] > 1.0:
        logger.warning(f"Громкость {config['volume']} вне допустимого диапазона [0.0, 1.0]. Установлено значение 0.8.")
        config["volume"] = 0.8
    if config["audio_buffer_sec"] < 1.0 or config["audio_buffer_sec"] > 30.0:
        logger.warning(
            f"Размер аудиобуфера {config['audio_buffer_sec']}с вне допустимого диапазона [1.0, 30.0]. Установлено 10.0с."
        )
        config["audio_buffer_sec"] = 10.0
    if config["thread_type"] not in ("AUTO", "FRAME", "SLICE"):
        logger.warning(f"Неизвестный thread_type '{config['thread_type']}'. Использую 'AUTO'.")
        config["thread_type"] = "AUTO"
    if config["use_gpu_decoder"] not in ("off", "on", "auto"):
        logger.warning(f"Неизвестный use_gpu_decoder '{config['use_gpu_decoder']}'. Использую 'off'.")
        config["use_gpu_decoder"] = "off"

    logger.info("Конфигурация успешно загружена и проверена.")
    return config


def save_config(config: Dict[str, Any], custom_path: Optional[Path] = None) -> None:
    path = custom_path or _config_path()
    logger.info(f"Сохранение конфигурации в {path}")

    full_config = DEFAULT_CONFIG.copy()
    full_config.update(config)

    tmp_path = path.with_suffix(".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(full_config, f, indent=4, ensure_ascii=False)
        tmp_path.replace(path)
        logger.info("Конфигурация сохранена.")
    except OSError as e:
        logger.error(f"Ошибка при сохранении конфигурации: {e}")
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(full_config, f, indent=4, ensure_ascii=False)
            logger.warning("Конфигурация сохранена напрямую (без атомарной замены).")
        except OSError as e2:
            logger.critical(f"Не удалось сохранить конфигурацию: {e2}")
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)