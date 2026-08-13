#!/usr/bin/env python3
"""
seek_stress_test.py – стресс-тест перемотки на реальном файле.
Исправлен: seek выполняется в отдельном потоке, главный поток контролирует таймаут.
"""

import sys, os, json, time, logging, threading, argparse
from pathlib import Path
from datetime import datetime
from PyQt5.QtWidgets import QApplication

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from player_window import PlayerWidget
from config.config import load_config
from index.idx_cache import prepare_mirror

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"seek_stress_test_{timestamp}.log"
METRICS_FILE = LOG_DIR / f"seek_stress_test_metrics_{timestamp}.json"

logger = logging.getLogger("SeekStressTest")
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(threadName)-12s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
fh.setFormatter(formatter)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

SEEK_ATTEMPTS = 3
SEEK_TIMEOUT = 5.0
CHECK_INTERVAL = 1

def save_metrics(metrics: dict):
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Метрики сохранены в {METRICS_FILE}")

def get_player_state(player) -> dict:
    state = {
        "timestamp": datetime.now().isoformat(),
        "playing": player.playing,
        "paused": getattr(player, "_paused", None),
        "seeking": getattr(player, "_seeking", None),
        "audio_clock": player.audio_clock if hasattr(player, "audio_clock") else None,
        "buffer_count": player.buffer_main.count if hasattr(player, "buffer_main") else None,
        "buffer_free": player.buffer_main.free_slots if hasattr(player, "buffer_main") else None,
        "buffer_keep_last_pts": player.buffer_main.get_keep_last_pts() if hasattr(player, "buffer_main") else None,
        "total_frames": player.total_frames if hasattr(player, "total_frames") else None,
        "current_frame": player.audio_clock // 1920 if hasattr(player, "audio_clock") else None,
        "first_pts": None,
        "sync_delta": None,
    }
    try:
        first = player.buffer_main.peek_first()
        if first:
            first_pts = first[0]
            state["first_pts"] = first_pts
            state["sync_delta"] = first_pts - state["audio_clock"]
    except Exception:
        pass
    return state

def wait_for_seek_done(widget, frame_idx, attempt_num):
    """
    Выполняет seek в отдельном потоке, контролируя таймаут.
    Возвращает (success, error_message).
    """
    seek_completed = threading.Event()
    seek_error = threading.Event()
    error_msg = []
    seek_thread = None

    def _do_seek():
        def on_complete():
            seek_completed.set()
        def on_error(msg):
            error_msg.append(msg)
            seek_error.set()
        try:
            widget.player.seek_absolute(frame_idx, callback=on_complete)
        except Exception as e:
            error_msg.append(str(e))
            seek_error.set()

    try:
        seek_thread = threading.Thread(target=_do_seek, daemon=True)
        seek_thread.start()

        deadline = time.monotonic() + SEEK_TIMEOUT
        while not seek_completed.is_set() and not seek_error.is_set():
            if time.monotonic() > deadline:
                # Пытаемся отменить текущий seek
                try:
                    widget.player._playback._seek_engine.cancel_current()
                except Exception:
                    pass
                return False, f"Таймаут {SEEK_TIMEOUT}с"
            time.sleep(0.1)
            QApplication.processEvents()

        if seek_error.is_set():
            return False, error_msg[0] if error_msg else "Неизвестная ошибка seek"
        return True, ""
    except Exception as e:
        return False, str(e)

def run_stress_test(mp4_path: Path, wait_seconds: int):
    logger.info(f"=== Запуск стресс-теста перемотки для {mp4_path} ===")
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    config = load_config()
    config.update({"fps": 25.0, "buffer_size": 600, "active_tracks": [2, 3]})

    # Поиск idx и ref
    idx_path = mp4_path.parent / "idx" / "mp4" / f"{mp4_path.stem}.idx"
    if not idx_path.exists():
        idx_path = mp4_path.parent / f"{mp4_path.stem}.idx"
    ref_path = mp4_path.parent / f"{mp4_path.stem}.mp4.ref"
    if not ref_path.exists():
        ref_path = mp4_path.parent / f"{mp4_path.stem}.ref"

    if not idx_path.exists() or not ref_path.exists():
        logger.error("Не найдены .idx или .ref файлы")
        return

    logger.info(f"Подготовка зеркала для {idx_path}...")
    try:
        mirror_path = prepare_mirror(idx_path)
        logger.info(f"Зеркало готово: {mirror_path}")
    except Exception as e:
        logger.exception("Не удалось подготовить зеркало")
        return

    widget = None
    try:
        widget = PlayerWidget(
            mp4_path=mp4_path,
            config=config,
            use_moov=False,
            mirror_path=str(mirror_path),
        )
        logger.info("Плеер создан, ожидание инициализации...")
        if not widget.player._ready.wait(timeout=60):
            logger.error("Плеер не инициализировался за 60 секунд")
            return
        if widget.player._init_error is not None:
            logger.error(f"Ошибка инициализации: {widget.player._init_error}")
            return

        logger.info("Плеер инициализирован. Запускаем воспроизведение.")
        widget.start_playback()
        time.sleep(2.0)

        initial_state = get_player_state(widget.player)
        logger.info(f"Начальное состояние: {json.dumps(initial_state, ensure_ascii=False)}")

        logger.info(f"Ожидание {wait_seconds} секунд перед перемотками...")
        wait_elapsed = 0
        while wait_elapsed < wait_seconds:
            time.sleep(CHECK_INTERVAL)
            wait_elapsed += CHECK_INTERVAL
            if wait_elapsed % 30 == 0:
                state = get_player_state(widget.player)
                logger.info(f"Ожидание: {wait_elapsed}/{wait_seconds} сек, состояние: {json.dumps(state, ensure_ascii=False)}")
            QApplication.processEvents()

        logger.info("Ожидание завершено. Начинаем серию быстрых перемоток.")
        total_frames = widget.player.total_frames
        if total_frames <= 0:
            logger.error("total_frames <= 0")
            return

        start_frame = 0
        end_frame = max(0, total_frames - 1)
        mid_frame = total_frames // 2

        seek_positions = [
            start_frame, end_frame, mid_frame,
            start_frame, end_frame, mid_frame,
            start_frame, end_frame, mid_frame,
            start_frame
        ]

        metrics_log = []
        for idx, target_frame in enumerate(seek_positions, 1):
            logger.info(f"=== Перемотка {idx}/10: целевой кадр {target_frame} ===")
            success = False
            last_error = ""

            for attempt in range(1, SEEK_ATTEMPTS + 1):
                logger.info(f"Попытка {attempt}/{SEEK_ATTEMPTS} для перемотки {idx}")
                success, last_error = wait_for_seek_done(widget, target_frame, attempt)
                if success:
                    logger.info(f"Перемотка {idx} успешна")
                    break
                else:
                    logger.warning(f"Перемотка {idx} не удалась: {last_error}. Пробуем ещё раз.")
                    time.sleep(0.5)
                    QApplication.processEvents()

            if not success:
                logger.error(f"Перемотка {idx} провалена после {SEEK_ATTEMPTS} попыток: {last_error}")

            state = get_player_state(widget.player)
            metrics_log.append({
                "seek_index": idx,
                "target_frame": target_frame,
                "attempts": attempt if success else SEEK_ATTEMPTS,
                "success": success,
                "error": "" if success else last_error,
                "state": state,
            })
            logger.info(f"Метрики после перемотки {idx}: {json.dumps(state, ensure_ascii=False)}")
            time.sleep(0.2)
            QApplication.processEvents()

        final_state = get_player_state(widget.player)
        logger.info(f"Итоговое состояние: {json.dumps(final_state, ensure_ascii=False)}")

        save_metrics({
            "file": str(mp4_path),
            "initial_state": initial_state,
            "final_state": final_state,
            "seek_results": metrics_log,
        })
        logger.info("Стресс-тест завершён.")

    except KeyboardInterrupt:
        logger.warning("Тест прерван пользователем (Ctrl+C)")
    except Exception as e:
        logger.exception("Критическая ошибка во время стресс-теста")
    finally:
        if widget is not None:
            try:
                logger.info("Закрываем плеер...")
                widget.player.close()
                widget.close()
            except Exception as e:
                logger.warning(f"Ошибка при закрытии плеера: {e}")
        QApplication.processEvents()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seek stress test")
    parser.add_argument("mp4", help="Путь к MP4 файлу")
    parser.add_argument("--wait-seconds", type=int, default=600, help="Время ожидания перед перемотками (сек)")
    args = parser.parse_args()

    mp4_file = Path(args.mp4)
    if not mp4_file.exists():
        print(f"Файл не найден: {mp4_file}")
        sys.exit(1)

    run_stress_test(mp4_file, args.wait_seconds)