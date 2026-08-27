#!/usr/bin/env python3
"""
main.py – точка входа ProxyPlayer v2.
Запуск: python main.py [путь_к_mp4] [--managed] [--mirror <путь_к_зеркалу>]
Режимы:
  - обычный плеер (main.py <mp4>)
  - управляемый плеер (main.py <mp4> --managed)
  - фоновый сервис индекса (main.py --index-service <idx> [poll_interval])
  - пустой менеджер (запуск без аргументов или с ".")
  - единый экземпляр менеджера (single instance)
Менеджер самостоятельно разрешает ID файлов, переданных из Dalet.
"""

import sys
import os
import re
import time
import threading
import logging
import faulthandler
import atexit
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QStandardPaths, QCommandLineParser, QCommandLineOption, Qt
from config.logger import setup_logging
from config.config import load_config
from index.idx_cache import cleanup_cache


# --- Мониторинг Seek ---
from seek_monitor import install_seek_monitoring, attach_to_controller
from player_telemetry import install_telemetry, attach_controller


# Глобальная ссылка на файл аварийных дампов: faulthandler пишет в него на
# уровне ОС в момент падения, поэтому файл обязан оставаться открытым весь
# сеанс. Без этой ссылки объект собрал бы сборщик мусора, дескриптор
# закрылся бы, и дамп ушёл бы в никуда — ровно та ситуация, ради которой
# всё и делается.
_crash_log_file = None


def _resolve_log_dir(config: dict = None) -> Path:
    """
    Каталог для логов. Вынесено в функцию, чтобы одинаково работало и в
    обычном режиме, и в ветке --index-service (там конфиг не загружается).
    """
    log_dir_str = (config or {}).get('log_directory', '')
    if log_dir_str:
        log_dir = Path(log_dir_str)
    else:
        log_dir = Path(QStandardPaths.writableLocation(
            QStandardPaths.AppConfigLocation)) / "ProxyPlayer" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def enable_crash_handler(log_dir: Path, tag: str = "player") -> None:
    """
    Включает faulthandler — перехват НАТИВНЫХ падений (access violation и
    подобных) в ctypes/WinAPI, PyAV, sounddevice, OpenGL.

    Зачем отдельно от sys.excepthook/threading.excepthook: те ловят только
    исключения уровня Python. Когда процесс умирает внутри C-кода, Python-хуки
    не вызываются вообще — процесс просто исчезает, и в логах пусто (именно
    этот симптом и наблюдался). faulthandler ставит обработчик сигналов на
    уровне ОС и успевает сбросить C-стек и стеки ВСЕХ Python-потоков в файл
    до смерти процесса.

    Дамп пишется в отдельный файл crash_<tag>_<pid>.log — не в общий
    player.log, чтобы аварийный вывод не смешивался с обычным (и потому что
    faulthandler пишет напрямую в дескриптор, минуя logging).
    """
    global _crash_log_file
    try:
        crash_path = log_dir / f"crash_{tag}_{os.getpid()}.log"
        _crash_log_file = open(crash_path, "a", encoding="utf-8", buffering=1)
        _crash_log_file.write(
            f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} | старт {tag} "
            f"(pid={os.getpid()}) =====\n"
        )
        _crash_log_file.flush()
        faulthandler.enable(file=_crash_log_file, all_threads=True)
        logging.getLogger("ProxyPlayer").info("faulthandler включён: %s", crash_path)

        # Пустой файл при штатном завершении смысла не имеет — прибираем,
        # чтобы каталог логов не зарастал следами нормальных запусков.
        atexit.register(_cleanup_crash_log, crash_path)
    except Exception as e:
        # Отсутствие аварийного лога не повод не запускать плеер.
        logging.getLogger("ProxyPlayer").warning(
            "Не удалось включить faulthandler: %s", e)


def _cleanup_crash_log(crash_path: Path) -> None:
    """Удаляет файл дампа, если в нём остался только заголовок (падений не было)."""
    global _crash_log_file
    try:
        faulthandler.disable()
        if _crash_log_file:
            _crash_log_file.close()
            _crash_log_file = None
        if crash_path.exists():
            text = crash_path.read_text(encoding="utf-8", errors="ignore")
            # Только строки-заголовки "===== ... =====" — значит, дампов нет.
            meaningful = [ln for ln in text.splitlines()
                          if ln.strip() and not ln.startswith("=====")]
            if not meaningful:
                crash_path.unlink()
    except Exception:
        pass


def global_exception_hook(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.getLogger("ProxyPlayer").critical(
        "Необработанное исключение:",
        exc_info=(exc_type, exc_value, exc_traceback),
    )


def thread_exception_hook(args):
    logging.getLogger("ProxyPlayer").critical(
        "Необработанное исключение в потоке %s:",
        args.thread.name,
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback)
    )


def send_to_existing_manager(file_path: str) -> bool:
    """
    Отправляет команду открытия файла в уже запущенный ManagerWindow.
    Если file_path состоит только из цифр, отправляется как ID:...,
    иначе как FILE:...
    """
    try:
        from PyQt5.QtNetwork import QLocalSocket
        socket = QLocalSocket()
        socket.connectToServer("ProxyPlayerManager")
        if socket.waitForConnected(500):
            if re.match(r'^\d+$', file_path):
                msg = f"ID:{file_path}"
            else:
                msg = f"FILE:{file_path}"
            socket.write(msg.encode())
            socket.flush()
            socket.disconnectFromServer()
            socket.close()
            return True
    except Exception:
        pass
    return False


def is_manager_running() -> bool:
    """Проверяет, запущен ли уже ManagerWindow."""
    try:
        from PyQt5.QtNetwork import QLocalSocket
        socket = QLocalSocket()
        socket.connectToServer("ProxyPlayerManager")
        if socket.waitForConnected(200):
            socket.disconnectFromServer()
            socket.close()
            return True
    except Exception:
        pass
    return False


def resolve_media_path(input_path: str, homedir: str = "") -> Path:
    """
    Преобразует путь от Video Helper (.wrec, .mp4) или ID файла
    в полный путь к MP4-файлу.
    Если input_path состоит только из цифр, ищет файл ID_YYYYMMDD_HHMMSS.mp4 в homedir.
    """
    path = Path(input_path)
    # 1. Абсолютный путь существует – сразу возвращаем
    if path.is_absolute() and path.exists():
        return _fix_wrec_path(path)
    if path.is_absolute():
        fixed = _fix_wrec_path(path)
        if fixed.exists():
            return fixed

    # 2. Проверяем, является ли input_path числовым ID
    is_numeric_id = bool(re.match(r'^\d+$', input_path))
    if is_numeric_id and homedir:
        from utils import resolve_id_to_mp4
        found = resolve_id_to_mp4(input_path, homedir)
        if found:
            return found

    # 3. Ищем в homedir по точному имени (как раньше)
    if homedir:
        home = Path(homedir)
        if home.exists():
            candidate = home / path.name
            if candidate.exists():
                return _fix_wrec_path(candidate)
            for found in home.rglob(path.name):
                return _fix_wrec_path(found)

    # 4. Текущая директория
    if path.exists():
        return _fix_wrec_path(path)
    cwd = Path.cwd()
    candidate = cwd / path.name
    if candidate.exists():
        return _fix_wrec_path(candidate)

    # 5. Если это ID и homedir не помог, возвращаем как есть (ошибка будет позже)
    return _fix_wrec_path(path)


def _fix_wrec_path(path: Path) -> Path:
    path_str = str(path)
    if 'wrec' in path.suffix.lower():
        fixed_str = path_str.replace('wrec', '')
        fixed = Path(fixed_str)
        if fixed.exists():
            return fixed
    return path


def main():
    # Устанавливаем перехватчик исключений для потоков
    threading.excepthook = thread_exception_hook

    # ═══════════════════════════════════════════════════════════════
    # Режим IndexService (без GUI)
    # ═══════════════════════════════════════════════════════════════
    if '--index-service' in sys.argv:
        try:
            idx_flag = sys.argv.index('--index-service')
            args = sys.argv[idx_flag + 1:]
            if len(args) < 1:
                print("Usage: --index-service <idx_path> [poll_interval]")
                sys.exit(1)
            idx_path = Path(args[0])
            poll_interval = float(args[1]) if len(args) > 1 else 10.0

            if not getattr(sys, 'frozen', False):
                sys.path.insert(0, str(Path(__file__).resolve().parent))

            # Логирование для сервиса индекса.
            # Раньше эта ветка была полностью немой: ни setup_logging(), ни
            # sys.excepthook — только print() в stderr, который в GUI-режиме
            # уходит в никуда. Если сервис падал, обновление индекса тихо
            # прекращалось, и в логах не оставалось никаких следов.
            # Имя файла привязано к имени .idx и pid, чтобы сервисы разных
            # файлов не писали в один лог.
            svc_log_dir = _resolve_log_dir()
            safe_idx = "".join(c if c.isalnum() or c in "._- " else "_"
                               for c in idx_path.stem) or "idx"
            setup_logging(
                level=logging.INFO,
                log_file=str(svc_log_dir / f"indexservice_{safe_idx}_{os.getpid()}.log"),
                mode='w',
            )
            sys.excepthook = global_exception_hook
            enable_crash_handler(svc_log_dir, tag="indexservice")
            logging.getLogger("ProxyPlayer").info(
                "IndexService: старт для %s (poll_interval=%.1f)", idx_path, poll_interval)

            # QApplication уже импортирован глобально, используем его
            app = QApplication(sys.argv)
            app.setApplicationName("ProxyPlayerIndexService")

            from index.index_service import IndexService
            install_telemetry(svc_log_dir, tag="indexservice")
            service = IndexService(str(idx_path), poll_interval)

            import signal
            signal.signal(signal.SIGINT, lambda sig, frame: service.stop())
            signal.signal(signal.SIGTERM, lambda sig, frame: service.stop())

            sys.exit(app.exec_())
        except Exception as e:
            # logging может быть ещё не настроен (падение до setup_logging) —
            # поэтому и print в stderr, и попытка записи в лог.
            print(f"Index service failed: {e}", file=sys.stderr)
            try:
                logging.getLogger("ProxyPlayer").exception("IndexService: аварийное завершение")
            except Exception:
                pass
            sys.exit(1)

    # Обычный запуск
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("ProxyPlayer")
    app.setApplicationVersion("2.0.0")

    parser = QCommandLineParser()
    parser.setApplicationDescription("Плеер для Dalet MP4")
    parser.addHelpOption()
    parser.addVersionOption()
    parser.addPositionalArgument("file", "Путь к MP4 файлу")
    parser.addOption(QCommandLineOption("managed", "Запустить плеер в режиме управления менеджером"))
    parser.addOption(QCommandLineOption("moov", "Принудительно использовать moov-режим"))
    parser.addOption(QCommandLineOption("mirror", "Путь к локальному зеркалу .idx", "path"))
    parser.process(app)

    args = parser.positionalArguments()
    managed = parser.isSet("managed")
    mirror_path = parser.value("mirror")  # может быть пустым

    # Загружаем конфиг заранее, чтобы получить путь к логам
    config_path = Path(QStandardPaths.writableLocation(QStandardPaths.AppConfigLocation)) / "player_config.json"
    config = load_config(config_path)

    log_dir = _resolve_log_dir(config)

    now = time.time()
    # Под маску *.log попадают и crash_*.log — старые дампы тоже подчищаются.
    for f in log_dir.glob("*.log"):
        if f.is_file() and (now - f.stat().st_mtime) > 86400:
            try:
                f.unlink()
            except Exception:
                pass

    # Имя лог-файла
    if not args:
        log_name = log_dir / "player.log"
    else:
        mp4_path = Path(args[0])
        safe_stem = "".join(c if c.isalnum() or c in "._- " else "_" for c in mp4_path.stem) if mp4_path.stem else "empty"
        log_name = log_dir / f"{safe_stem}_{os.getpid()}.log"

    setup_logging(level=logging.INFO, log_file=str(log_name), mode='w')
    sys.excepthook = global_exception_hook
    # Нативные падения (ctypes/WinAPI, PyAV, sounddevice, OpenGL) не проходят
    # через sys.excepthook — их ловит только faulthandler. Включаем сразу
    # после настройки логирования, до создания любых компонентов плеера.
    enable_crash_handler(log_dir, tag="managed" if managed else "player")
    install_telemetry(log_dir, tag="manged" if managed else "player")

    # ──────────────────────────────────────────────
    # Включаем отладку для аудио‑компонентов (v2: MasterClock и pipeline)
    # ──────────────────────────────────────────────
    debug_handler = logging.StreamHandler(sys.stderr)
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%H:%M:%S'
    ))

    for name in ["pipeline.chunk_pipeline.DemuxerStage",
                 "pipeline.chunk_pipeline.AudioDecoderStage",
                 "core.master_clock"]:
        lg = logging.getLogger(name)
        lg.setLevel(logging.DEBUG)
        lg.addHandler(debug_handler)
        lg.propagate = False  # не дублируем в файл

    logger = logging.getLogger(__name__)
    logger.info("Запуск ProxyPlayer v2")

    # ──────────────────────────────────────────────
    # Включаем debug-запись SyncManager в файл sync_debug.log,
    # если запущены через main_debug_sync.py
    # ──────────────────────────────────────────────
    if os.environ.get("PYPLAYER_DEBUG_SYNC") == "1":
        sync_logger = logging.getLogger("SyncMonitor")
        sync_logger.setLevel(logging.DEBUG)
        # Убираем старые обработчики, чтобы не дублировать
        for handler in sync_logger.handlers[:]:
            sync_logger.removeHandler(handler)

        # Файл sync_debug.log в папке рядом с main.py
        debug_log_path = Path(__file__).resolve().parent / "sync_debug.log"
        file_handler = logging.FileHandler(debug_log_path, encoding="utf-8", mode="w")
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        sync_logger.addHandler(file_handler)
        sync_logger.propagate = False

    # --- Включаем мониторинг Seek ---
    install_seek_monitoring()

    # --- Пустой аргумент или "." ---
    if not args or (len(args) == 1 and args[0] in ('', '.', './', '.\\')):
        if is_manager_running():
            logger.info("Менеджер уже запущен, завершаюсь")
            sys.exit(0)
        logger.info("Запуск без аргументов – открывается пустой менеджер")
        from player_window import ManagerWindow
        window = ManagerWindow(None, config, False)
        window.setWindowTitle("ProxyPlayer v2 – Manager")
        window.show()
        window.hide()   # сразу скрываем панель, остаётся только трей
        exit_code = app.exec_()
        logger.info(f"Приложение завершилось с кодом {exit_code}")
        sys.exit(exit_code)

    # --- Есть аргумент – обрабатываем ---
    homedir = config.get('homedir', '')
    original_path = args[0]
    mp4_path = resolve_media_path(original_path, homedir)

    if not mp4_path.exists():
        logger.error(f"Файл не найден: {original_path} -> {mp4_path} (homedir={homedir})")
        QMessageBox.critical(None, "Ошибка", f"Файл не найден:\n{mp4_path}\n\nБазовая папка: {homedir or 'не задана'}")
        sys.exit(1)

    logger.info(f"Открывается файл: {mp4_path}")

    if not managed:
        # Если менеджер уже запущен, отправляем ему путь (ID или полный)
        if send_to_existing_manager(original_path):
            logger.info("Файл отправлен в существующий ManagerWindow")
            sys.exit(0)

    # Проверяем обязательное наличие .idx (кроме случая принудительного moov)
    use_moov = parser.isSet("moov")
    idx_candidate = mp4_path.parent / "idx" / "mp4" / f"{mp4_path.stem}.idx"
    if not idx_candidate.exists():
        idx_candidate = mp4_path.parent / f"{mp4_path.stem}.idx"
    if not use_moov and not idx_candidate.exists():
        logger.error(f"Индексный файл не найден: {idx_candidate}")
        QMessageBox.critical(None, "Ошибка", f"Не найден индексный файл (.idx) для:\n{mp4_path}")
        sys.exit(1)

    try:
        if managed:
            try:
                from player_window import PlayerWidget
                widget = PlayerWidget(mp4_path, config, use_moov, mirror_path=mirror_path)
                # --- Прикрепляем мониторинг к созданному контроллеру ---
                if hasattr(widget, 'player'):
                    attach_to_controller(widget.player)
                    attach_controller(widget.player)
                widget.setWindowTitle(f"Player - {mp4_path.name}")
                widget.show()
                widget.send_hwnd_to_manager()
                widget.start_playback()
                app._widget = widget
            except Exception as e:
                logger.exception("Ошибка в managed режиме")
                QMessageBox.critical(None, "Ошибка", f"Не удалось запустить плеер:\n{e}")
                sys.exit(1)
        else:
            from player_window import ManagerWindow
            window = ManagerWindow(mp4_path, config, use_moov)
            window.setWindowTitle("ProxyPlayer v2 – Manager")
            window.show()
            window.hide()   # сразу сворачиваем в трей
    except FileNotFoundError as e:
        logger.error(f"Ошибка открытия файла: {e}")
        QMessageBox.critical(None, "Ошибка", f"Не удалось открыть файл:\n{e}")
        sys.exit(1)
    except Exception as e:
        logger.exception("Ошибка инициализации плеера")
        QMessageBox.critical(None, "Ошибка", f"Ошибка инициализации:\n{e}")
        sys.exit(1)

    if not managed:
        logger.info("Главное окно менеджера отображено")

    try:
        exit_code = app.exec_()
    except Exception as e:
        logger.exception("Исключение в цикле событий Qt")
        exit_code = 1

    try:
        if not use_moov and idx_candidate.exists():
            cleanup_cache(idx_candidate)
    except Exception:
        pass
    logger.info(f"Приложение завершилось с кодом {exit_code}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()