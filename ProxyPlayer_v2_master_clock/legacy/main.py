#!/usr/bin/env python3
"""
main.py – точка входа в Dalet Proxy Player v5.
Поддержка режимов:
  - обычный плеер (main.py <mp4>)
  - управляемый плеер (main.py <mp4> --managed)
  - фоновый построитель индекса (main.py --index-builder <idx> [интервал])
  - пустой менеджер (запуск без аргументов или с ".")
  - единый экземпляр менеджера (single instance)
Менеджер самостоятельно разрешает ID файлов, переданных из Dalet.
"""

import sys
import os
import re
import time
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# Режим построителя индекса (без GUI)
# ═══════════════════════════════════════════════════════════════════
if '--index-builder' in sys.argv:
    try:
        idx_flag = sys.argv.index('--index-builder')
        args = sys.argv[idx_flag+1:]
        if len(args) < 1:
            print("Usage: --index-builder <idx_path> [poll_interval]")
            sys.exit(1)
        idx_path = Path(args[0])
        poll_interval = float(args[1]) if len(args) > 1 else 0.5

        if not getattr(sys, 'frozen', False):
            sys.path.insert(0, str(Path(__file__).resolve().parent))

        from index_builder import IndexBuilder
        builder = IndexBuilder(idx_path, poll_interval)
        builder.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            builder.stop()
        sys.exit(0)
    except Exception as e:
        print(f"Index builder failed: {e}", file=sys.stderr)
        sys.exit(1)

# ═══════════════════════════════════════════════════════════════════
# Функция преобразования пути от Video Helper (расширенная для ID)
# ═══════════════════════════════════════════════════════════════════
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


from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QStandardPaths, QCommandLineParser, QCommandLineOption, Qt
from logger import setup_logging
from config import load_config
from idx_cache import cleanup_cache
import logging


def global_exception_hook(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.getLogger("DaletPlayer").critical(
        "Необработанное исключение:",
        exc_info=(exc_type, exc_value, exc_traceback),
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
        socket.connectToServer("DaletPlayerManager")
        if socket.waitForConnected(500):
            # Определяем, является ли аргумент числовым ID
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
        socket.connectToServer("DaletPlayerManager")
        if socket.waitForConnected(200):
            socket.disconnectFromServer()
            socket.close()
            return True
    except Exception:
        pass
    return False


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("DaletProxyPlayer")
    app.setApplicationVersion("5.0.0")

    parser = QCommandLineParser()
    parser.setApplicationDescription("Плеер для Dalet MP4")
    parser.addHelpOption()
    parser.addVersionOption()
    parser.addPositionalArgument("file", "Путь к MP4 файлу")
    parser.addOption(QCommandLineOption("managed", "Запустить плеер в режиме управления менеджером"))
    parser.addOption(QCommandLineOption("moov", "Принудительно использовать moov-режим"))
    parser.process(app)

    args = parser.positionalArguments()
    managed = parser.isSet("managed")

    # Загружаем конфиг заранее, чтобы получить путь к логам
    config_path = Path(QStandardPaths.writableLocation(QStandardPaths.AppConfigLocation)) / "player_config.json"
    config = load_config(config_path)

    log_dir_str = config.get('log_directory', '')
    if log_dir_str:
        log_dir = Path(log_dir_str)
    else:
        log_dir = Path(QStandardPaths.writableLocation(QStandardPaths.AppConfigLocation)) / "DaletProxyPlayer" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    now = time.time()
    for f in log_dir.glob("*.log"):
        if f.is_file() and (now - f.stat().st_mtime) > 86400:
            try:
                f.unlink()
            except Exception:
                pass

    # Имя лог-файла
    if not args:
        log_name = log_dir / "dalet_player.log"
    else:
        mp4_path = Path(args[0])
        safe_stem = "".join(c if c.isalnum() or c in "._- " else "_" for c in mp4_path.stem) if mp4_path.stem else "empty"
        log_name = log_dir / f"{safe_stem}_{os.getpid()}.log"

    setup_logging(level=logging.INFO, log_file=str(log_name), mode='w')
    sys.excepthook = global_exception_hook
    logger = logging.getLogger(__name__)
    logger.info("Запуск Dalet Proxy Player v5 (ID-aware)")

    # --- Пустой аргумент или "." ---
    if not args or (len(args) == 1 and args[0] in ('', '.', './', '.\\')):
        if is_manager_running():
            logger.info("Менеджер уже запущен, завершаюсь")
            sys.exit(0)
        logger.info("Запуск без аргументов – открывается пустой менеджер")
        from player_window import ManagerWindow
        window = ManagerWindow(Path(""), config, False)
        window.setWindowTitle("Dalet Proxy Player v4 – Manager")
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
            from player_window import PlayerWidget
            widget = PlayerWidget(mp4_path, config, use_moov)
            widget.setWindowTitle(f"Player - {mp4_path.name}")
            widget.show()
            widget.send_hwnd_to_manager()
            widget.start_playback()
        else:
            from player_window import ManagerWindow
            window = ManagerWindow(mp4_path, config, use_moov)
            window.setWindowTitle("Dalet Proxy Player v4 – Manager")
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

    exit_code = app.exec_()
    try:
        if not use_moov and idx_candidate.exists():
            cleanup_cache(idx_candidate)
    except Exception:
        pass
    logger.info(f"Приложение завершилось с кодом {exit_code}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()