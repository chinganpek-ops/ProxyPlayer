"""
seek_monitor.py – расширенный мониторинг Seek и падений плеера.
Адаптирован под новый SeekEngine (пул воркеров с координатором).

Подключается через monkey-patching. Логирует:
- Вызовы seek_async с поколениями и целевыми кадрами.
- Запуск/завершение задач воркеров, состояния воркеров (free/busy/stuck).
- Отмены, таймауты, блокировки.
- Исключения в потоках с полным трейсбеком.
- Qt-сообщения (включая фатальные).
- Активные потоки при завершении процесса.
- Стек вызовов при закрытии окна.

Использование:
    from seek_monitor import install_seek_monitoring, patch_close_event
    install_seek_monitoring()
    # ... после создания PlayerWidget:
    from player_window import PlayerWidget
    patch_close_event(PlayerWidget)
"""

import sys
import atexit
import logging
import threading
import traceback
import time
from logging.handlers import RotatingFileHandler

# Настройка логгера
logger = logging.getLogger("SeekMonitorDebug")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = RotatingFileHandler(
        "seek_monitor_debug.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(threadName)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.propagate = False

# --- Перехват Qt-сообщений ---
try:
    from PyQt5.QtCore import qInstallMessageHandler, QtMsgType

    def qt_message_handler(mode, context, message):
        if mode == QtMsgType.QtFatalMsg:
            logger.critical("Qt Fatal: %s", message)
        elif mode == QtMsgType.QtCriticalMsg:
            logger.error("Qt Critical: %s", message)
        elif mode == QtMsgType.QtWarningMsg:
            logger.warning("Qt Warning: %s", message)
        else:
            logger.debug("Qt: %s", message)

    qInstallMessageHandler(qt_message_handler)
    logger.info("Qt message handler установлен")
except ImportError:
    logger.warning("PyQt5 не найден, Qt-сообщения не перехватываются")

# --- Расширенный перехват исключений в потоках ---
def thread_excepthook(args):
    logger.critical(
        "Необработанное исключение в потоке %s: %s\n%s",
        args.thread.name, args.exc_value, traceback.format_exc()
    )
    # Логируем все активные потоки
    logger.critical("Активные потоки на момент исключения:")
    for t in threading.enumerate():
        logger.critical("  - %s (daemon=%s, alive=%s)", t.name, t.daemon, t.is_alive())

threading.excepthook = thread_excepthook

# --- Перехват при завершении процесса ---
def atexit_handler():
    logger.info("Процесс завершается. Активные потоки:")
    for t in threading.enumerate():
        logger.info("  - %s (daemon=%s, alive=%s)", t.name, t.daemon, t.is_alive())

atexit.register(atexit_handler)

# --- Оригинальные методы для обёртки ---
_original_methods = {}

# Пороги предупреждений (сек)
SEEK_WARN_TIMEOUT = 5.0
CLOSE_WARN_TIMEOUT = 3.0

# Флаг остановки фонового наблюдателя
_monitor_stop_event = threading.Event()


def _log_exception(e: Exception, context: str):
    """Логирует исключение с полным трейсбеком."""
    logger.error(
        "Исключение в %s: %s\n%s",
        context, e, traceback.format_exc()
    )


def _wrap_method(obj, method_name: str, before=None, after=None):
    """Обёртка метода: вызывает before, оригинал, after."""
    if not hasattr(obj, method_name):
        logger.debug(f"Метод {method_name} не найден у {obj.__class__.__name__}, обёртка пропущена")
        return
    original = getattr(obj, method_name)
    if method_name in _original_methods:
        return

    def wrapper(*args, **kwargs):
        if before:
            try:
                before(*args, **kwargs)
            except Exception as e:
                _log_exception(e, f"before {method_name}")
        result = None
        try:
            result = original(*args, **kwargs)
            if after:
                try:
                    after(result, *args, **kwargs)
                except Exception as e:
                    _log_exception(e, f"after {method_name}")
        except Exception as e:
            _log_exception(e, method_name)
            raise
        return result

    _original_methods[method_name] = original
    setattr(obj, method_name, wrapper)


def _monitor_seek_engine(seek_engine):
    """Оборачивает методы SeekEngine (адаптировано под пул воркеров)."""

    # Публичные методы
    _wrap_method(
        seek_engine,
        'seek_async',
        before=lambda *args, **kwargs: logger.info(
            "[SEEK ASYNC CALL] frame=%s", args[0] if args else None
        ),
        after=lambda result, *args, **kwargs: logger.info(
            "[SEEK REQUEST CREATED] generation=%s", result.generation if result else None
        )
    )

    _wrap_method(
        seek_engine,
        'cancel_current',
        before=lambda *args, **kwargs: logger.info("[SEEK CANCEL CALL]")
    )

    _wrap_method(
        seek_engine,
        'close',
        before=lambda *args, **kwargs: logger.info("[SEEK ENGINE CLOSE START]"),
        after=lambda result, *args, **kwargs: logger.info("[SEEK ENGINE CLOSE END]")
    )

    # Внутренние хуки для отслеживания задач воркеров
    if hasattr(seek_engine, '_handle_command'):
        _wrap_method(
            seek_engine,
            '_handle_command',
            before=lambda *args, **kwargs: logger.info(
                "[SEEK DISPATCH] frame=%s gen=%s",
                args[0].frame_idx if args and hasattr(args[0], 'frame_idx') else None,
                args[0].request.generation if args and hasattr(args[0], 'request') else None
            )
        )

    if hasattr(seek_engine, '_run_worker_task'):
        _wrap_method(
            seek_engine,
            '_run_worker_task',
            before=lambda *args, **kwargs: logger.info(
                "[WORKER TASK START] worker_id=%s frame=%s",
                args[0].id if args else None,
                args[1].frame_idx if len(args) > 1 and hasattr(args[1], 'frame_idx') else None
            ),
            after=lambda result, *args, **kwargs: logger.info(
                "[WORKER TASK END] worker_id=%s",
                args[0].id if args else None
            )
        )

    if hasattr(seek_engine, '_finalize_worker'):
        _wrap_method(
            seek_engine,
            '_finalize_worker',
            before=lambda *args, **kwargs: logger.info(
                "[WORKER FINALIZE] result=%s",
                args[0].worker.id if args and hasattr(args[0], 'worker') else None
            )
        )


def _monitor_playback_engine(pe):
    """Оборачивает методы PlaybackEngine."""
    _wrap_method(
        pe,
        'seek',
        before=lambda *args, **kwargs: logger.info(
            "[PLAYBACK SEEK] frame=%s gen_before=%s",
            args[0] if args else None,
            getattr(pe, '_seek_generation', None)
        ),
        after=lambda result, *args, **kwargs: logger.info(
            "[PLAYBACK SEEK RETURNED] gen_after=%s", getattr(pe, '_seek_generation', None)
        )
    )

    _wrap_method(
        pe,
        'apply_seek_buffer',
        before=lambda *args, **kwargs: logger.info("[APPLY SEEK BUFFER]"),
        after=lambda result, *args, **kwargs: logger.info("[APPLY SEEK BUFFER DONE]")
    )

    _wrap_method(
        pe,
        'close',
        before=lambda *args, **kwargs: logger.info("[PLAYBACK CLOSE START]"),
        after=lambda result, *args, **kwargs: logger.info("[PLAYBACK CLOSE END]")
    )


def _monitor_stream_controller(sc):
    """Оборачивает методы StreamController."""
    _wrap_method(
        sc,
        'seek_absolute',
        before=lambda *args, **kwargs: logger.info(
            "[STREAM SEEK] frame=%s callback=%s",
            args[0] if args else None,
            'callback' in kwargs or len(args) > 1
        ),
        after=lambda result, *args, **kwargs: logger.info("[STREAM SEEK RETURNED]")
    )

    _wrap_method(
        sc,
        'close',
        before=lambda *args, **kwargs: logger.info("[STREAM CLOSE START]"),
        after=lambda result, *args, **kwargs: logger.info("[STREAM CLOSE END]")
    )


def _background_watcher(pe, seek_engine):
    """Фоновый поток: логирует состояние каждые 2 секунды."""
    while not _monitor_stop_event.is_set():
        try:
            flags = {
                "seek_in_progress": getattr(pe, '_seek_in_progress', None),
                "sliding": getattr(pe, '_sliding_in_progress', None).is_set(),
                "growth_refresh": getattr(pe, '_growth_refresh_in_progress', None).is_set(),
                "closed": getattr(pe, '_closed', None).is_set(),
                "current_frame": getattr(pe, '_current_frame_idx', None),
                "audio_clock": getattr(pe, 'audio_clock', None),
            }
            # Состояние воркеров нового SeekEngine
            worker_states = []
            if hasattr(seek_engine, 'workers'):
                for w in seek_engine.workers:
                    worker_states.append(
                        f"W{w.id}:{w.state}(buf={w.buffer.count},alive={w.thread.is_alive() if w.thread else False})"
                    )
            flags["workers"] = " | ".join(worker_states) if worker_states else "N/A"
            logger.debug("[STATE] %s", flags)
        except Exception:
            logger.exception("Ошибка в фоновом наблюдателе")
        _monitor_stop_event.wait(2.0)


def install_seek_monitoring():
    """Включает мониторинг (без объектов)."""
    logger.info("Мониторинг Seek включён (ожидание объектов)")
    # Запускаем перехват исключений
    sys.excepthook = lambda *args: logger.critical(
        "Необработанное исключение в главном потоке:\n%s",
        traceback.format_exception(*args)
    )


def attach_to_controller(controller):
    """Прикрепляет мониторинг к StreamController и его компонентам."""
    _monitor_stream_controller(controller)
    if controller._playback:
        _monitor_playback_engine(controller._playback)
    if controller._seek_engine:
        _monitor_seek_engine(controller._seek_engine)

    watcher = threading.Thread(
        target=_background_watcher,
        args=(controller._playback, controller._seek_engine),
        daemon=True,
        name="SeekMonitorWatcher"
    )
    watcher.start()
    logger.info("Мониторинг Seek прикреплён к StreamController")


def patch_close_event(widget_class):
    """Обёртка closeEvent для логирования стека вызовов."""
    if not hasattr(widget_class, 'closeEvent'):
        return
    original_close = widget_class.closeEvent

    def new_close(self, event):
        logger.info(
            "closeEvent вызван для %s. Стек:\n%s",
            widget_class.__name__,
            traceback.format_stack()[-4:-2]
        )
        try:
            original_close(self, event)
        except Exception as e:
            _log_exception(e, f"closeEvent {widget_class.__name__}")

    widget_class.closeEvent = new_close
    logger.info("closeEvent обёрнут для %s", widget_class.__name__)


def stop_monitoring():
    """Останавливает фоновый наблюдатель."""
    _monitor_stop_event.set()
    logger.info("Мониторинг Seek остановлен")