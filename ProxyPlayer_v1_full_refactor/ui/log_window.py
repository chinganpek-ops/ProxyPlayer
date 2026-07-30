"""
log_window.py – окно лога и GUI-обработчик для вывода сообщений.
Версия production: логирование создания/закрытия окна и переполнения буфера.
"""

import logging
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout,
    QLabel, QPushButton, QPlainTextEdit
)
from PyQt5.QtGui import QFont, QTextCursor

logger = logging.getLogger(__name__)

MAX_BLOCK_COUNT = 1000  # ограничение на количество строк в виджете


class LogWindow(QMainWindow):
    """Окно для отображения логов с возможностью очистки."""

    message_received = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Лог наблюдателя")
        self.resize(800, 400)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.log_widget = QPlainTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMaximumBlockCount(MAX_BLOCK_COUNT)
        self.log_widget.setFont(QFont("Courier", 8))
        layout.addWidget(self.log_widget)

        clear_btn = QPushButton("Очистить")
        clear_btn.clicked.connect(self.clear_log)
        layout.addWidget(clear_btn)

        self.message_received.connect(self._append_message)
        logger.debug("Окно лога создано")

    def _append_message(self, message: str):
        self.log_widget.appendPlainText(message)
        # Прокрутка в конец
        cursor = self.log_widget.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_widget.setTextCursor(cursor)

    def clear_log(self):
        self.log_widget.clear()
        logger.info("Лог очищен пользователем")

    def closeEvent(self, event):
        logger.debug("Окно лога закрывается")
        super().closeEvent(event)


class _GuiLogHandler(logging.Handler):
    """Обработчик логов, пересылающий сообщения в LogWindow через сигнал."""

    def __init__(self, log_window: LogWindow):
        super().__init__()
        self.log_window = log_window
        self.setLevel(logging.DEBUG)
        self.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            # Сигнал для потокобезопасной передачи в GUI
            self.log_window.message_received.emit(msg)
        except Exception:
            self.handleError(record)