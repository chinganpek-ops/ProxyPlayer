"""
video_widget.py – виджет для отображения видео через QOpenGLWidget (production).
Содержит класс GLVideoWidget и вспомогательную функцию numpy_to_qimage.
Строгая проверка входных данных, детальное логирование, поддержка aspect ratio.
Водяной знак в нижнем правом углу: мельче, прозрачнее, прижат к низу.
Поддержка заглушки (placeholder) с иконкой и анимированными точками при загрузке.
Добавлена статичная заглушка "Трансляция завершена".
"""

import logging
import os
import numpy as np
from PyQt5.QtCore import QRect, QSize, Qt, QTimer
from PyQt5.QtGui import QImage, QPainter, QFont, QColor, QPixmap, QIcon
from PyQt5.QtWidgets import QOpenGLWidget, QSizePolicy

logger = logging.getLogger(__name__)


def numpy_to_qimage(img: np.ndarray) -> QImage:
    """
    Преобразует numpy-массив RGB24 (H, W, 3) uint8 в QImage без копирования,
    если массив C-contiguous. Иначе делает копию.
    При некорректном входе возвращает QImage и логирует ошибку.
    """
    if img is None or not isinstance(img, np.ndarray):
        logger.error("numpy_to_qimage: входное изображение None или не ndarray")
        return QImage()

    if img.ndim != 3 or img.shape[2] != 3:
        logger.error(f"numpy_to_qimage: ожидался 3-канальный RGB, получена форма {img.shape}")
        return QImage()

    if img.dtype != np.uint8:
        logger.warning(f"numpy_to_qimage: неверный dtype {img.dtype}, ожидался uint8 – попытка преобразования")
        img = np.clip(img, 0, 255).astype(np.uint8)

    h, w, ch = img.shape
    bytes_per_line = ch * w

    if img.flags['C_CONTIGUOUS']:
        return QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
    else:
        logger.debug("numpy_to_qimage: преобразование в C-contiguous")
        contiguous = np.ascontiguousarray(img)
        return QImage(contiguous.data, w, h, bytes_per_line, QImage.Format_RGB888)


class GLVideoWidget(QOpenGLWidget):
    """Виджет для отрисовки видео с поддержкой aspect ratio, водяным знаком и заглушкой."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: QImage = None
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(640, 360)
        self.aspect_mode = '16:9'  # 'fit', '4:3', '16:9'
        self._watermark_text = "Protected by YaroslavBurtsev"

        # Режим заглушки
        self._show_placeholder = False
        self._placeholder_icon: QPixmap = None
        self._placeholder_dots = 0  # 0, 1, 2, 3, -1 = завершено

        # Таймер анимации точек
        self._dots_timer = QTimer(self)
        self._dots_timer.timeout.connect(self._animate_dots)
        self._dots_interval = 500  # мс

        # Загружаем иконку из файла icon.ico (если он есть в корне проекта)
        icon_path = os.path.join(os.path.dirname(__file__), 'icon.ico')
        if os.path.exists(icon_path):
            icon = QIcon(icon_path)
            self._placeholder_icon = icon.pixmap(64, 64)  # фиксированный размер иконки
        else:
            # Создаём простую заглушку, если иконка не найдена
            self._placeholder_icon = QPixmap(64, 64)
            self._placeholder_icon.fill(QColor(80, 80, 80))

        logger.info("GLVideoWidget создан")

    def cleanup(self):
        """Останавливает и корректно удаляет внутренние таймеры."""
        if hasattr(self, '_dots_timer') and self._dots_timer is not None:
            self._dots_timer.stop()
            self._dots_timer.deleteLater()
            self._dots_timer = None

    def show_placeholder(self):
        """Показывает заглушку с иконкой и анимированными точками."""
        self._show_placeholder = True
        self._placeholder_dots = 0
        self._dots_timer.start(self._dots_interval)
        self.update()

    def hide_placeholder(self):
        """Скрывает заглушку, возвращая обычный рендеринг видео."""
        self._show_placeholder = False
        self._dots_timer.stop()
        self._placeholder_dots = 0
        self.update()

    def show_transmission_ended(self):
        """Показывает статичную заглушку 'Трансляция завершена'."""
        self._show_placeholder = True
        self._placeholder_dots = -1   # специальное значение: завершено
        self._dots_timer.stop()
        self.update()

    def _animate_dots(self):
        """Циклически меняет количество точек в анимации (только если не завершено)."""
        if self._placeholder_dots >= 0:
            self._placeholder_dots = (self._placeholder_dots + 1) % 4
        self.update()

    def set_frame(self, img: np.ndarray):
        if self._show_placeholder:
            # Если заглушка активна, игнорируем кадры
            return
        if img is None:
            logger.debug("set_frame: img is None, пропускаем")
            return
        if img.size == 0:
            logger.warning("set_frame: пустой массив")
            return

        qimage = numpy_to_qimage(img)
        if qimage.isNull():
            logger.warning("set_frame: не удалось создать QImage")
            return

        self._image = qimage
        self.update()
        logger.debug(f"set_frame: новый кадр {img.shape[1]}x{img.shape[0]}")

    def paintGL(self):
        super().paintGL()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        if self._show_placeholder:
            self._draw_placeholder(painter)
        elif self._image is not None and not self._image.isNull():
            src_rect = QRect(0, 0, self._image.width(), self._image.height())
            dst_rect = self._calculate_destination_rect(self.size())
            painter.drawImage(dst_rect, self._image, src_rect)
            self._draw_watermark(painter, dst_rect)

        painter.end()

    def _draw_placeholder(self, painter: QPainter):
        """Рисует заглушку с иконкой, текстом (и анимированными точками или без)."""
        widget_rect = self.rect()
        center_x = widget_rect.center().x()
        center_y = widget_rect.center().y()

        # Иконка по центру
        if self._placeholder_icon:
            icon_size = self._placeholder_icon.size()
            icon_x = center_x - icon_size.width() // 2
            icon_y = center_y - icon_size.height() // 2 - 30  # чуть выше текста
            painter.drawPixmap(icon_x, icon_y, self._placeholder_icon)

        # Текст
        font = QFont("Arial", 14, QFont.Bold)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255, 200))

        if self._placeholder_dots == -1:
            text = "Трансляция завершена"
        else:
            text = "Загрузка медиа"

        fm = painter.fontMetrics()
        text_width = fm.horizontalAdvance(text)
        text_x = center_x - text_width // 2
        text_y = center_y + icon_size.height() // 2 + 10  # под иконкой
        painter.drawText(text_x, text_y, text)

        # Анимированные точки (только если не завершено)
        if self._placeholder_dots >= 0:
            dots = "." * self._placeholder_dots + " " * (3 - self._placeholder_dots)
            font_dots = QFont("Arial", 18, QFont.Bold)
            painter.setFont(font_dots)
            dots_width = painter.fontMetrics().horizontalAdvance(dots)
            painter.drawText(center_x - dots_width // 2, text_y + 25, dots)

    def _draw_watermark(self, painter: QPainter, video_rect: QRect):
        """Рисует прозрачный водяной знак в правом нижнем углу видео."""
        font_size = max(6, min(14, video_rect.height() // 36))
        font = QFont("Arial", font_size)
        font.setBold(False)
        painter.setFont(font)

        text = self._watermark_text
        fm = painter.fontMetrics()
        text_width = fm.horizontalAdvance(text)

        margin = 4
        x = video_rect.right() - text_width - margin
        y = video_rect.bottom() - margin - 2

        painter.setPen(QColor(0, 0, 0, 80))
        painter.drawText(x + 1, y + 1, text)

        painter.setPen(QColor(255, 255, 255, 100))
        painter.drawText(x, y, text)

    def _calculate_destination_rect(self, widget_size: QSize) -> QRect:
        if self._image is None or self._image.isNull():
            return QRect(0, 0, widget_size.width(), widget_size.height())

        img_w = self._image.width()
        img_h = self._image.height()
        if img_w == 0 or img_h == 0:
            return QRect(0, 0, widget_size.width(), widget_size.height())

        w = widget_size.width()
        h = widget_size.height()

        if self.aspect_mode == '4:3':
            target_aspect = 4.0 / 3.0
        elif self.aspect_mode == '16:9':
            target_aspect = 16.0 / 9.0
        else:
            target_aspect = img_w / img_h

        aspect_widget = w / h
        if target_aspect > aspect_widget:
            draw_w = w
            draw_h = int(w / target_aspect)
        else:
            draw_h = h
            draw_w = int(h * target_aspect)

        x = (w - draw_w) // 2
        y = (h - draw_h) // 2

        logger.debug(f"_calculate_destination_rect: виджет {w}x{h}, картинка {img_w}x{img_h}, "
                     f"aspect={self.aspect_mode}, rect=({x},{y},{draw_w},{draw_h})")
        return QRect(x, y, draw_w, draw_h)