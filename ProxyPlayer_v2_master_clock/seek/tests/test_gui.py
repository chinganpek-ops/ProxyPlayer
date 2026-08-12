#!/usr/bin/env python3
"""
Программный тест GUI плеера.
Проверяет инициализацию, отображение первого кадра и таймкод.
Запускается через pytest.
"""

import os
import sys
from pathlib import Path
import pytest
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from player_window import PlayerWidget
from config.config import load_config

TEST_MP4 = os.environ.get('TEST_MP4_PATH')
if not TEST_MP4:
    pytest.skip("TEST_MP4_PATH не задан", allow_module_level=True)


@pytest.fixture(scope="module")
def app():
    """Создаёт QApplication один раз для всех тестов."""
    app = QApplication.instance() or QApplication(sys.argv)
    yield app
    app.quit()


@pytest.fixture(scope="module")
def player_widget(app):
    """Создаёт и инициализирует плеер."""
    mp4_path = Path(TEST_MP4)
    config = load_config()
    widget = PlayerWidget(mp4_path, config)
    widget.show()
    # Ожидание готовности StreamController
    if not widget.player._ready.wait(timeout=120):
        widget.close()
        pytest.fail("StreamController не инициализировался за 120 секунд")
    # Даём время на первый кадр
    QTest.qWait(3000)
    yield widget
    widget.close()


class TestGUI:
    def test_first_frame(self, player_widget):
        """Проверяет, что после старта появляется кадр."""
        frame = player_widget.player.get_display_frame()
        assert frame is not None, "Нет кадра после старта"
        assert frame.shape == (576, 720, 3), "Неверный размер кадра"

    def test_timecode_display(self, player_widget):
        """Проверяет, что таймкод отображается и не равен 00:00:00;00."""
        tc = player_widget.player.get_local_timecode_str()
        assert tc != "00:00:00;00", "Таймкод не изменился после старта"

    def test_toggle_pause(self, player_widget):
        """Проверяет работу паузы."""
        # Запускаем воспроизведение (если на паузе)
        if not player_widget.player.playing:
            player_widget.player.resume()
            QTest.qWait(500)
        assert player_widget.player.playing, "Плеер не играет"
        # Пауза
        player_widget.player.pause()
        QTest.qWait(500)
        assert not player_widget.player.playing, "Плеер не встал на паузу"

    def test_seek_by_timeline(self, player_widget):
        """Проверяет перемотку через слайдер (seek_absolute)."""
        total = player_widget.player.total_frames
        if total < 100:
            pytest.skip("Файл слишком короткий для теста перемотки")
        target = min(500, total - 1)
        player_widget.player.seek_absolute(target)
        QTest.qWait(2000)
        frame = player_widget.player.get_display_frame()
        assert frame is not None, "Нет кадра после перемотки"

    def test_jkl_keys(self, player_widget):
        """Проверяет JKL-перемотку (хотя бы вызов методов)."""
        # J (назад)
        player_widget.player.set_seek_speed(-1)
        display = player_widget.player.get_seek_speed_display()
        assert "x2" in display or "x4" in display or "x8" in display
        # K (стоп)
        player_widget.player.reset_seek_speed()
        assert "x1" in player_widget.player.get_seek_speed_display()
        # L (вперёд)
        player_widget.player.set_seek_speed(1)
        display = player_widget.player.get_seek_speed_display()
        assert "x2" in display or "x4" in display or "x8" in display
        player_widget.player.reset_seek_speed()