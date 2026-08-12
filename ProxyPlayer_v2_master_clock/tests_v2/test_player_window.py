"""
test_player_window.py – GUI тесты для стабильности интерфейса.
Проверяет: Play/Pause, слайдер, _seek_pending, блокировку кнопок при seek.
"""

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from unittest.mock import MagicMock, PropertyMock

from player_window import PlayerWidget


@pytest.fixture
def player_widget(qtbot, monkeypatch):
    """Создаёт PlayerWidget с замоканным StreamController."""
    # Мокаем конструктор StreamController, чтобы не требовались реальные файлы
    mock_controller = MagicMock()
    mock_controller.playing = False
    mock_controller._ready = MagicMock()
    mock_controller._ready.is_set.return_value = True
    mock_controller.total_frames = 1000
    mock_controller.start_frame_offset = 0
    mock_controller.fps = 25.0
    mock_controller.audio_clock = 0
    mock_controller.get_local_timecode_str.return_value = "00:00:00;00"
    mock_controller.get_real_timecode_str.return_value = "00:00:00;00"
    mock_controller.get_seek_speed_display.return_value = "▶ x1"
    mock_controller.master_clock = MagicMock()

    monkeypatch.setattr(
        "player_window.StreamController",
        lambda *args, **kwargs: mock_controller
    )
    # Передаём фейковые пути
    widget = PlayerWidget(
        mp4_path=MagicMock(),
        config={"fps": 25.0, "buffer_size": 600, "active_tracks": [2, 3]},
        use_moov=False,
        mirror_path=None
    )
    # Пропускаем инициализацию
    widget.player = mock_controller
    widget._active_player = mock_controller
    qtbot.addWidget(widget)
    return widget


class TestPlayerWindow:
    def test_play_pause_toggles_state(self, player_widget):
        player_widget.player.playing = False
        QTest.mouseClick(player_widget.play_pause_btn, Qt.LeftButton)
        player_widget.player.toggle_pause.assert_called_once()

    def test_slider_does_not_auto_play(self, player_widget):
        # Симулируем нажатие на слайдер
        player_widget._on_slider_pressed()
        assert player_widget._seek_pending is True
        assert player_widget.player.playing is False

        # Отпускаем слайдер
        player_widget._on_slider_released()
        # Плеер должен остаться на паузе, seek вызван
        player_widget.player.seek_absolute.assert_called_once()
        assert player_widget.player.playing is False
        # _seek_pending всё ещё True (ждёт явного Play)
        assert player_widget._seek_pending is True

        # Нажимаем Play — возобновление
        player_widget.player.playing = False
        player_widget._toggle_play_pause()
        assert player_widget._seek_pending is False
        player_widget.player.toggle_pause.assert_called()

    def test_double_click_play_during_seek(self, player_widget):
        # Симулируем состояние seek
        player_widget._seeking = True
        # Два быстрых клика по Play
        QTest.mouseClick(player_widget.play_pause_btn, Qt.LeftButton)
        QTest.mouseClick(player_widget.play_pause_btn, Qt.LeftButton)
        # toggle_pause не должен вызываться, пока идёт seek
        player_widget.player.toggle_pause.assert_not_called()