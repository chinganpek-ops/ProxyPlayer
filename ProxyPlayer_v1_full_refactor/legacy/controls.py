"""
controls.py – панель управления плеера с выпадающим меню выбора дорожек,
кнопкой Mute и опциональной кнопкой настроек.
Все кнопки и слайдеры не забирают фокус, чтобы не блокировать
клавиатурное управление плеером.
"""

import logging
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QLabel, QPushButton, QSlider, QMenu, QAction, QLineEdit
)
from PyQt5.QtGui import QFont

logger = logging.getLogger(__name__)


def build_controls(parent=None, show_settings_button=False):
    widgets = {}

    # --- Таймкод ---
    tc_label = QLabel("00:00:00;00", parent)
    tc_label.setFont(QFont("Courier", 14))
    tc_label.setAlignment(Qt.AlignCenter)
    tc_label.setStyleSheet("color: #ffffff;")
    widgets['tc_label'] = tc_label

    # --- Поле ввода таймкода ---
    tc_input = QLineEdit(parent)
    tc_input.setPlaceholderText("00:00:00;00")
    #tc_input.setInputMask("00:00:00;00")
    tc_input.setFont(QFont("Courier", 12))
    tc_input.setAlignment(Qt.AlignCenter)
    tc_input.setStyleSheet("""
        QLineEdit {
            color: #ffffff;
            background-color: #3a3a3a;
            border: 1px solid #555555;
            padding: 2px;
            max-width: 120px;
        }
    """)
    tc_input.setFocusPolicy(Qt.StrongFocus)
    widgets['tc_input'] = tc_input
    tc_input.hide()

    # --- Информационная строка ---
    #bitrate_label = QLabel("Битрейт: ---", parent)
    #bitrate_label.setFont(QFont("Courier", 8))
    #widgets['bitrate_label'] = bitrate_label

    #speed_label = QLabel("Скорость: ---", parent)
    #speed_label.setFont(QFont("Courier", 8))
    #widgets['speed_label'] = speed_label

    # --- Кнопки настроек (опционально) ---
    if show_settings_button:
        settings_btn = QPushButton("⚙️", parent)
        settings_btn.setFixedSize(30, 30)
        settings_btn.setFocusPolicy(Qt.NoFocus)
        widgets['settings_btn'] = settings_btn
    else:
        widgets['settings_btn'] = None

    aspect_btn = QPushButton("📐", parent)
    aspect_btn.setFixedSize(60, 30)
    aspect_btn.setFocusPolicy(Qt.NoFocus)
    widgets['aspect_btn'] = aspect_btn

    #log_btn = QPushButton("📋", parent)
    #log_btn.setFixedSize(30, 30)
    #log_btn.setFocusPolicy(Qt.NoFocus)
    #widgets['log_btn'] = log_btn

    # --- Слайдер позиции ---
    slider = QSlider(Qt.Horizontal, parent)
    slider.setRange(0, 1000)
    slider.setValue(0)
    slider.setFocusPolicy(Qt.NoFocus)
    widgets['slider'] = slider

    # --- Кнопки управления воспроизведением ---
    play_pause_btn = QPushButton("⏸ Pause", parent)
    play_pause_btn.setFocusPolicy(Qt.NoFocus)
    widgets['play_pause_btn'] = play_pause_btn

    live_btn = QPushButton("🔴 Live", parent)
    live_btn.setFocusPolicy(Qt.NoFocus)
    widgets['live_btn'] = live_btn

    rew_btn = QPushButton("⏪ -5s", parent)
    rew_btn.setFocusPolicy(Qt.NoFocus)
    widgets['rew_btn'] = rew_btn

    fwd_btn = QPushButton("⏩ +5s", parent)
    fwd_btn.setFocusPolicy(Qt.NoFocus)
    widgets['fwd_btn'] = fwd_btn

    step_back_btn = QPushButton("⏮", parent)
    step_back_btn.setFocusPolicy(Qt.NoFocus)
    widgets['step_back_btn'] = step_back_btn

    step_fwd_btn = QPushButton("⏭", parent)
    step_fwd_btn.setFocusPolicy(Qt.NoFocus)
    widgets['step_fwd_btn'] = step_fwd_btn

    tc_btn = QPushButton("TC: Лок", parent)
    tc_btn.setFocusPolicy(Qt.NoFocus)
    widgets['tc_btn'] = tc_btn

    open_btn = QPushButton("📂 Open...", parent)
    open_btn.setFocusPolicy(Qt.NoFocus)
    widgets['open_btn'] = open_btn

    # --- Слайдер громкости ---
    volume_slider = QSlider(Qt.Horizontal, parent)
    volume_slider.setRange(0, 100)
    volume_slider.setValue(80)
    volume_slider.setToolTip("Громкость")
    volume_slider.setFixedWidth(120)
    volume_slider.setFocusPolicy(Qt.NoFocus)
    widgets['volume_slider'] = volume_slider

    # --- Кнопка Mute ---
    mute_btn = QPushButton("🔊", parent)
    mute_btn.setToolTip("Включить/выключить звук")
    mute_btn.setFixedSize(40, 30)
    mute_btn.setFocusPolicy(Qt.NoFocus)
    widgets['mute_btn'] = mute_btn

    # --- Кнопка выбора дорожек ---
    track_btn = QPushButton("🔊 Дорожки", parent)
    track_btn.setToolTip("Выбрать активные аудиодорожки")
    track_btn.setFocusPolicy(Qt.NoFocus)
    widgets['track_btn'] = track_btn

    # Создаём меню с checkable QAction (дорожки 2..5, отображаются как 1..4)
    track_menu = QMenu(parent)
    track_actions = {}
    for track_id in range(2, 6):
        action = QAction(f"Дорожка {track_id - 1}", parent, checkable=True)
        action.setChecked(track_id in (2, 3))  # по умолчанию дорожки 1 и 2 (ID 2,3)
        track_menu.addAction(action)
        track_actions[track_id] = action

    track_btn.setMenu(track_menu)

    widgets['track_menu'] = track_menu
    widgets['track_actions'] = track_actions

    logger.debug("Панель управления создана (все кнопки без фокуса, добавлены Mute и поле ввода таймкода)")
    return widgets