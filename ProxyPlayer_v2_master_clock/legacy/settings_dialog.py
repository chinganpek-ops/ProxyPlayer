"""
settings_dialog.py – диалог настроек плеера с вкладкой дорожек (production).
"""

import logging
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QFormLayout,
    QSpinBox, QDoubleSpinBox, QPushButton, QWidget, QComboBox, QCheckBox,
    QSlider, QLabel
)
from PyQt5.QtCore import Qt
from config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class SettingsDialog(QDialog):
    def __init__(self, current_settings: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Настройки плеера")
        self.resize(500, 550)
        self.settings = current_settings.copy()
        self.default_settings = DEFAULT_CONFIG.copy()
        self._build_ui()
        self._load_values()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        self.tabs = QTabWidget()

        # --- Вкладка "Буфер" ---
        buf_tab = QWidget()
        buf_layout = QFormLayout(buf_tab)
        self.buffer_size = QSpinBox()
        self.buffer_size.setRange(50, 1000)
        buf_layout.addRow("Размер буфера (кадров)", self.buffer_size)

        self.free_slots = QSpinBox()
        self.free_slots.setRange(5, 100)
        buf_layout.addRow("Свободных слотов для загрузки", self.free_slots)

        self.drop_back = QSpinBox()
        self.drop_back.setRange(10, 500)
        buf_layout.addRow("Окно удержания назад (кадров)", self.drop_back)

        self.drop_forward = QSpinBox()
        self.drop_forward.setRange(10, 500)
        buf_layout.addRow("Окно удержания вперёд (кадров)", self.drop_forward)
        self.tabs.addTab(buf_tab, "Буфер")

        # --- Вкладка "Скорость" ---
        speed_tab = QWidget()
        speed_layout = QFormLayout(speed_tab)
        self.normal_speed = QDoubleSpinBox()
        self.normal_speed.setRange(0.5, 10.0)
        self.normal_speed.setSingleStep(0.05)
        speed_layout.addRow("Коэф. скорости (обычный)", self.normal_speed)

        self.live_speed = QDoubleSpinBox()
        self.live_speed.setRange(0.5, 10.0)
        self.live_speed.setSingleStep(0.05)
        speed_layout.addRow("Коэф. скорости (live)", self.live_speed)

        self.rate_limit = QSpinBox()
        self.rate_limit.setRange(0, 100_000_000)
        self.rate_limit.setSingleStep(100_000)
        speed_layout.addRow("Лимит скорости (байт/с, 0=без)", self.rate_limit)
        self.tabs.addTab(speed_tab, "Скорость")

        # --- Вкладка "Live" ---
        live_tab = QWidget()
        live_layout = QFormLayout(live_tab)
        self.live_delay = QDoubleSpinBox()
        self.live_delay.setRange(0.1, 10.0)
        self.live_delay.setSingleStep(0.1)
        live_layout.addRow("Задержка live (сек)", self.live_delay)
        self.tabs.addTab(live_tab, "Live")

        # --- Вкладка "Общие" ---
        gen_tab = QWidget()
        gen_layout = QFormLayout(gen_tab)
        self.auto_refresh = QSpinBox()
        self.auto_refresh.setRange(1, 120)
        gen_layout.addRow("Автообновление индекса (сек)", self.auto_refresh)

        self.fps = QDoubleSpinBox()
        self.fps.setRange(10.0, 60.0)
        gen_layout.addRow("Частота кадров", self.fps)

        self.group_chunks = QSpinBox()
        self.group_chunks.setRange(1, 100)
        gen_layout.addRow("Чанков за раз", self.group_chunks)

        self.max_retries = QSpinBox()
        self.max_retries.setRange(1, 10)
        gen_layout.addRow("Повторов чтения", self.max_retries)
        self.tabs.addTab(gen_tab, "Общие")

        # --- Вкладка "Производительность" ---
        perf_tab = QWidget()
        perf_layout = QFormLayout(perf_tab)

        self.thread_type = QComboBox()
        self.thread_type.addItems(["AUTO", "FRAME", "SLICE"])
        self.thread_type.setToolTip("Режим многопоточности декодера")
        perf_layout.addRow("Тип многопоточности", self.thread_type)

        self.thread_count = QSpinBox()
        self.thread_count.setRange(0, 32)
        perf_layout.addRow("Число потоков", self.thread_count)

        self.skip_frame = QCheckBox("Пропускать B-кадры")
        perf_layout.addRow(self.skip_frame)

        self.gpu_mode = QComboBox()
        self.gpu_mode.addItems(["Выключен", "Включен", "Авто"])
        perf_layout.addRow("GPU-декодер", self.gpu_mode)
        self.tabs.addTab(perf_tab, "Производительность")

        # --- Вкладка "Аудио" ---
        audio_tab = QWidget()
        audio_layout = QFormLayout(audio_tab)

        self.audio_enabled = QCheckBox("Включить звук")
        audio_layout.addRow(self.audio_enabled)

        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_label = QLabel("80%")
        self.volume_slider.valueChanged.connect(lambda v: self.volume_label.setText(f"{v}%"))
        vol_row = QHBoxLayout()
        vol_row.addWidget(self.volume_slider)
        vol_row.addWidget(self.volume_label)
        audio_layout.addRow("Громкость", vol_row)

        self.audio_buffer_sec = QDoubleSpinBox()
        self.audio_buffer_sec.setRange(1.0, 30.0)
        self.audio_buffer_sec.setSingleStep(1.0)
        self.audio_buffer_sec.setToolTip("Размер аудиобуфера в секундах (10 сек = 480000 сэмплов)")
        audio_layout.addRow("Буфер аудио (сек)", self.audio_buffer_sec)

        self.audio_device = QComboBox()
        self.audio_device.addItem("По умолчанию")
        self.audio_device.setToolTip("Выбор устройства вывода звука (пока только по умолчанию)")
        audio_layout.addRow("Устройство", self.audio_device)
        self.tabs.addTab(audio_tab, "Аудио")

        # --- Вкладка "Дорожки" ---
        tracks_tab = QWidget()
        tracks_layout = QVBoxLayout(tracks_tab)
        tracks_layout.addWidget(QLabel("Выберите дорожки по умолчанию (2..5):"))
        self.track_checkboxes = {}
        for track_id in range(2, 6):
            cb = QCheckBox(f"Дорожка {track_id - 1}")
            self.track_checkboxes[track_id] = cb
            tracks_layout.addWidget(cb)
        tracks_layout.addStretch()
        self.tabs.addTab(tracks_tab, "Дорожки")

        main_layout.addWidget(self.tabs)

        # --- Кнопки ---
        btn_layout = QHBoxLayout()
        default_btn = QPushButton("По умолчанию")
        default_btn.clicked.connect(self._reset_to_defaults)
        save_btn = QPushButton("Сохранить")
        save_btn.clicked.connect(self._save_and_close)
        cancel_btn = QPushButton("Отмена")
        cancel_btn.clicked.connect(self.reject)

        btn_layout.addWidget(default_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        main_layout.addLayout(btn_layout)

    def _load_values(self):
        s = self.settings
        self.buffer_size.setValue(s.get('buffer_size', 300))
        self.free_slots.setValue(s.get('free_slots_required', 25))
        self.drop_back.setValue(s.get('drop_window_backward', 144))
        self.drop_forward.setValue(s.get('drop_window_forward', 144))
        self.normal_speed.setValue(s.get('normal_playback_speed_factor', 1.25))
        self.live_speed.setValue(s.get('live_speed_factor', 1.15))
        self.rate_limit.setValue(s.get('default_rate_limit', 0))
        self.live_delay.setValue(s.get('live_delay', 1.0))
        self.auto_refresh.setValue(s.get('auto_refresh_interval', 10))
        self.fps.setValue(s.get('fps', 25.0))
        self.group_chunks.setValue(s.get('group_chunks', 20))
        self.max_retries.setValue(s.get('max_retries', 3))

        thread_type = s.get('thread_type', 'AUTO')
        idx = self.thread_type.findText(thread_type)
        if idx >= 0:
            self.thread_type.setCurrentIndex(idx)
        self.thread_count.setValue(s.get('thread_count', 0))
        self.skip_frame.setChecked(s.get('skip_frame', False))
        gpu_mode = s.get('use_gpu_decoder', 'off')
        mapping = {"off": 0, "on": 1, "auto": 2}
        self.gpu_mode.setCurrentIndex(mapping.get(gpu_mode, 0))

        # Аудио
        self.audio_enabled.setChecked(s.get('audio_enabled', True))
        vol = int(s.get('volume', 0.8) * 100)
        self.volume_slider.setValue(vol)
        self.volume_label.setText(f"{vol}%")
        self.audio_buffer_sec.setValue(s.get('audio_buffer_sec', 10.0))

        # Дорожки
        active_tracks = s.get('active_tracks', [2, 3])
        for track_id, cb in self.track_checkboxes.items():
            cb.setChecked(track_id in active_tracks)

    def _reset_to_defaults(self):
        logger.info("Сброс настроек на значения по умолчанию")
        self.settings = self.default_settings.copy()
        self._load_values()

    def _save_and_close(self):
        # Собираем все настройки, валидируем и логируем
        new_settings = {
            'buffer_size': self.buffer_size.value(),
            'free_slots_required': self.free_slots.value(),
            'drop_window_backward': self.drop_back.value(),
            'drop_window_forward': self.drop_forward.value(),
            'normal_playback_speed_factor': self.normal_speed.value(),
            'live_speed_factor': self.live_speed.value(),
            'default_rate_limit': self.rate_limit.value(),
            'live_delay': self.live_delay.value(),
            'auto_refresh_interval': self.auto_refresh.value(),
            'fps': self.fps.value(),
            'group_chunks': self.group_chunks.value(),
            'max_retries': self.max_retries.value(),
            'thread_type': self.thread_type.currentText(),
            'thread_count': self.thread_count.value(),
            'skip_frame': self.skip_frame.isChecked(),
            'use_gpu_decoder': {0: "off", 1: "on", 2: "auto"}[self.gpu_mode.currentIndex()],
            'audio_enabled': self.audio_enabled.isChecked(),
            'volume': self.volume_slider.value() / 100.0,
            'audio_buffer_sec': self.audio_buffer_sec.value(),
            'audio_device': "default",
            'active_tracks': [tid for tid, cb in self.track_checkboxes.items() if cb.isChecked()]
        }

        # Гарантируем, что хотя бы одна дорожка выбрана
        if not new_settings['active_tracks']:
            new_settings['active_tracks'] = [2, 3]
            logger.warning("Не выбрано ни одной дорожки, установлены 2 и 3 по умолчанию.")

        # Проверка диапазонов (громкость, буфер)
        if not (0.0 <= new_settings['volume'] <= 1.0):
            logger.warning(f"Некорректная громкость {new_settings['volume']}, исправлена на 0.8")
            new_settings['volume'] = 0.8
        if not (1.0 <= new_settings['audio_buffer_sec'] <= 30.0):
            logger.warning(f"Некорректный размер аудиобуфера {new_settings['audio_buffer_sec']}, исправлен на 10.0")
            new_settings['audio_buffer_sec'] = 10.0

        # Логируем изменения относительно текущих
        for key in new_settings:
            old = self.settings.get(key)
            if old != new_settings[key]:
                logger.info(f"Параметр '{key}' изменён: {old!r} -> {new_settings[key]!r}")

        self.settings = new_settings
        self.accept()

    def get_settings(self) -> dict:
        return self.settings