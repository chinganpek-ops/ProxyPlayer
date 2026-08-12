#!/usr/bin/env python3
"""
player_window.py – главное окно плеера и менеджер окон (ProxyPlayer v2).
Адаптирован под MasterClock: mute и переключение дорожек работают через master_clock.
Исправления v3:
- Единая точка seek (_start_seek) с поколением запросов.
- Кнопка Play блокируется до завершения актуального seek.
- Кнопка Live, слайдер, таймкод – всё идёт через _start_seek.
- Устранена лавина потоков при многократных нажатиях.
"""

import sys
import os
import re
import time
import ctypes
from ctypes import wintypes
from pathlib import Path
import logging
from PyQt5.QtCore import Qt, QTimer, QProcess, QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QMessageBox, QFileDialog, QAction, QDesktopWidget, QToolBar, QPushButton,
    QSystemTrayIcon, QMenu, QStyle, QDialog
)
from PyQt5.QtGui import QIcon, QMouseEvent
from PyQt5.QtNetwork import QLocalServer, QLocalSocket, QTcpServer, QTcpSocket
from PyQt5.QtWidgets import QApplication

from core.stream_controller import StreamController
from config.config import save_config
from output.video_widget import GLVideoWidget
from ui.controls import build_controls
from index.idx_cache import prepare_mirror
from config.timebase import pts_to_video_frame

logger = logging.getLogger(__name__)

# Windows API
user32 = ctypes.windll.user32
MoveWindow = user32.MoveWindow
MoveWindow.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.BOOL]
MoveWindow.restype = wintypes.BOOL
SetWindowPos = user32.SetWindowPos
SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.UINT]
SetWindowPos.restype = wintypes.BOOL
GetWindowLongW = user32.GetWindowLongW
GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
GetWindowLongW.restype = wintypes.LONG
SetWindowLongW = user32.SetWindowLongW
SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.LONG]
SetWindowLongW.restype = wintypes.LONG

HWND_TOP = 0
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040
GWL_EXSTYLE = -20
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000

PANEL_WIDTH = 30
COLS = 3
ROWS = 2
MAX_PLAYERS = COLS * ROWS


class OpenPlayerWorker(QThread):
    found_signal = pyqtSignal(object)

    def __init__(self, manager, file_id, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.file_id = file_id

    def run(self):
        try:
            mp4_path = self.manager._resolve_id_to_mp4(self.file_id)
            self.found_signal.emit(mp4_path)
        except Exception as e:
            self.found_signal.emit(None)


class PlayerWidget(QWidget):
    def __init__(self, mp4_path: Path, config: dict, use_moov: bool = False, 
                 mirror_path: str = None, parent=None):
        super().__init__(parent)
        self.mp4_path = mp4_path
        self.use_moov = use_moov
        self.config = config
        self.mirror_path = mirror_path
        self._playback_started = False
        self._seek_pending = False          # true, если слайдер был отпущен и ждём Play
        self._seeking = False               # true, пока асинхронный seek не завершён
        self._seek_generation = 0           # увеличивается при каждом новом seek

        self.setStyleSheet("""
            background-color: #2b2b2b;
            QLabel { color: #ffffff; }
            QPushButton {
                background-color: #3a3a3a; color: #ffffff;
                border: 1px solid #555555; padding: 3px;
            }
            QPushButton:hover { background-color: #4a4a4a; }
            QSlider::groove:horizontal { background: #555555; height: 6px; }
            QSlider::handle:horizontal { background: #888888; width: 12px; margin: -4px 0; }
        """)

        self._muted = False
        self._updating_tracks = False
        self.tc_mode = 0
        self.active_tracks = config.get('active_tracks', [2, 3])
        self.active_tracks = [t for t in self.active_tracks if 2 <= t <= 5] or [2, 3]

        if not use_moov:
            self.ref_path, self.idx_path = self._find_related_files(mp4_path)
            if not self.idx_path.exists():
                QMessageBox.critical(self, "Ошибка", f"Отсутствует .idx для {mp4_path}")
                raise FileNotFoundError(f"Missing .idx for {mp4_path}")
        else:
            self.ref_path, self.idx_path = None, None

        self.player = self._create_controller()
        self._active_player = self.player
        self._init_ui()

        self._updating_tracks = True
        for tid, action in self.track_actions.items():
            action.setChecked(tid in self.active_tracks)
        self._updating_tracks = False

        if self.player and self.player._ready.is_set():
            self.player.set_active_tracks(self.active_tracks)

        self.render_timer = QTimer(self)
        self.render_timer.timeout.connect(self._update_frame)
        self._update_render_interval()

        self.video_widget.show_placeholder()
        if not self.player._ready.wait(timeout=120):
            QMessageBox.critical(self, "Ошибка", "Не удалось инициализировать плеер за 120 секунд")
            self.player.close()
            raise RuntimeError("StreamController initialization timeout")
        
        self.setFocusPolicy(Qt.StrongFocus)

    @staticmethod
    def _find_related_files(mp4_path):
        stem = mp4_path.stem
        parent = mp4_path.parent
        ref = parent / f"{stem}.mp4.ref"
        if not ref.exists(): ref = parent / f"{stem}.ref"
        idx = parent / "idx" / "mp4" / f"{stem}.idx"
        if not idx.exists(): idx = parent / f"{stem}.idx"
        return ref, idx

    def _create_controller(self):
        avcc_override = None
        avcc_hex = self.config.get('default_avcc', '')
        if avcc_hex:
            try:
                avcc_override = bytes.fromhex(avcc_hex)
            except ValueError:
                pass
        return StreamController(
            ref_path=self.ref_path,
            idx_path=self.idx_path,
            mp4_path=self.mp4_path,
            avcc_override=avcc_override,
            fps=self.config.get('fps', 25.0),
            buffer_size=self.config.get('buffer_size', 600),
            free_slots_required=self.config.get('free_slots_required', 25),
            group_chunks=self.config.get('group_chunks', 30),
            max_retries=self.config.get('max_retries', 3),
            thread_type=self.config.get('thread_type', 'AUTO'),
            thread_count=self.config.get('thread_count', 0),
            skip_frame=self.config.get('skip_frame', False),
            gpu_mode=self.config.get('use_gpu_decoder', 'off'),
            audio_delay_ms=self.config.get('audio_delay_ms', 0),
            start_from_live=True,
            mirror_path=self.mirror_path,
        )

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.video_widget = GLVideoWidget(self)
        main_layout.addWidget(self.video_widget, 1)

        ctrl = build_controls(show_settings_button=False)
        self.tc_label = ctrl['tc_label']
        self.tc_input = ctrl['tc_input']
        self.aspect_btn = ctrl['aspect_btn']
        self.slider = ctrl['slider']
        self.play_pause_btn = ctrl['play_pause_btn']
        self.live_btn = ctrl['live_btn']
        self.rew_btn = ctrl['rew_btn']
        self.fwd_btn = ctrl['fwd_btn']
        self.step_back_btn = ctrl['step_back_btn']
        self.step_fwd_btn = ctrl['step_fwd_btn']
        self.tc_btn = ctrl['tc_btn']
        self.settings_btn = ctrl.get('settings_btn')
        self.volume_slider = ctrl.get('volume_slider')
        self.mute_btn = ctrl.get('mute_btn')
        self.track_btn = ctrl['track_btn']
        self.track_actions = ctrl['track_actions']

        self.slider.setFocusPolicy(Qt.NoFocus)
        self.tc_label.mouseDoubleClickEvent = self._enter_timecode_edit_mode

        info_layout = QHBoxLayout()
        info_layout.addWidget(self.track_btn)
        if self.mute_btn: info_layout.addWidget(self.mute_btn)
        info_layout.addStretch()
        info_layout.addWidget(self.tc_label)
        info_layout.addWidget(self.tc_input)
        info_layout.addWidget(self.tc_btn)
        info_layout.addStretch()
        info_layout.addWidget(self.aspect_btn)
        if self.settings_btn is not None: info_layout.addWidget(self.settings_btn)
        main_layout.addLayout(info_layout)
        main_layout.addWidget(self.slider)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.rew_btn)
        btn_layout.addWidget(self.step_back_btn)
        btn_layout.addWidget(self.play_pause_btn)
        btn_layout.addWidget(self.step_fwd_btn)
        btn_layout.addWidget(self.fwd_btn)
        btn_layout.addWidget(self.live_btn)
        main_layout.addLayout(btn_layout)

        if self.settings_btn is not None: self.settings_btn.clicked.connect(self._show_settings)
        self.aspect_btn.clicked.connect(self._toggle_aspect)
        self.slider.sliderPressed.connect(self._on_slider_pressed)
        self.slider.sliderMoved.connect(self._on_slider_moved)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.play_pause_btn.clicked.connect(self._toggle_play_pause)
        self.live_btn.clicked.connect(self._go_live)
        self.rew_btn.clicked.connect(lambda: self._seek_relative(-5))
        self.fwd_btn.clicked.connect(lambda: self._seek_relative(5))
        self.step_back_btn.clicked.connect(lambda: self._seek_relative(-1))
        self.step_fwd_btn.clicked.connect(lambda: self._seek_relative(1))
        self.tc_btn.clicked.connect(self._toggle_tc_mode)
        if self.mute_btn: self.mute_btn.clicked.connect(self._toggle_mute)
        if self.volume_slider: self.volume_slider.valueChanged.connect(self._on_volume_changed)
        for tid, action in self.track_actions.items():
            action.toggled.connect(lambda checked, tid=tid: self._on_track_toggled(tid, checked))
        self.tc_input.returnPressed.connect(self._on_timecode_entered)

    def _enter_timecode_edit_mode(self, event=None):
        if self._active_player.playing: self._toggle_play_pause()
        self.tc_label.hide(); self.tc_input.show(); self.tc_input.setFocus(); self.tc_input.selectAll()

    def _exit_timecode_edit_mode(self):
        self.tc_input.hide(); self.tc_label.show(); self.setFocus()

    # ------------------------------------------------------------------
    # Единая точка seek
    # ------------------------------------------------------------------
    def _start_seek(self, frame_idx: int):
        """Запускает асинхронный seek с защитой от гонок."""
        self._seek_generation += 1
        gen = self._seek_generation
        self._seeking = True
        self._seek_pending = True   # после перемотки всегда пауза, ждём Play

        def on_seek_done():
            if gen != self._seek_generation:
                return             # устаревший запрос – ничего не делаем
            self._seeking = False
            self._seek_pending = False
            self._update_frame()
            self.video_widget.hide_placeholder()
            self.video_widget.update()

        # Отменяем предыдущий seek-поток (если есть)
        if hasattr(self.player, '_seek_engine'):
            self.player._seek_engine.cancel_current()

        self.player.seek_absolute(frame_idx, callback=on_seek_done)

    # ------------------------------------------------------------------
    # Обработчики кнопок
    # ------------------------------------------------------------------
    def _on_slider_pressed(self):
        self._seeking = True
        self._seek_pending = True
        if self._active_player and self._active_player.playing:
            self._active_player.pause()
        self.play_pause_btn.setText("▶ Play")

    def _on_slider_moved(self, value):
        self._show_tc_for_frame(value)

    def _on_slider_released(self):
        self._start_seek(self.slider.value())

    def _go_live(self):
        if isinstance(self._active_player, StreamController):
            live_frame = max(0, self._active_player.total_frames - 1600)
            self._start_seek(live_frame)
        self.setFocus()

    def _on_timecode_entered(self):
        if not self.player: return
        tc_text = self.tc_input.text().strip()
        if not tc_text or tc_text == "00:00:00;00":
            self._exit_timecode_edit_mode(); return
        try:
            from config.timebase import timecode_to_frame
            target_frame = timecode_to_frame(tc_text, self.player.fps)
            if self.tc_mode == 1:
                target_frame -= self.player.start_frame_offset
            target_frame = max(0, min(target_frame, self.player.total_frames - 1))
            if target_frame < 0 or target_frame >= self.player.total_frames:
                logger.warning(f"Таймкод {tc_text} вне диапазона")
                self._exit_timecode_edit_mode(); return
            self._start_seek(target_frame)
        except ValueError as e:
            logger.warning(f"Ошибка парсинга таймкода '{tc_text}': {e}")
        finally:
            self._exit_timecode_edit_mode()

    def _seek_relative(self, delta_sec: float):
        frame = pts_to_video_frame(self.player.audio_clock)
        target = frame + int(round(delta_sec * self.player.fps))
        target = max(0, min(target, self.player.total_frames - 1))
        self._start_seek(target)

    def _toggle_play_pause(self):
        if self._seeking:
            return   # идёт перемотка, play/pause недоступен
        if self._seek_pending:
            self._seek_pending = False
        self._active_player.toggle_pause()

    def _jkl_seek(self, direction):
        if hasattr(self.player, 'set_seek_speed'):
            self.player.set_seek_speed(direction)
            self.play_pause_btn.setText(self.player.get_seek_speed_display())

    def _jkl_stop(self):
        if hasattr(self.player, 'reset_seek_speed'):
            self.player.reset_seek_speed()
            self.play_pause_btn.setText("⏸ Pause" if self.player.playing else "▶ Play")

    # ------------------------------------------------------------------
    # Остальные методы (аудио, таймер, утилиты)
    # ------------------------------------------------------------------
    def _toggle_mute(self):
        if not self.player or not self.player.master_clock:
            return
        self._muted = not self._muted
        self.player.master_clock.set_muted(self._muted)
        if self.mute_btn:
            self.mute_btn.setText("🔇" if self._muted else "🔊")
        logger.debug("Mute переключён: %s", self._muted)

    def _on_track_toggled(self, track_id: int, checked: bool):
        if self._updating_tracks:
            return
        if track_id in self.active_tracks:
            if not checked:
                self.active_tracks.remove(track_id)
        else:
            if checked:
                self.active_tracks.append(track_id)
        self.player.set_active_tracks(self.active_tracks)

    def _update_render_interval(self):
        fps = self.config.get('fps', 25.0)
        self.render_timer.setInterval(int(1000.0 / fps))

    def _update_frame(self):
        try:
            if self._seeking:
                return
            if self._active_player is None:
                return
            frame = self._active_player.get_display_frame()
            if frame is not None and frame.size > 0:
                self.video_widget.set_frame(frame)
            if isinstance(self._active_player, StreamController):
                audio_clock = self.player.audio_clock
                frame_idx = audio_clock // 1920
                total = self.player.total_frames
                if total > 1:
                    max_slider = total - 1
                    if not self.player._finalized: max_slider = max(0, total - 1600)
                    if self.slider.maximum() != max_slider: self.slider.setRange(0, max_slider)
                if self._active_player.playing:
                    self.slider.blockSignals(True)
                    self.slider.setValue(frame_idx)
                    self.slider.blockSignals(False)
            self._update_tc_label()
            if hasattr(self._active_player, 'get_seek_speed_display'):
                d = self._active_player.get_seek_speed_display()
                self.play_pause_btn.setText(d if "x1" not in d else ("⏸ Pause" if self._active_player.playing else "▶ Play"))
            else:
                self.play_pause_btn.setText("⏸ Pause" if self._active_player.playing else "▶ Play")
        except Exception as e:
            logger.exception("Ошибка в _update_frame, восстановление")

    def _update_tc_label(self):
        if self.tc_mode == 0: tc = self._active_player.get_local_timecode_str()
        else: tc = self._active_player.get_real_timecode_str()
        self.tc_label.setText(tc)
        if not self.tc_input.hasFocus(): self.tc_input.setText(tc)

    def _show_tc_for_frame(self, frame_idx):
        total_seconds = frame_idx / self.player.fps
        h, m = divmod(int(total_seconds), 3600); m, s = divmod(m, 60)
        f = int(round((total_seconds - int(total_seconds)) * self.player.fps))
        if self.tc_mode == 0: self.tc_label.setText(f"{h:02d}:{m:02d}:{s:02d};{f:02d}")
        else:
            abs_frame = self.player.start_frame_offset + frame_idx
            total_seconds_abs = abs_frame / self.player.fps
            h_abs, m_abs = divmod(int(total_seconds_abs), 3600); m_abs, s_abs = divmod(m_abs, 60)
            f_abs = int(round((total_seconds_abs - int(total_seconds_abs)) * self.player.fps))
            self.tc_label.setText(f"{h_abs:02d}:{m_abs:02d}:{s_abs:02d};{f_abs:02d}")

    def _on_volume_changed(self, value):
        logger.debug("Регулировка громкости не реализована в MasterClock (значение=%d)", value)

    def _toggle_tc_mode(self):
        self.tc_mode = 1 - self.tc_mode; self.tc_btn.setText("TC: Лок" if self.tc_mode == 0 else "TC: Реал")

    def _toggle_aspect(self):
        modes = ['fit', '4:3', '16:9']; cur = self.video_widget.aspect_mode
        nxt = modes[(modes.index(cur) + 1) % len(modes)]
        self.video_widget.aspect_mode = nxt; self.aspect_btn.setText(f"📐 {nxt}   "); self.video_widget.update(); self.setFocus()

    def _show_settings(self):
        from config.settings_dialog import SettingsDialog
        dlg = SettingsDialog(self.config, self)
        if dlg.exec_() == QDialog.Accepted:
            self.config = dlg.get_settings(); save_config(self.config); self._reload_file()

    def _reload_file(self):
        self.render_timer.stop()
        if hasattr(self, 'render_timer') and self.render_timer is not None:
            self.render_timer.deleteLater(); self.render_timer = None
        self.player.close()
        self.player = self._create_controller()
        self.player._ready.wait()
        self._active_player = self.player
        self.render_timer = QTimer(self)
        self.render_timer.timeout.connect(self._update_frame)
        self._update_render_interval(); self.render_timer.start()

    def send_hwnd_to_manager(self):
        self._send_hwnd_attempt(0)

    def _send_hwnd_attempt(self, attempt):
        if attempt > 5:
            logger.error("Failed to send HWND after 5 attempts")
            return
        hwnd = int(self.winId())
        if hwnd == 0:
            QTimer.singleShot(200, lambda: self._send_hwnd_attempt(attempt + 1))
            return
        socket = QLocalSocket(self)
        socket.connectToServer("ProxyPlayerManager")
        if socket.waitForConnected(1000):
            socket.write(str(hwnd).encode())
            socket.flush()
            socket.disconnectFromServer()
            logger.debug(f"Sent HWND {hwnd}")
        else:
            logger.warning(f"Connection attempt {attempt} failed, retrying...")
            QTimer.singleShot(500, lambda: self._send_hwnd_attempt(attempt + 1))

    def start_playback(self):
        if self._playback_started:
            return
        self._playback_started = True
        self._seek_pending = False
        try:
            self.player.start_playback()
            if not self.render_timer.isActive():
                self.render_timer.start()
            self.video_widget.hide_placeholder()
        except Exception as e:
            logger.exception("Ошибка в start_playback")
            raise

    def keyPressEvent(self, event):
        if self.tc_input.hasFocus():
            if event.key() == Qt.Key_Escape: self._exit_timecode_edit_mode(); return
            super().keyPressEvent(event); return
        if event.isAutoRepeat(): return
        key = event.key()
        if key == Qt.Key_Space: self._toggle_play_pause()
        elif key == Qt.Key_Left: self._seek_relative(-1)
        elif key == Qt.Key_Right: self._seek_relative(1)
        elif key == Qt.Key_Up: self._seek_relative(10)
        elif key == Qt.Key_Down: self._seek_relative(-10)
        elif key == Qt.Key_J: self._jkl_seek(-1)
        elif key == Qt.Key_L: self._jkl_seek(1)
        elif key == Qt.Key_K: self._jkl_stop()
        else: super().keyPressEvent(event)

    def closeEvent(self, event):
        if hasattr(self, 'render_timer') and self.render_timer is not None:
            self.render_timer.stop(); self.render_timer.deleteLater(); self.render_timer = None
        if hasattr(self, 'video_widget') and self.video_widget is not None:
            if hasattr(self.video_widget, 'cleanup'): self.video_widget.cleanup()
        if self.player: self.player.close()
        QApplication.processEvents()
        super().closeEvent(event)


class ManagedProcess(QObject):
    process_started = pyqtSignal(int)
    def __init__(self, mp4_path: Path, config: dict, use_moov: bool = False, 
                 mirror_path: str = None, parent=None):
        super().__init__(parent)
        self.process = QProcess(self)
        exe = sys.executable
        script = [] if getattr(sys, 'frozen', False) else [os.path.join(os.path.dirname(__file__), 'main.py')]
        args = script + [str(mp4_path), '--managed']
        if use_moov: args.append('--moov')
        if mirror_path:
            args.extend(['--mirror', mirror_path])
        self.process.start(exe, args)
        if self.process.waitForStarted(5000):
            pid = self.process.processId()
            if pid: self.process_started.emit(pid)
        else: logger.error("Cannot start player process")

    def close(self):
        if self.process:
            self.process.terminate()
            if not self.process.waitForFinished(3000):
                self.process.kill()


class ManagerWindow(QMainWindow):
    def __init__(self, mp4_path, config, use_moov=False):
        super().__init__()
        self.config = config
        self.use_moov = use_moov
        self.processes = []; self.hwnd_positions = {}; self._closing = False
        self._index_builder_process = None; self._last_worker = None
        self.setWindowTitle("ProxyPlayer v2 – Panel")
        self.setStyleSheet("background-color: #2b2b2b;")
        icon_path = os.path.join(os.path.dirname(__file__), 'icon.ico')
        app_icon = QIcon(icon_path) if os.path.exists(icon_path) else self.style().standardIcon(QStyle.SP_ComputerIcon)
        self.setWindowIcon(app_icon)
        screen = QDesktopWidget().availableGeometry(self)
        self.setGeometry(0, 0, PANEL_WIDTH, screen.height())
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        hwnd = int(self.winId())
        ex_style = GetWindowLongW(hwnd, GWL_EXSTYLE) | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
        SetWindowLongW(hwnd, GWL_EXSTYLE, ex_style)

        self.tray_icon = QSystemTrayIcon(self); self.tray_icon.setIcon(app_icon); self.tray_icon.setToolTip("ProxyPlayer v2")
        tray_menu = QMenu()
        add_action = QAction("Добавить плеер", self); add_action.triggered.connect(self._open_new_player); tray_menu.addAction(add_action)
        exit_action = QAction("Выход", self); exit_action.triggered.connect(self._stop_all); tray_menu.addAction(exit_action)
        self.tray_icon.setContextMenu(tray_menu); self.tray_icon.activated.connect(self._on_tray_activated); self.tray_icon.show()

        toolbar = QToolBar("Управление", self); toolbar.setOrientation(Qt.Vertical); self.addToolBar(Qt.LeftToolBarArea, toolbar)
        add_btn = QAction("➕", self); add_btn.setToolTip("Добавить плеер"); add_btn.triggered.connect(self._open_new_player); toolbar.addAction(add_btn)
        self.stop_all_btn = QPushButton("з\nа\nв\nе\nр\nш\nи\nт\nь"); self.stop_all_btn.setFixedSize(28, 140)
        self.stop_all_btn.setStyleSheet("QPushButton { background-color: #3a3a3a; color: #ffffff; border: 1px solid #555555; font-size: 11px; padding: 4px 2px; } QPushButton:hover { background-color: #4a4a4a; }")
        self.stop_all_btn.clicked.connect(self._stop_all); toolbar.addWidget(self.stop_all_btn)

        self.server = QLocalServer(self); self.server.newConnection.connect(self._on_new_connection); self.server.listen("ProxyPlayerManager")
        self.http_port = self.config.get('http_port', 18080)
        self.http_server = QTcpServer(self); self.http_server.newConnection.connect(self._on_http_new_connection)
        if self.http_server.listen(port=self.http_port):
            logger.info(f"HTTP-сервер Video Helper запущен на порту {self.http_port}")
        else: logger.error(f"Не удалось запустить HTTP-сервер на порту {self.http_port}")

        if mp4_path and mp4_path.exists() and mp4_path.suffix.lower() == '.mp4':
            try:
                mirror = prepare_mirror(self._find_idx_for(mp4_path))
                self._add_player(mp4_path, use_moov, mirror_path=str(mirror))
            except Exception as e:
                logger.error(f"Не удалось подготовить зеркало: {e}")
        QTimer.singleShot(2000, self._force_place_first_player)
        self.hide()

    def _find_idx_for(self, mp4_path: Path) -> Path:
        idx = mp4_path.parent / "idx" / "mp4" / f"{mp4_path.stem}.idx"
        if not idx.exists(): idx = mp4_path.parent / f"{mp4_path.stem}.idx"
        return idx

    def _start_index_builder(self, mp4_path: Path, use_moov: bool):
        if use_moov: return
        idx_path = self._find_idx_for(mp4_path)
        if not idx_path.exists():
            logger.warning("IDX не найден, IndexBuilder не запущен")
            return
        exe = sys.executable
        if getattr(sys, 'frozen', False):
            args = ['--index-builder', str(idx_path)]
        else:
            script = os.path.join(os.path.dirname(__file__), 'main.py')
            args = [script, '--index-builder', str(idx_path)]
        self._index_builder_process = QProcess(self)
        self._index_builder_process.setProcessChannelMode(QProcess.ForwardedChannels)
        self._index_builder_process.start(exe, args)
        logger.info("IndexBuilder запущен для %s", mp4_path)

    def _force_place_first_player(self):
        if not self.hwnd_positions and self.processes:
            pid = self.processes[0].process.processId()
            if pid:
                import win32gui, win32process
                def callback(hwnd, hwnds):
                    if win32gui.IsWindowVisible(hwnd):
                        _, found_pid = win32process.GetWindowThreadProcessId(hwnd)
                        if found_pid == pid: hwnds.append(hwnd)
                hwnds = []
                win32gui.EnumWindows(callback, hwnds)
                if hwnds: self._place_new_player(hwnds[0])

    def _add_player(self, mp4_path, use_moov, mirror_path=None):
        if len(self.processes) >= MAX_PLAYERS:
            QMessageBox.warning(self, "Ограничение", f"Нельзя открыть больше {MAX_PLAYERS} окон плееров.")
            return
        proc = ManagedProcess(mp4_path, self.config, use_moov, mirror_path=mirror_path, parent=self)
        proc.process.finished.connect(lambda: self._on_player_closed(proc))
        self.processes.append(proc)

    def _on_new_connection(self):
        socket = self.server.nextPendingConnection()
        if socket.waitForReadyRead(1000):
            data = socket.readAll().data().decode().strip()
            if data.startswith("FILE:"):
                self._open_player_from_path(Path(data[5:]))
            elif data.startswith("ID:"):
                mp4_path = self._resolve_id_to_mp4(data[3:])
                if mp4_path: self._open_player_from_path(mp4_path)
                else: logger.error(f"Не удалось найти MP4 для ID {data[3:]}")
            else:
                try:
                    hwnd = int(data)
                    self._place_new_player(hwnd)
                except ValueError: logger.warning(f"Неизвестный формат данных: {data}")
        socket.disconnectFromServer()

    def _on_http_new_connection(self):
        client = self.http_server.nextPendingConnection()
        if client:
            client.readyRead.connect(lambda c=client: self._on_http_ready_read(c))
            client.disconnected.connect(lambda c=client: c.deleteLater())
            if client.bytesAvailable(): self._on_http_ready_read(client)

    def _on_http_ready_read(self, client):
        try:
            data = bytes(client.readAll()).decode('utf-8', errors='ignore')
            request_line = data.split('\r\n')[0]
            if 'GET' in request_line and '/open?file=' in request_line:
                parts = request_line.split('=')
                if len(parts) > 1:
                    file_part = parts[1].split()[0]
                    file_id = file_part.rsplit('.', 1)[0] if '.' in file_part else file_part
                    client.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK")
                    client.disconnectFromHost()
                    worker = OpenPlayerWorker(self, file_id)
                    worker.found_signal.connect(self._on_mp4_found)
                    self._last_worker = worker
                    worker.start()
                    return
            client.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
            client.disconnectFromHost()
        except Exception as e:
            logger.error(f"Ошибка HTTP: {e}")
            client.write(b"HTTP/1.1 500 Internal Server Error\r\n\r\n")
            client.disconnectFromHost()

    def _on_mp4_found(self, mp4_path: Path | None):
        if mp4_path:
            self._open_player_from_path(mp4_path)
        else: logger.error("Файл не найден, плеер не открыт")

    def _resolve_id_to_mp4(self, file_id: str) -> Path | None:
        homedir = self.config.get('homedir', '')
        if not homedir: return None
        home = Path(homedir)
        if not home.is_dir(): return None
        if not re.match(r'^\d+$', file_id): return None
        pattern = re.compile(re.escape(file_id) + r'_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{5}\.mp4$', re.IGNORECASE)
        try:
            for entry in home.glob(f"{file_id}_*.mp4"):
                if entry.is_file() and pattern.match(entry.name):
                    return entry
        except PermissionError: logger.warning(f"Нет доступа к папке {home}")
        return None

    def _open_player_from_path(self, mp4_path: Path):
        if not mp4_path.exists(): return
        idx = self._find_idx_for(mp4_path)
        if not idx.exists():
            QMessageBox.critical(self, "Ошибка", f"Отсутствует индексный файл (.idx) для:\n{mp4_path}")
            return
        try:
            mirror = prepare_mirror(idx)
            self._add_player(mp4_path, use_moov=False, mirror_path=str(mirror))
        except Exception as e:
            logger.error(f"Ошибка подготовки зеркала: {e}")

    def _open_new_player(self):
        start_dir = self.config.get('homedir', '')
        if start_dir and not os.path.isdir(start_dir): start_dir = ''
        path, _ = QFileDialog.getOpenFileName(self, "Open MP4", start_dir, "MP4 files (*.mp4)")
        if path: self._open_player_from_path(Path(path))

    def _place_new_player(self, hwnd):
        screen = QDesktopWidget().availableGeometry(self)
        cell_w = (screen.width() - PANEL_WIDTH) // COLS
        cell_h = screen.height() // ROWS
        for row in range(ROWS):
            for col in range(COLS):
                x = PANEL_WIDTH + col * cell_w
                y = row * cell_h
                if not self._is_cell_occupied(x, y, hwnd):
                    self._set_position(hwnd, x, y, cell_w, cell_h)
                    return
        row = len(self.hwnd_positions) // COLS
        col = len(self.hwnd_positions) % COLS
        x = PANEL_WIDTH + col * cell_w
        y = row * cell_h
        self._set_position(hwnd, x, y, cell_w, cell_h)

    def _is_cell_occupied(self, x, y, hwnd):
        for h, pos in self.hwnd_positions.items():
            if h != hwnd and abs(pos[0] - x) < 10 and abs(pos[1] - y) < 10: return True
        return False

    def _set_position(self, hwnd, x, y, w, h):
        MoveWindow(hwnd, x, y, w, h, True)
        SetWindowPos(hwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOACTIVATE | SWP_SHOWWINDOW)
        self.hwnd_positions[hwnd] = (x, y, w, h)
        QTimer.singleShot(500, lambda: MoveWindow(hwnd, x, y, w, h, True))

    def _on_player_closed(self, proc):
        if proc in self.processes: self.processes.remove(proc)
        self._cleanup_hwnd_positions()

    def _cleanup_hwnd_positions(self):
        import win32gui
        dead = []
        for hwnd in self.hwnd_positions:
            if not win32gui.IsWindow(hwnd): dead.append(hwnd)
        for hwnd in dead: del self.hwnd_positions[hwnd]

    def _on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.DoubleClick or reason == QSystemTrayIcon.Trigger: self._toggle_visible()

    def _toggle_visible(self):
        if self.isVisible(): self.hide()
        else: self._show_from_tray()

    def _show_from_tray(self):
        self.show(); self.raise_(); self.activateWindow()

    def _stop_all(self):
        if self._closing: return
        self._closing = True
        msg = QMessageBox(self)
        msg.setWindowFlags(msg.windowFlags() | Qt.WindowStaysOnTopHint)
        msg.setWindowTitle("Подтверждение")
        msg.setText("Вы точно хотите завершить воспроизведение видео?")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)
        msg.button(QMessageBox.Yes).setText("Да")
        msg.button(QMessageBox.No).setText("Нет")
        msg.setStyleSheet("QMessageBox { background-color: #2b2b2b; color: #ffffff; } QLabel { color: #ffffff; } QPushButton { background-color: #3a3a3a; color: #ffffff; border: 1px solid #555555; padding: 5px 15px; } QPushButton:hover { background-color: #4a4a4a; }")
        screen = QDesktopWidget().availableGeometry(self)
        msg.move(screen.center() - msg.rect().center())
        reply = msg.exec_()
        if reply == QMessageBox.Yes:
            self.server.close()
            if self._index_builder_process:
                self._index_builder_process.kill()
                self._index_builder_process.waitForFinished(1000)
            for proc in self.processes:
                if hasattr(proc, 'close'): proc.close()
            self.tray_icon.hide()
            QApplication.quit()
        else: self._closing = False

    def closeEvent(self, event):
        if self._closing:
            event.accept()
        else:
            self._stop_all()
            self.tray_icon.hide()
            QApplication.quit()
            event.accept()