# -*- mode: python ; coding: utf-8 -*-
"""
ProxyPlayer.spec — сборка ProxyPlayer через PyInstaller.

СБОРКА
    pyinstaller ProxyPlayer.spec --noconfirm
    (результат: dist\\ProxyPlayer\\ProxyPlayer.exe)

ПОЧЕМУ ONEDIR, А НЕ ONEFILE
Это принципиально для данного проекта, а не вопрос вкуса.

ProxyPlayer запускает САМ СЕБЯ как дочерние процессы:
  - main.py <mp4> --managed          — окно плеера (до 6 штук)
  - main.py --index-service <idx>    — сервис индексирования
Оба спавнятся через sys.executable (см. ManagedProcess и
ManagerWindow._ensure_index_service).

В режиме onefile каждый запуск exe распаковывает ВЕСЬ архив во временный
каталог _MEIxxxxx. То есть при сетке из шести окон плюс сервис вы получите
семь независимых распаковок (PyQt5 + PyAV + numpy — сотни мегабайт каждая),
семикратный расход диска и заметную задержку старта каждого окна.
В onedir распаковки нет вообще: дочерние процессы стартуют мгновенно и
делят одни и те же файлы.

Дополнительно: faulthandler и аварийные дампы в onefile указывают на пути
внутри временного каталога, который удаляется при выходе, — отлаживать
падения становится заметно труднее.
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

# Каталог проекта. spec-файл лежит в корне, рядом с main.py.
try:
    PROJECT_ROOT = Path(SPECPATH)          # PyInstaller определяет SPECPATH сам
except NameError:
    PROJECT_ROOT = Path(".").resolve()

# ---------------------------------------------------------------------------
# Внешние зависимости, которые PyInstaller не находит автоматически
# ---------------------------------------------------------------------------

# PyAV: нативные библиотеки FFmpeg лежат рядом с пакетом (av.libs) и без
# явного сбора в дистрибутив не попадают — декодирование упадёт на импорте.
av_datas, av_binaries, av_hidden = collect_all("av")

# sounddevice: сам модуль PyInstaller находит (он на cffi), но нативная
# библиотека PortAudio лежит НЕ внутри пакета sounddevice, а в отдельном
# каталоге верхнего уровня _sounddevice_data\portaudio-binaries\.
# collect_all("sounddevice") её не забирает — это другой пакет. Без DLL
# импорт проходит, а MasterClock.start() падает с
# "OSError: PortAudio library not found" уже в рантайме.
sd_datas, sd_binaries, sd_hidden = collect_all("sounddevice")

# Каталог с самой DLL. Собираем и как данные, и как бинарники: в разных
# версиях wheel библиотека попадает то в одну категорию, то в другую.
try:
    pa_datas, pa_binaries, pa_hidden = collect_all("_sounddevice_data")
    sd_datas += pa_datas
    sd_binaries += pa_binaries
    sd_hidden += pa_hidden
except Exception:
    # PortAudio может быть установлен в системе, а не в пакете —
    # тогда каталога нет, и это не ошибка.
    pass

# Подстраховка: явный поиск DLL на случай, если collect_all её не увидит.
try:
    import _sounddevice_data
    _pa_dir = Path(_sounddevice_data.__file__).parent / "portaudio-binaries"
    if _pa_dir.is_dir():
        for _dll in _pa_dir.glob("*.dll"):
            sd_binaries.append((str(_dll), "_sounddevice_data/portaudio-binaries"))
except Exception:
    pass

# numpy иногда недосчитывает подмодули при агрессивной оптимизации.
numpy_hidden = collect_submodules("numpy")

hiddenimports = [
    # --- Qt: QtNetwork используется для IPC (QLocalServer/QLocalSocket)
    # между менеджером, окнами плеера и IndexService, а также для
    # HTTP-сервера Video Helper (QTcpServer). Анализатор его часто
    # пропускает, потому что импорт идёт внутри методов.
    "PyQt5.QtCore",
    "PyQt5.QtGui",
    "PyQt5.QtWidgets",
    "PyQt5.QtNetwork",
    "PyQt5.QtOpenGL",          # GLVideoWidget наследует QOpenGLWidget

    # --- pywin32: размещение окон в сетке (ManagerWindow._place_new_player)
    "win32gui",
    "win32process",
    "win32api",
    "win32con",
    "pywintypes",

    # --- Модули проекта, импортируемые ЛЕНИВО (внутри функций/методов).
    # Статический анализ их не видит, и в сборке они отсутствуют — ошибка
    # проявится только при первом обращении к соответствующему режиму.
    "config.settings_dialog",   # открывается по кнопке настроек
    "index.index_service",      # режим --index-service
    "index_service",            # он же при плоской раскладке
    "seek_monitor",             # install_seek_monitoring в main.py
    "player_telemetry",         # телеметрия
    "utils.utils",
    "utils",
    "ui.controls",

    # --- cffi: механизм, через который sounddevice обращается к PortAudio.
    # _cffi_backend — скомпилированное расширение, статически не видное.
    "cffi",
    "_cffi_backend",

    # --- Прочее
    "faulthandler",
    "psutil",                   # используется телеметрией, если установлен
]
hiddenimports += av_hidden + sd_hidden + numpy_hidden

binaries = av_binaries + sd_binaries

# ---------------------------------------------------------------------------
# Файлы данных
# ---------------------------------------------------------------------------
datas = av_datas + sd_datas

# Иконка приложения и трея (ManagerWindow ищет icon.ico рядом с модулем).
_icon = PROJECT_ROOT / "icon.ico"
if _icon.exists():
    datas.append((str(_icon), "."))

# ---------------------------------------------------------------------------
# Исключения: не тащим то, что заведомо не используется
# ---------------------------------------------------------------------------
excludes = [
    "tkinter",
    "matplotlib",
    "PIL",
    "pandas",
    "scipy",
    "IPython",
    "jupyter",
    "pytest",
    "PyQt5.QtWebEngineWidgets",
    "PyQt5.QtQml",
    "PyQt5.QtQuick",
    "PyQt5.Qt3DCore",
    "PyQt5.QtBluetooth",
    "PyQt5.QtDesigner",
    # ВНИМАНИЕ: не исключайте PyQt5.QtNetwork — на нём держится весь IPC.
]

a = Analysis(
    ["main.py"],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,          # onedir: бинарники рядом, не внутри exe
    name="ProxyPlayer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                      # UPX ломает подпись и мешает антивирусам;
                                    # на сетевых шарах это лишний риск
    console=False,                  # GUI-приложение, консоль не нужна
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(_icon) if _icon.exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="ProxyPlayer",
)
