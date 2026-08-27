#!/usr/bin/env python3
"""
test_contracts.py – проверка контрактов между модулями ProxyPlayer.

ЗАЧЕМ ЭТОТ ТЕСТ СУЩЕСТВУЕТ
За время отладки повторился один и тот же класс ошибок: интерфейс модуля
менялся, а вызывающий код — нет. Примеры из реальной практики проекта:

  * PlaybackEngine.seek(frame_idx, window, on_complete) вызывался из
    StreamController как seek(frame_idx, on_complete=...) — TypeError,
    проглоченный except'ом, из-за чего перемотка молча не работала;
  * WinSequentialReader лишился read_sequential()/set_cancel_event(),
    а chunk_pipeline и seek_engine продолжали их звать;
  * index_builder заменён на index_service, но ссылки остались.

Каждая такая ошибка проявлялась только в рантайме, часто — в отдельном
потоке, где исключение попадало в лог и терялось. Этот тест находит их
СТАТИЧЕСКИ, за секунды, без запуска плеера, без PyQt5/av/sounddevice.

ЧТО ПРОВЕРЯЕТСЯ
1. Синтаксис всех .py файлов проекта.
2. Существование методов, вызываемых через известные атрибуты
   (self._playback.X, self._pipeline.X, self._lazy_index.X, ...).
3. Совместимость аргументов: число позиционных, имена ключевых,
   обязательные без значения по умолчанию.
4. Импорты несуществующих модулей проекта (ловит остатки удалённых
   модулей вроде index_builder).

ЧЕГО ТЕСТ НЕ ДЕЛАЕТ
Это статический анализ: он не запускает код и не проверяет логику. Он
отвечает ровно на один вопрос — «совпадают ли вызовы с объявлениями».
Динамические вызовы (getattr, **kwargs-прокси) намеренно пропускаются,
чтобы не давать ложных срабатываний.

ЗАПУСК
    python test_contracts.py                 # из корня проекта
    python test_contracts.py --root C:\\path\\to\\ProxyPlayer
    python test_contracts.py --verbose       # показать, что проверено

Код возврата: 0 — контракты согласованы, 1 — найдены несоответствия
(подходит для CI / pre-commit).
"""

import argparse
import ast
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Карта: имя атрибута -> класс, на который он ссылается.
# Это единственное место, которое нужно дополнять при появлении новых
# компонентов. Namespace-соглашение проекта делает такую карту надёжной:
# self._playback — всегда PlaybackEngine и ничто иное.
# ---------------------------------------------------------------------------
ATTR_CLASS_MAP = {
    "_playback": "PlaybackEngine",
    "_pipeline": "ChunkPipeline",
    "_lazy_index": "LazyIndex",
    "lazy_index": "LazyIndex",
    "_seek_engine": "SeekEngine",
    "_scheduler": "StreamScheduler",
    "_reader": "WinSequentialReader",
    "_decoder": "Decoder",
    "_sync": "SyncManager",
    "_sync_mgr": "SyncManager",
    "master_clock": "MasterClock",
    "_master_clock": "MasterClock",
    "_buffer": "FrameRingBuffer",
    "_display_buffer": "FrameRingBuffer",
    "_fill_buffer": "FrameRingBuffer",
    "_free_buffer": "FrameRingBuffer",
    "buffer_main": "FrameRingBuffer",
    "_video_buffer": "FrameRingBuffer",
    "player": "StreamController",
    "_active_player": "StreamController",
}

# Классы, для которых проверка отключена: слишком динамичны либо являются
# обёртками с **kwargs, где статический вывод даёт ложные срабатывания.
SKIP_CLASSES = set()

# Каталоги, которые не считаем частью проекта.
SKIP_DIRS = {".git", "__pycache__", "venv", ".venv", "build", "dist",
             "node_modules", ".idea", ".vscode"}


class MethodSignature:
    """Сигнатура метода, извлечённая из AST."""

    __slots__ = ("name", "cls", "pos_args", "defaults_count", "kwonly",
                 "kwonly_required", "has_varargs", "has_kwargs", "lineno", "file")

    def __init__(self, name, cls, node, file):
        self.name = name
        self.cls = cls
        self.file = file
        self.lineno = node.lineno
        a = node.args
        # self отбрасываем — интересует то, что передаёт вызывающий.
        self.pos_args = [x.arg for x in (a.posonlyargs + a.args)][1:]
        self.defaults_count = len(a.defaults)
        self.kwonly = [x.arg for x in a.kwonlyargs]
        self.kwonly_required = [x.arg for x, d in zip(a.kwonlyargs, a.kw_defaults)
                                if d is None]
        self.has_varargs = a.vararg is not None
        self.has_kwargs = a.kwarg is not None

    @property
    def min_pos(self):
        """Минимум позиционных аргументов (без значений по умолчанию)."""
        return max(0, len(self.pos_args) - self.defaults_count)

    def describe(self):
        parts = list(self.pos_args)
        if self.has_varargs:
            parts.append("*args")
        parts.extend(self.kwonly)
        if self.has_kwargs:
            parts.append("**kwargs")
        return f"{self.cls}.{self.name}({', '.join(parts)})"


class CallSite:
    __slots__ = ("cls", "method", "n_pos", "keywords", "has_star", "file", "lineno")

    def __init__(self, cls, method, n_pos, keywords, has_star, file, lineno):
        self.cls = cls
        self.method = method
        self.n_pos = n_pos
        self.keywords = keywords
        self.has_star = has_star
        self.file = file
        self.lineno = lineno


def collect_python_files(root: Path):
    for path in sorted(root.rglob("*.py")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def parse_project(root: Path):
    """Возвращает (классы, вызовы, модули, синтаксические ошибки)."""
    classes = defaultdict(dict)      # class -> {method -> MethodSignature}
    calls = []
    modules = set()
    syntax_errors = []
    imports = []                     # (module_name, file, lineno)

    for path in collect_python_files(root):
        rel = path.relative_to(root)
        modules.add(path.stem)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"),
                             filename=str(rel))
        except SyntaxError as e:
            syntax_errors.append((rel, e.lineno, e.msg))
            continue

        # --- объявления классов и методов ---
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        classes[node.name][item.name] = MethodSignature(
                            item.name, node.name, item, rel)

        # --- вызовы вида <что-то>.<attr>.<method>(...) ---
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            owner = func.value
            if not isinstance(owner, ast.Attribute):
                continue
            attr_name = owner.attr
            cls = ATTR_CLASS_MAP.get(attr_name)
            if cls is None or cls in SKIP_CLASSES:
                continue

            n_pos = len(node.args)
            has_star = any(isinstance(a, ast.Starred) for a in node.args)
            kws = [k.arg for k in node.keywords]
            has_star = has_star or any(k.arg is None for k in node.keywords)
            calls.append(CallSite(cls, func.attr, n_pos,
                                  [k for k in kws if k], has_star,
                                  rel, node.lineno))

        # --- импорты модулей проекта ---
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imports.append((node.module, rel, node.lineno))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append((alias.name, rel, node.lineno))

    return classes, calls, modules, syntax_errors, imports


def check_calls(classes, calls, verbose=False):
    problems = []
    checked = 0

    for call in calls:
        methods = classes.get(call.cls)
        if not methods:
            continue          # класс не найден в проекте — молчим, не гадаем
        sig = methods.get(call.method)
        if sig is None:
            # Метод может быть унаследован — проверяем осторожно: сообщаем,
            # только если ни один класс проекта его не объявляет.
            declared_anywhere = any(call.method in m for m in classes.values())
            if not declared_anywhere:
                problems.append(
                    f"{call.file}:{call.lineno}: {call.cls}.{call.method}() — "
                    f"метод не найден ни в одном классе проекта")
            continue

        checked += 1
        if call.has_star:
            continue          # *args/**kwargs — статически не проверить

        # Слишком много позиционных
        if not sig.has_varargs and call.n_pos > len(sig.pos_args):
            problems.append(
                f"{call.file}:{call.lineno}: {call.cls}.{call.method}() — "
                f"передано {call.n_pos} позиционных, принимает "
                f"{len(sig.pos_args)}   [{sig.describe()}]")
            continue

        # Неизвестные ключевые
        if not sig.has_kwargs:
            valid = set(sig.pos_args) | set(sig.kwonly)
            for kw in call.keywords:
                if kw not in valid:
                    problems.append(
                        f"{call.file}:{call.lineno}: {call.cls}.{call.method}() — "
                        f"неизвестный аргумент '{kw}'   [{sig.describe()}]")

        # Недостаёт обязательных
        supplied = set(sig.pos_args[:call.n_pos]) | set(call.keywords)
        missing = [a for a in sig.pos_args[:sig.min_pos] if a not in supplied]
        missing += [a for a in sig.kwonly_required if a not in supplied]
        if missing:
            problems.append(
                f"{call.file}:{call.lineno}: {call.cls}.{call.method}() — "
                f"не передан обязательный аргумент: {', '.join(missing)}   "
                f"[{sig.describe()}]")

    return problems, checked


# Внешние зависимости проекта: их отсутствие в каталоге — норма.
EXTERNAL_PACKAGES = {
    "PyQt5", "numpy", "np", "av", "sounddevice", "psutil", "win32gui",
    "win32process", "win32api", "win32con", "pywintypes", "cv2", "scipy",
    "pytest", "setuptools", "pkg_resources", "dateutil", "PIL",
}


def check_imports(modules, imports, root: Path):
    """
    Ищет импорты модулей, которых нет ни в проекте, ни в стандартной
    библиотеке, ни в списке внешних зависимостей.

    Именно так ловятся остатки удалённых модулей (реальный случай:
    index_builder.py заменили на index_service.py, а импорт остался и падал
    только при запуске соответствующего режима).

    Проверка намеренно консервативна: всё, что удалось отнести к stdlib или
    к известным внешним пакетам, пропускается без анализа — задача теста
    находить рассинхрон внутри проекта, а не ревизовать окружение.
    """
    problems = []
    stdlib = getattr(sys, "stdlib_module_names", None)
    if stdlib is None:      # Python < 3.10
        stdlib = set(sys.builtin_module_names)

    for mod, file, lineno in imports:
        head = mod.split(".")[0]
        if head in stdlib or head in EXTERNAL_PACKAGES:
            continue

        # Разрешается ли импорт файлом/пакетом внутри проекта?
        rel = Path(mod.replace(".", "/"))
        if (root / rel).with_suffix(".py").exists():
            continue
        if (root / rel / "__init__.py").exists():
            continue
        # Модуль верхнего уровня рядом с main.py (плоская раскладка).
        if head in modules:
            continue

        problems.append(
            f"{file}:{lineno}: импорт '{mod}' — модуль не найден ни в проекте, "
            f"ни в stdlib, ни в списке внешних зависимостей")
    return problems


def main():
    ap = argparse.ArgumentParser(description="Проверка контрактов ProxyPlayer")
    ap.add_argument("--root", default=".", help="корень проекта")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"Каталог не найден: {root}", file=sys.stderr)
        return 2

    print(f"Проверка проекта: {root}\n")
    classes, calls, modules, syntax_errors, imports = parse_project(root)

    failed = False

    # --- 1. Синтаксис ---
    print(f"[1] Синтаксис: разобрано модулей — {len(modules)}")
    if syntax_errors:
        failed = True
        for f, line, msg in syntax_errors:
            print(f"    ОШИБКА {f}:{line}: {msg}")
    else:
        print("    OK")

    # --- 2. Контракты вызовов ---
    problems, checked = check_calls(classes, calls, args.verbose)
    print(f"\n[2] Контракты вызовов: проверено {checked} вызовов "
          f"по {len(ATTR_CLASS_MAP)} известным атрибутам")
    if problems:
        failed = True
        for p in problems:
            print(f"    ОШИБКА {p}")
    else:
        print("    OK")

    # --- 3. Импорты ---
    imp_problems = check_imports(modules, imports, root)
    print(f"\n[3] Импорты модулей проекта: проверено {len(imports)}")
    if imp_problems:
        failed = True
        for p in imp_problems:
            print(f"    ОШИБКА {p}")
    else:
        print("    OK")

    if args.verbose:
        print(f"\nНайдено классов: {len(classes)}")
        for cls in sorted(classes):
            if cls in set(ATTR_CLASS_MAP.values()):
                print(f"  {cls}: {len(classes[cls])} методов")

    print("\n" + "=" * 60)
    if failed:
        print("РЕЗУЛЬТАТ: найдены несоответствия (см. выше)")
        return 1
    print("РЕЗУЛЬТАТ: контракты согласованы")
    return 0


if __name__ == "__main__":
    sys.exit(main())
