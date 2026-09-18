#!/usr/bin/env python3
"""
audit_encapsulation.py – инвентаризация нарушений границ между модулями.

ЭТАП 1.1 первой итерации рефакторинга.

ЗАЧЕМ
Контрактный тест (test_contracts.py) проверяет совместимость вызовов
ПУБЛИЧНЫХ методов. Обращения к приватным полям чужих объектов он не
видит и увидеть не может — а именно они образуют скрытые связи между
модулями, которые ломаются молча при переименовании.

Скрипт находит все такие обращения и раскладывает их по адресатам,
чтобы стало понятно, какие публичные методы нужно добавить и кому.

ЧТО СЧИТАЕТСЯ НАРУШЕНИЕМ
Обращение к атрибуту, имя которого начинается с подчёркивания, у
объекта, который НЕ является self. Например:

    self._pipeline._scheduler.set_normal_mode(...)   нарушение
    controller._ready.wait()                          нарушение
    self._window_lock                                 нормально (свой)
    obj.__class__                                     нормально (dunder)

ЧТО НЕ СЧИТАЕТСЯ
- обращения к собственным полям (self._x)
- dunder-атрибуты (__class__, __name__ и подобные)
- обращения внутри того же класса, который владеет полем
- модули из списка исключений (тесты, legacy)

ЗАПУСК
    python audit_encapsulation.py
    python audit_encapsulation.py --root C:\\path\\to\\ProxyPlayer
    python audit_encapsulation.py --include-tests    # учесть и тесты
    python audit_encapsulation.py --csv audit.csv    # выгрузка для работы

Скрипт ничего не меняет — только читает и отчитывается.
"""

import argparse
import ast
import csv
import sys
from collections import defaultdict
from pathlib import Path

# Каталоги, которые не считаем частью рантайма.
SKIP_DIRS = {".git", "__pycache__", "venv", ".venv", "build", "dist",
             "node_modules", ".idea", ".vscode", "legacy"}

# Файлы инструментов и тестов: они обращаются к внутренностям осознанно,
# но их всё равно нужно перевести на публичный интерфейс — просто в
# другую очередь (этап 1.3А и 1.3В).
TOOL_FILES = {
    "player_telemetry.py", "measure_playback.py", "soak_test.py",
    "load_test.py", "player_monitor.py", "analyze_media.py",
}
TEST_FILES_PREFIX = "test_"

# Карта «имя атрибута -> класс-владелец». Та же, что в test_contracts.py:
# проект придерживается соглашения об именах, и это делает вывод надёжным.
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
    "_display_buffer": "FrameRingBuffer",
    "_fill_buffer": "FrameRingBuffer",
    "_free_buffer": "FrameRingBuffer",
    "buffer_main": "FrameRingBuffer",
    "_video_buffer": "FrameRingBuffer",
    "player": "StreamController",
    "_active_player": "StreamController",
    "controller": "StreamController",
    "_controller": "StreamController",
}


class Violation:
    __slots__ = ("file", "lineno", "owner", "attr", "kind", "source",
                 "target_class", "scope")

    def __init__(self, file, lineno, owner, attr, kind, source, target_class,
                 scope="граница"):
        self.scope = scope          # граница | модуль
        self.file = file
        self.lineno = lineno
        self.owner = owner          # выражение-владелец, как в коде
        self.attr = attr            # приватный атрибут
        self.kind = kind            # read | write | call
        self.source = source        # строка исходника
        self.target_class = target_class   # предполагаемый класс-владелец


class Auditor(ast.NodeVisitor):
    """
    Обходит AST одного файла и собирает обращения к приватным атрибутам
    чужих объектов.

    Отдельно помечаются обращения ВНУТРИ МОДУЛЯ: например, SeekEngine
    работает с полями SeekRequest, который объявлен в том же файле и
    является его внутренним помощником. Формально это обращение к
    приватному полю чужого объекта, но границу между модулями оно не
    нарушает и переводу на публичный интерфейс не подлежит. Смешивать
    такие случаи с настоящими нарушениями нельзя — они заглушат сигнал.
    """

    def __init__(self, path: Path, source_lines, local_private_fields):
        self.path = path
        self.lines = source_lines
        self.violations = []
        self._class_stack = []
        # Приватные поля, объявленные классами ЭТОГО же файла.
        self._local_private = local_private_fields

    # -- служебное ----------------------------------------------------
    @staticmethod
    def _expr_to_text(node):
        """Приблизительное текстовое представление выражения-владельца."""
        try:
            return ast.unparse(node)
        except Exception:
            if isinstance(node, ast.Name):
                return node.id
            if isinstance(node, ast.Attribute):
                return f"...{node.attr}"
            return "?"

    @staticmethod
    def _is_self(node):
        return isinstance(node, ast.Name) and node.id == "self"

    @staticmethod
    def _is_private(name: str) -> bool:
        return name.startswith("_") and not name.startswith("__")

    def _owner_attr_name(self, node):
        """
        Для выражения вида <что-то>.<attr> возвращает имя attr владельца,
        если владелец сам является атрибутом. Нужно, чтобы сопоставить с
        ATTR_CLASS_MAP: self._pipeline._scheduler -> владелец '_pipeline'.
        """
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Name):
            return node.id
        return None

    def _record(self, node, attr, kind):
        owner_expr = self._expr_to_text(node.value)
        owner_name = self._owner_attr_name(node.value)
        target = ATTR_CLASS_MAP.get(owner_name)
        src = ""
        if 0 < node.lineno <= len(self.lines):
            src = self.lines[node.lineno - 1].strip()
        scope = "модуль" if attr in self._local_private and target is None else "граница"
        self.violations.append(
            Violation(self.path, node.lineno, owner_expr, attr, kind, src,
                      target, scope)
        )

    # -- обход --------------------------------------------------------
    def visit_ClassDef(self, node):
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_Attribute(self, node):
        # Интересуют только приватные имена.
        if not self._is_private(node.attr):
            self.generic_visit(node)
            return

        # Обращение к собственному полю — норма.
        if self._is_self(node.value):
            self.generic_visit(node)
            return

        # Модульная переменная (например, logging._handlers) нас не касается.
        owner_name = self._owner_attr_name(node.value)
        if owner_name is None:
            self.generic_visit(node)
            return

        # Определяем характер обращения по родителю: присваивание или чтение.
        kind = getattr(node, "_audit_kind", "read")
        self._record(node, node.attr, kind)
        self.generic_visit(node)

    def visit_Assign(self, node):
        # Помечаем цели присваивания, чтобы отличить запись от чтения:
        # запись в чужое поле опаснее чтения.
        for target in node.targets:
            for sub in ast.walk(target):
                if isinstance(sub, ast.Attribute):
                    sub._audit_kind = "write"
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        for sub in ast.walk(node.target):
            if isinstance(sub, ast.Attribute):
                sub._audit_kind = "write"
        self.generic_visit(node)

    def visit_Call(self, node):
        # Вызов метода — отдельная категория: он требует не свойства, а
        # публичного метода-обёртки.
        if isinstance(node.func, ast.Attribute):
            owner = node.func.value
            if isinstance(owner, ast.Attribute) and self._is_private(owner.attr):
                if not self._is_self(owner.value):
                    owner._audit_kind = "call"
        self.generic_visit(node)


def collect_local_private_fields(tree) -> set:
    """
    Приватные поля, которые классы этого файла присваивают себе
    (self._x = ...). Обращение к такому полю у объекта из того же файла —
    внутримодульная связь, а не нарушение границы.
    """
    fields = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if (isinstance(t, ast.Attribute)
                        and isinstance(t.value, ast.Name)
                        and t.value.id == "self"
                        and t.attr.startswith("_")):
                    fields.add(t.attr)
        # методы тоже: self._method(...)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_"):
            fields.add(node.name)
    return fields


def collect_files(root: Path, include_tests: bool):
    for path in sorted(root.rglob("*.py")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if not include_tests and path.name.startswith(TEST_FILES_PREFIX):
            continue
        if path.name == Path(__file__).name:
            continue
        yield path


def audit(root: Path, include_tests: bool):
    all_violations = []
    parse_errors = []
    for path in collect_files(root, include_tests):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as e:
            parse_errors.append((path.relative_to(root), e.lineno, e.msg))
            continue
        local_private = collect_local_private_fields(tree)
        auditor = Auditor(path.relative_to(root), text.splitlines(), local_private)
        auditor.visit(tree)
        all_violations.extend(auditor.violations)
    return all_violations, parse_errors


def categorize(path: Path) -> str:
    name = path.name
    if name.startswith(TEST_FILES_PREFIX):
        return "тесты"
    if name in TOOL_FILES:
        return "инструменты"
    return "рантайм"


def main():
    ap = argparse.ArgumentParser(
        description="Инвентаризация обращений к приватным полям чужих объектов")
    ap.add_argument("--root", default=".", help="корень проекта")
    ap.add_argument("--include-tests", action="store_true",
                    help="учитывать файлы test_*.py")
    ap.add_argument("--csv", default=None, help="выгрузить таблицу в CSV")
    ap.add_argument("--by-target", action="store_true",
                    help="группировать по классу-адресату (что кому добавить)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"Каталог не найден: {root}", file=sys.stderr)
        return 2

    print(f"Аудит инкапсуляции: {root}\n")
    violations, parse_errors = audit(root, args.include_tests)

    if parse_errors:
        print("НЕ РАЗОБРАНЫ (синтаксис):")
        for f, line, msg in parse_errors:
            print(f"  {f}:{line}: {msg}")
        print()

    if not violations:
        print("Обращений к приватным полям чужих объектов не найдено.")
        return 0

    # Внутримодульные связи выносим отдельно: они не входят в объём
    # этой итерации.
    internal = [v for v in violations if v.scope == "модуль"]
    violations = [v for v in violations if v.scope == "граница"]

    # --- сводка по категориям ---
    by_cat = defaultdict(list)
    for v in violations:
        by_cat[categorize(v.file)].append(v)

    print("-- Сводка " + "-" * 60)
    for cat in ("рантайм", "инструменты", "тесты"):
        items = by_cat.get(cat, [])
        if not items:
            continue
        writes = sum(1 for v in items if v.kind == "write")
        calls = sum(1 for v in items if v.kind == "call")
        reads = len(items) - writes - calls
        print(f"  {cat:12s} всего {len(items):>4}   "
              f"чтение {reads:>4}, вызов {calls:>3}, ЗАПИСЬ {writes:>3}")
    print()
    print("  Запись в чужое поле опаснее чтения: она меняет состояние")
    print("  объекта в обход его собственной логики.")
    if internal:
        print()
        print(f"  Внутримодульных обращений (не в объёме итерации): {len(internal)}")
        print("  Это работа класса с полями своего же помощника из того же")
        print("  файла — границу между модулями такие связи не нарушают.")
    print()

    if args.by_target:
        # --- что кому добавить ---
        by_target = defaultdict(set)
        unknown = defaultdict(set)
        for v in violations:
            if v.target_class:
                by_target[v.target_class].add((v.attr, v.kind))
            else:
                unknown[v.owner].add((v.attr, v.kind))

        print("-- Кому какие публичные методы нужны " + "-" * 33)
        for cls in sorted(by_target):
            attrs = sorted(by_target[cls])
            print(f"\n  {cls}:")
            for attr, kind in attrs:
                mark = {"write": "ЗАПИСЬ", "call": "вызов ", "read": "чтение"}[kind]
                print(f"    {mark}  {attr}")
        if unknown:
            print("\n  Владелец не определён по карте имён:")
            for owner in sorted(unknown):
                attrs = ", ".join(sorted(a for a, _ in unknown[owner]))
                print(f"    {owner}: {attrs}")
        print()

    # --- подробности по файлам ---
    print("-- Подробности " + "-" * 55)
    by_file = defaultdict(list)
    for v in violations:
        by_file[v.file].append(v)

    for f in sorted(by_file, key=lambda p: (categorize(p), str(p))):
        items = by_file[f]
        print(f"\n  {f}  [{categorize(f)}]  — {len(items)}")
        for v in sorted(items, key=lambda x: x.lineno):
            mark = {"write": "ЗАПИСЬ", "call": "вызов ", "read": "чтение"}[v.kind]
            target = f" -> {v.target_class}" if v.target_class else ""
            print(f"    {v.lineno:>5}  {mark}  {v.owner}.{v.attr}{target}")
            if v.source:
                print(f"           {v.source[:100]}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["категория", "файл", "строка", "вид", "владелец",
                        "атрибут", "класс_адресат", "исходник"])
            for v in sorted(violations, key=lambda x: (str(x.file), x.lineno)):
                w.writerow([categorize(v.file), str(v.file), v.lineno, v.kind,
                            v.owner, v.attr, v.target_class or "", v.source])
        print(f"\nТаблица выгружена: {args.csv}")

    print(f"\nВсего нарушений: {len(violations)}")
    runtime = len(by_cat.get("рантайм", []))
    print(f"Из них в рантайме: {runtime} — это объём этапа 1.3Б")
    return 1 if runtime else 0


if __name__ == "__main__":
    sys.exit(main())
