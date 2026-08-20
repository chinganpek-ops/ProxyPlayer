#!/usr/bin/env python3
"""
test_seek_pool_extended.py – расширенный стресс-тест пула воркеров (v3).

Адаптирован под новый SeekEngine (пул с координатором):
- Конструктор SeekEngine(lazy_index, decoder) — без mp4_path и mdat_end.
- Запуск seek через seek_async(frame_idx, on_complete, on_error).
- Мониторинг состояний воркеров: free/busy/stuck.
- Сбор метрик: успешные/ошибочные/отменённые, блокировки, время.

Запуск:
    python test_seek_pool_extended.py <mp4_path> <idx_path> [mdat_end]
"""

import sys
import time
import threading
import logging
from pathlib import Path
from collections import defaultdict

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(threadName)-20s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("SeekStressTest")

# Импорты наших модулей
from index.lazy_index import LazyIndex
from index.idx_cache import get_mirror_path
from decode.decoder import Decoder
from seek.seek_engine import SeekEngine
from utils.utils import get_real_size


class SeekPoolMonitor:
    """Мониторинг состояния пула воркеров и сбор метрик."""

    def __init__(self, pool: SeekEngine):
        self.pool = pool
        self._stop_event = threading.Event()
        self._monitor_thread = None
        self.state_history = defaultdict(list)
        self.lock = threading.Lock()

    def start(self):
        """Запускает фоновый мониторинг."""
        self._monitor_thread = threading.Thread(
            target=self._run, daemon=True, name="PoolMonitor"
        )
        self._monitor_thread.start()

    def stop(self):
        """Останавливает мониторинг."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

    def _run(self):
        """Логирует состояние воркеров каждые 100 мс."""
        while not self._stop_event.is_set():
            timestamp = time.time()
            snapshot = []
            for worker in self.pool.workers:
                state_info = {
                    'id': worker.id,
                    'state': worker.state,
                    'generation': 0,  # в новом пуле нет прямого generation у воркера
                    'buffer_count': worker.buffer.count,
                    'thread_alive': worker.thread.is_alive() if worker.thread else False,
                }
                snapshot.append(state_info)
                with self.lock:
                    self.state_history[worker.id].append((timestamp, worker.state))

            states = " | ".join(
                f"W{w['id']}:{w['state'][:4].ljust(4)} "
                f"buf={w['buffer_count']} alive={int(w['thread_alive'])}"
                for w in snapshot
            )
            logger.debug(f"[STATES] {states}")

            self._stop_event.wait(0.1)

    def get_summary(self):
        """Возвращает сводку по состояниям воркеров."""
        summary = {}
        with self.lock:
            for worker_id, history in self.state_history.items():
                states = [s for _, s in history]
                total = len(states)
                if total == 0:
                    continue
                state_counts = defaultdict(int)
                for s in states:
                    state_counts[s] += 1
                summary[worker_id] = {
                    'total_samples': total,
                    'free': state_counts.get('free', 0),
                    'busy': state_counts.get('busy', 0),
                    'stuck': state_counts.get('stuck', 0),
                }
        return summary


class SeekStressTester:
    """Стресс-тест с эмуляцией быстрых перемоток."""

    def __init__(self, mp4_path: Path, mirror_path: Path, mdat_end: int):
        self.mp4_path = mp4_path
        self.mirror_path = mirror_path
        self.mdat_end = mdat_end

        logger.info("Инициализация LazyIndex...")
        self.lazy_index = LazyIndex(mirror_path, mp4_path, mdat_end)

        logger.info("Инициализация Decoder...")
        self.decoder = Decoder(
            bytes.fromhex("014d001fffe1002e674d401f9652816824dff80200016a50101014000003000400000300cb8180009600000301e848fc6383b428532c01000568e9093520"),
            mp4_path
        )

        logger.info("Создание пула воркеров (3 шт.)...")
        self.pool = SeekEngine(self.lazy_index, self.decoder)

        # Метрики теста
        self.metrics = {
            'total_seeks': 0,
            'successful_seeks': 0,
            'failed_seeks': 0,
            'cancelled_seeks': 0,
            'blocking_events': 0,
            'total_time': 0.0,
            'seek_times': [],
        }

        self._start_time = 0.0
        self._lock = threading.Lock()

    def _on_seek_complete(self, buffer):
        """Колбэк при успешном завершении seek."""
        with self._lock:
            elapsed = time.time() - self._start_time
            self.metrics['successful_seeks'] += 1
            self.metrics['seek_times'].append(elapsed)
            logger.info(
                f"[SEEK COMPLETE] buffer_count={buffer.count}, elapsed={elapsed:.3f}s"
            )

    def _on_seek_error(self, error_msg):
        """Колбэк при ошибке seek."""
        with self._lock:
            self.metrics['failed_seeks'] += 1
            logger.error(f"[SEEK ERROR] {error_msg}")

    def run_stress_test(self, target_frames, delay_between=0.2):
        """Запускает серию быстрых seek."""
        logger.info(f"Запуск стресс-теста: {len(target_frames)} перемоток, "
                     f"пауза {delay_between} сек")

        # Запускаем мониторинг
        monitor = SeekPoolMonitor(self.pool)
        monitor.start()

        self._start_time = time.time()

        for i, frame_idx in enumerate(target_frames):
            with self._lock:
                self.metrics['total_seeks'] += 1

            logger.info(f"\n{'='*60}")
            logger.info(f"Запрос {i+1}: кадр {frame_idx}")

            # Проверяем, есть ли свободные воркеры
            free_workers = [w for w in self.pool.workers if w.state == 'free']
            busy_workers = [w for w in self.pool.workers if w.state == 'busy']
            if not free_workers and len(busy_workers) == len(self.pool.workers):
                with self._lock:
                    self.metrics['blocking_events'] += 1
                logger.warning("ВСЕ ВОРКЕРЫ ЗАНЯТЫ - будет отменён самый старый")

            # Запускаем seek
            self.pool.seek_async(
                frame_idx,
                on_complete=self._on_seek_complete,
                on_error=self._on_seek_error,
            )

            time.sleep(delay_between)

        # Ждём завершения всех активных операций
        self._wait_for_idle(timeout=10.0)

        # Останавливаем мониторинг
        monitor.stop()

        # Собираем статистику воркеров
        worker_summary = monitor.get_summary()

        # Итоговое время
        self.metrics['total_time'] = time.time() - self._start_time

        # Выводим сводку
        self._print_summary(worker_summary)

    def _wait_for_idle(self, timeout: float):
        """Ожидает, пока все воркеры не освободятся."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            active = [w for w in self.pool.workers if w.state == 'busy']
            if not active:
                return
            time.sleep(0.1)
        logger.warning("Таймаут ожидания завершения воркеров")

    def _print_summary(self, worker_summary):
        """Выводит итоговую сводку."""
        print("\n" + "=" * 70)
        print("ИТОГОВАЯ СВОДКА СТРЕСС-ТЕСТА")
        print("=" * 70)

        m = self.metrics
        print(f"\n--- Метрики Seek ---")
        print(f"Всего запросов:         {m['total_seeks']}")
        print(f"Успешно завершено:      {m['successful_seeks']}")
        print(f"Ошибок:                 {m['failed_seeks']}")
        print(f"Отменено (устарело):    {m['total_seeks'] - m['successful_seeks'] - m['failed_seeks']}")
        print(f"Блокировок (все заняты): {m['blocking_events']}")
        print(f"Общее время:            {m['total_time']:.2f} сек")

        if m['seek_times']:
            avg_time = sum(m['seek_times']) / len(m['seek_times'])
            min_time = min(m['seek_times'])
            max_time = max(m['seek_times'])
            print(f"Среднее время seek:     {avg_time*1000:.1f} мс")
            print(f"Мин/макс время seek:    {min_time*1000:.1f} / {max_time*1000:.1f} мс")

        print(f"\n--- Состояния воркеров ---")
        for worker_id, stats in sorted(worker_summary.items()):
            total = stats['total_samples']
            print(f"Воркер {worker_id}:")
            print(f"  Всего выборок: {total}")
            for state in ['free', 'busy', 'stuck']:
                count = stats.get(state, 0)
                pct = (count / total * 100) if total > 0 else 0
                print(f"  {state:10s}: {count:4d} ({pct:.1f}%)")

        print(f"\n--- Текущее состояние ---")
        print(f"Воркеров в пуле: {len(self.pool.workers)}")
        active = [w for w in self.pool.workers if w.state == 'busy']
        print(f"Активных воркеров: {len(active)}")

        print("=" * 70)


def main():
    if len(sys.argv) < 3:
        print("Usage: python test_seek_pool_extended.py <mp4_path> <idx_path> [mdat_end]")
        sys.exit(1)

    mp4_path = Path(sys.argv[1])
    idx_path = Path(sys.argv[2])

    mdat_end = int(sys.argv[3]) if len(sys.argv) > 3 else get_real_size(str(mp4_path))
    if mdat_end == 0:
        print("Не удалось определить размер MP4")
        sys.exit(1)

    mirror_path = get_mirror_path(idx_path)
    if not mirror_path.exists():
        mirror_path = idx_path

    logger.info(f"MP4: {mp4_path}")
    logger.info(f"IDX mirror: {mirror_path}")
    logger.info(f"mdat_end: {mdat_end}")

    tester = SeekStressTester(mp4_path, mirror_path, mdat_end)

    total_frames = tester.lazy_index.total_frames
    if total_frames == 0:
        print("Индекс пуст")
        sys.exit(1)

    # Генерируем последовательность перемоток с повтором 5 раз
    seek_sequence = []
    for _ in range(5):
        live_frame = max(0, total_frames - 1600)
        seek_sequence.append(0)          # в начало
        seek_sequence.append(live_frame) # в live

        current = live_frame
        for minutes in [1, 2, 3, 4, 5]:
            target = max(0, current - minutes * 60 * 25)
            seek_sequence.append(target)
            current = target

    logger.info(f"Сгенерировано {len(seek_sequence)} перемоток")

    logger.info(f"Последовательность кадров: {seek_sequence}")

    tester.run_stress_test(seek_sequence, delay_between=0.2)

    # Останавливаем пул
    tester.pool.close()


if __name__ == "__main__":
    main()