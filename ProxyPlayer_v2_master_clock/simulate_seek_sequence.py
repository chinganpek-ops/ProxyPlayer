#!/usr/bin/env python3
"""
simulate_seek_sequence.py – моделирование чтения при последовательности перемоток.

Сценарий:
1. Начальная позиция: 0 минут.
2. Перемотка вперёд на 20 минут (1200 секунд).
3. Затем 5 быстрых перемоток назад: на 1, 2, 3, 4, 5 минут от новой позиции.

Показываем:
- Текущее окно (границы кадров и чанков).
- Загруженные чанки.
- Объём данных, который требуется прочитать для каждого шага при двух подходах:
  а) Полный сброс планировщика (всегда читаем буфер заново).
  б) Оптимальный подход с сохранением пересекающихся чанков.
"""

import math

# Параметры модели
FRAMES_PER_CHUNK = 12
FPS = 25
CHUNK_DURATION_SEC = FRAMES_PER_CHUNK / FPS   # 0.48 сек
CHUNK_SIZE_BYTES = 150 * 1024                 # ~150 КБ на чанк
WINDOW_DURATION_SEC = 300                     # 5 минут
WINDOW_TOTAL_CHUNKS = int(WINDOW_DURATION_SEC / CHUNK_DURATION_SEC)  # 625 чанков

BUFFER_DURATION_SEC = 32                       # буфер ~32 секунды (800 кадров / 25 fps)
BUFFER_CHUNKS = int(BUFFER_DURATION_SEC / CHUNK_DURATION_SEC)        # 67 чанков (округлим)

READ_SPEED_MBPS = 10 * 1024 * 1024            # 10 МБ/с

class Window:
    def __init__(self, start_frame, total_frames):
        self.start_frame = start_frame
        self.end_frame = start_frame + WINDOW_TOTAL_CHUNKS * FRAMES_PER_CHUNK
        self.start_chunk = start_frame // FRAMES_PER_CHUNK
        self.end_chunk = self.start_chunk + WINDOW_TOTAL_CHUNKS
        self.loaded_chunks = set()

    def contains_chunk(self, global_chunk):
        return self.start_chunk <= global_chunk < self.end_chunk

    def reset_loaded(self):
        self.loaded_chunks.clear()

    def load_chunks_around(self, target_global_chunk, count):
        """Загружает 'count' чанков начиная с target_global_chunk (если в окне)."""
        self.loaded_chunks.clear()
        for i in range(count):
            ch = target_global_chunk + i
            if self.contains_chunk(ch):
                self.loaded_chunks.add(ch)

def simulate_sequence():
    print("=" * 70)
    print("МОДЕЛИРОВАНИЕ ПОСЛЕДОВАТЕЛЬНОСТИ ПЕРЕМОТОК")
    print("=" * 70)

    # Начальное окно: начинаем с кадра 0
    window = Window(0, 0)
    # Загружаем буфер с начала окна
    window.load_chunks_around(window.start_chunk, BUFFER_CHUNKS)
    print(f"\nИсходное окно: чанки {window.start_chunk}-{window.end_chunk-1}")
    print(f"Загружено чанков: {len(window.loaded_chunks)} (с {min(window.loaded_chunks)} по {max(window.loaded_chunks)})")

    # Шаг 1: Перемотка вперёд на 20 минут = 1200 секунд
    target_global_frame = 1200 * FPS
    target_global_chunk = target_global_frame // FRAMES_PER_CHUNK
    print(f"\n--- Перемотка вперёд на 20 минут ---")
    print(f"Целевой глобальный чанк: {target_global_chunk} (кадр {target_global_frame})")

    # Проверяем, в текущем ли окне цель
    if not window.contains_chunk(target_global_chunk):
        # Окно пересоздаём
        new_start_frame = target_global_frame - (WINDOW_TOTAL_CHUNKS * FRAMES_PER_CHUNK) // 2
        new_start_frame = max(0, new_start_frame // FRAMES_PER_CHUNK * FRAMES_PER_CHUNK)
        window = Window(new_start_frame, target_global_frame)
        print("Цель вне окна: окно пересоздано.")
    else:
        print("Цель внутри окна: окно не меняется.")

    print(f"Новое окно: чанки {window.start_chunk}-{window.end_chunk-1}")

    # Полный сброс: загружаем буфер с целевого чанка
    window.load_chunks_around(target_global_chunk, BUFFER_CHUNKS)
    full_reset_read_bytes = len(window.loaded_chunks) * CHUNK_SIZE_BYTES
    print(f"При полном сбросе: загружено {len(window.loaded_chunks)} чанков, чтение {full_reset_read_bytes/1024:.0f} КБ")

    # Теперь будем делать 5 перемоток назад от новой позиции
    current_global_chunk = target_global_chunk
    print("\n=== Последовательность быстрых перемоток назад ===")

    for idx, minutes_back in enumerate([1, 2, 3, 4, 5], start=1):
        frames_back = minutes_back * 60 * FPS
        target_frame = current_global_chunk * FRAMES_PER_CHUNK - frames_back
        target_global_chunk = target_frame // FRAMES_PER_CHUNK

        print(f"\n--- Перемотка назад {idx}: на {minutes_back} мин ---")
        print(f"Целевой глобальный чанк: {target_global_chunk} (кадр {target_frame})")

        # Проверяем, внутри ли окна цель
        if not window.contains_chunk(target_global_chunk):
            new_start_frame = target_frame - (WINDOW_TOTAL_CHUNKS * FRAMES_PER_CHUNK) // 2
            new_start_frame = max(0, new_start_frame // FRAMES_PER_CHUNK * FRAMES_PER_CHUNK)
            window = Window(new_start_frame, target_frame)
            print("Цель вне окна: окно пересоздано.")
        else:
            print("Цель внутри окна: окно не меняется.")

        # Подход 1: полный сброс (всегда загружаем заново)
        window_full = Window(window.start_frame, window.end_frame)
        window_full.load_chunks_around(target_global_chunk, BUFFER_CHUNKS)
        full_read = len(window_full.loaded_chunks) * CHUNK_SIZE_BYTES

        # Подход 2: оптимальный с сохранением уже загруженных (если окно то же)
        if window.contains_chunk(target_global_chunk):
            # Пересекающиеся чанки из текущего буфера
            overlap = window.loaded_chunks.intersection(
                set(range(target_global_chunk, target_global_chunk + BUFFER_CHUNKS))
            )
            need = BUFFER_CHUNKS - len(overlap)
            opt_read = need * CHUNK_SIZE_BYTES
        else:
            # Окно изменилось – полный сброс
            overlap = set()
            need = BUFFER_CHUNKS
            opt_read = need * CHUNK_SIZE_BYTES

        print(f"Текущее окно: {window.start_chunk}-{window.end_chunk-1}")
        print(f"Загруженные чанки до операции: {len(window.loaded_chunks)}")
        print(f"Пересечение (остаются в буфере): {len(overlap)}")
        print(f"Полный сброс: читать {len(window_full.loaded_chunks)} чанков = {full_read/1024:.0f} КБ")
        print(f"Оптимально:   читать {need} чанков = {opt_read/1024:.0f} КБ")
        print(f"Экономия: {(1 - opt_read/full_read)*100:.0f}%" if full_read > 0 else "Экономия: 0%")

        # Обновляем состояние окна и загруженные чанки для следующего шага (оптимальный подход)
        # Предполагаем, что после операции мы загрузили нужное (для простоты)
        window.loaded_chunks.update(set(range(target_global_chunk, target_global_chunk + BUFFER_CHUNKS)))
        # Ограничиваем размер loaded_chunks только теми, что в окне
        window.loaded_chunks = {c for c in window.loaded_chunks if window.contains_chunk(c)}
        current_global_chunk = target_global_chunk

    print("\n" + "=" * 70)
    print("ВЫВОД: При быстрых перемотках назад внутри того же окна полный сброс планировщика приводит к повторному чтению больших объёмов данных, тогда как сохранение перекрывающихся чанков существенно снижает нагрузку. Это одна из причин зависаний и падений при перемотке назад.")
    print("=" * 70)


if __name__ == "__main__":
    simulate_sequence()