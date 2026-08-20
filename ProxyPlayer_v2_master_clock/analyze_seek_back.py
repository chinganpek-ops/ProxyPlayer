#!/usr/bin/env python3
"""
analyze_seek_back.py – анализ математики чтения при seek назад.

Скрипт моделирует процесс перемотки назад внутри текущего окна и
показывает, какой объём данных приходится читать заново из-за полного
сброса планировщика, и сравнивает с оптимальным сценарием (перемещение
указателя без сброса загруженных чанков).

Запуск: python analyze_seek_back.py
"""

import math

# Параметры модели
FRAMES_PER_CHUNK = 12
CHUNK_SIZE_BYTES = 150 * 1024          # средний размер чанка ~150 КБ
WINDOW_TOTAL_CHUNKS = 626              # около 5 минут при 25 fps
BUFFER_CAPACITY_FRAMES = 800
BUFFER_CAPACITY_CHUNKS = math.ceil(BUFFER_CAPACITY_FRAMES / FRAMES_PER_CHUNK)

# Текущее состояние перед seek назад
current_chunk = 400                    # локальный индекс текущего чанка в окне
loaded_chunks = set(range(350, 450))   # уже загруженные чанки (100 штук)

# Целевой кадр при seek назад (например, на 1 минуту назад)
target_frame = (current_chunk - 30) * FRAMES_PER_CHUNK   # минус 30 чанков
target_chunk = target_frame // FRAMES_PER_CHUNK          # целевой чанк

print("=== Исходные данные ===")
print(f"Окно: {WINDOW_TOTAL_CHUNKS} чанков")
print(f"Текущий чанк: {current_chunk}")
print(f"Целевой чанк (seek назад): {target_chunk}")
print(f"Загруженные чанки до seek: {len(loaded_chunks)} (диапазон {min(loaded_chunks)}-{max(loaded_chunks)})")

# Сценарий 1: полный сброс планировщика (set_normal_mode(0, total_chunks))
# и последующая установка на целевой чанк (set_normal_mode(local_chunk))
# При этом все загруженные чанки очищаются, и нужно прочитать все чанки
# от target_chunk до target_chunk + BUFFER_CAPACITY_CHUNKS (примерно)
chunks_to_read_full = min(BUFFER_CAPACITY_CHUNKS, WINDOW_TOTAL_CHUNKS - target_chunk)
data_to_read_full = chunks_to_read_full * CHUNK_SIZE_BYTES

# Сценарий 2: без сброса – перемещаем указатель, сохраняя пересекающиеся
# загруженные чанки (если целевой чанк попадает в диапазон загруженных)
overlap_chunks = loaded_chunks.intersection(
    range(target_chunk, target_chunk + BUFFER_CAPACITY_CHUNKS)
)
chunks_to_read_opt = max(0, BUFFER_CAPACITY_CHUNKS - len(overlap_chunks))
data_to_read_opt = chunks_to_read_opt * CHUNK_SIZE_BYTES

print("\n=== Сценарий 1: полный сброс планировщика ===")
print(f"Очищено загруженных чанков: {len(loaded_chunks)}")
print(f"Необходимо прочитать чанков для заполнения буфера: {chunks_to_read_full}")
print(f"Объём чтения: {data_to_read_full / 1024:.0f} КБ")

print("\n=== Сценарий 2: перемещение указателя с сохранением перекрытия ===")
print(f"Пересекающихся чанков (уже в буфере): {len(overlap_chunks)}")
print(f"Необходимо прочитать новых чанков: {chunks_to_read_opt}")
print(f"Объём чтения: {data_to_read_opt / 1024:.0f} КБ")

print("\n=== Вывод ===")
if data_to_read_opt < data_to_read_full:
    savings = (1 - data_to_read_opt / data_to_read_full) * 100
    print(f"Оптимальный сценарий экономит {savings:.0f}% чтения.")
else:
    print("В данном примере полный сброс не приводит к избыточному чтению (пересечение минимально).")
print("Однако в реальности при частых перемотках и большом окне избыточное чтение и гонки потоков усугубляют проблему.")

# Дополнительно: оценка времени при скорости чтения 10 МБ/с
read_speed_mbps = 10 * 1024 * 1024  # байт/с
time_full = data_to_read_full / read_speed_mbps
time_opt = data_to_read_opt / read_speed_mbps
print(f"\nВремя чтения (при 10 МБ/с):")
print(f"  Полный сброс: {time_full:.2f} сек")
print(f"  Оптимально:   {time_opt:.2f} сек")