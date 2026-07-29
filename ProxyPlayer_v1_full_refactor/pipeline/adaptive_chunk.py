"""
adaptive_chunk.py – адаптивный размер чанка для ProxyPlayer v1.
Возвращает размер чанка и шаг для JKL-перемотки в зависимости от скорости.
"""

from config.timebase import FRAMES_PER_CHUNK


class AdaptiveChunkStrategy:
    """
    Определяет размер чанка и шаг для JKL-перемотки на основе текущей скорости.
    
    При нормальной скорости используется базовый размер чанка (12 кадров).
    При ускоренной перемотке размер увеличивается для снижения количества чтений,
    а шаг между загружаемыми чанками растёт пропорционально скорости.
    """

    def __init__(self, base_frames_per_chunk: int = FRAMES_PER_CHUNK):
        self.base_frames_per_chunk = base_frames_per_chunk

    def get_chunk_size(self, speed: float) -> int:
        """
        Возвращает количество кадров в чанке для заданной скорости.
        
        speed = 1.0 → 12 кадров (стандарт)
        speed ≤ 4.0 → 24 кадра
        speed > 4.0 → 48 кадров
        """
        if speed <= 1.0:
            return self.base_frames_per_chunk
        elif speed <= 4.0:
            return self.base_frames_per_chunk * 2
        else:
            return self.base_frames_per_chunk * 4

    def get_fast_forward_stride(self, speed: float) -> int:
        """
        Возвращает шаг между загружаемыми чанками для JKL-перемотки.
        
        speed = 1.0 → 1 (каждый чанк)
        speed = 2.0 → 2 (каждый второй)
        speed = 4.0 → 4 (каждый четвёртый)
        speed = 8.0 → 8 (каждый восьмой)
        """
        return max(1, int(speed))

    def get_lookahead_chunks(self, speed: float, buffer_size_frames: int) -> int:
        """
        Возвращает количество чанков для предварительной загрузки.
        
        При нормальной скорости: буфер заполняется на 30% вперёд.
        При ускоренной: загружается по 2 чанка (быстрый темп).
        """
        if speed <= 1.0:
            # 30% буфера в чанках
            return max(1, int(buffer_size_frames * 0.3 / self.base_frames_per_chunk))
        else:
            return 2