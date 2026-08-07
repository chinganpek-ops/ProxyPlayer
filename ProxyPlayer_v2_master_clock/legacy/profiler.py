"""
profiler.py – сбор метрик производительности Dalet Proxy Player (production).
Запускается параллельно с плеером и пишет статистику в JSON Lines файл.
Расширенный мониторинг: аудиобуфер, аудиовыход, видеобуфер.
"""

import sys
import time
import json
import subprocess
import threading
from pathlib import Path
import logging
import psutil
import numpy as np

logger = logging.getLogger(__name__)


def monitor(pid: int, interval: float = 1.0, output_path: Path = None,
            player_controller=None):
    """
    Собирает метрики процесса с заданным pid.
    Пишет JSON-строки в output_path (если указан).
    Опционально принимает ссылку на PlayerController для чтения внутренней статистики.
    """
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        logger.error(f"Процесс {pid} не найден")
        return

    out_file = None
    if output_path:
        try:
            out_file = open(output_path, 'w', encoding='utf-8')
            logger.info(f"Профилирование в {output_path}")
        except OSError as e:
            logger.error(f"Не удалось открыть файл {output_path}: {e}")
            return

    try:
        while proc.is_running():
            metrics = {
                'timestamp': time.time(),
                'cpu_percent': proc.cpu_percent(interval=0),
                'memory_mb': proc.memory_info().rss / 1024 / 1024,
                'io_counters': proc.io_counters()._asdict() if proc.io_counters() else {},
                'num_threads': proc.num_threads(),
            }

            # Статистика из плеера (если передан controller)
            if player_controller:
                try:
                    # Видеобуфер
                    if hasattr(player_controller, 'video_buffer'):
                        vb = player_controller.video_buffer
                        metrics['video_buffer_main'] = {
                            'count': vb.count,
                            'free_slots': vb.free_slots,
                            'latest_pts': vb.latest_pts(),
                        }
                    # Аудиобуфер
                    if hasattr(player_controller, 'audio_buffer') and player_controller.audio_buffer:
                        metrics['audio_buffer'] = player_controller.audio_buffer.get_stats()
                    # Аудиовыход
                    if hasattr(player_controller, 'audio_output') and player_controller.audio_output:
                        metrics['audio_clock'] = player_controller.audio_output.current_clock()
                except Exception as e:
                    logger.debug(f"Ошибка сбора метрик плеера: {e}")

            line = json.dumps(metrics, ensure_ascii=False)
            if out_file:
                out_file.write(line + '\n')
                out_file.flush()
            else:
                # В консоль выводим только при явном запуске без файла
                print(line)

            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("Мониторинг прерван пользователем")
    except Exception as e:
        logger.exception(f"Ошибка в цикле мониторинга: {e}")
    finally:
        if out_file:
            out_file.close()
            logger.info(f"Файл метрик закрыт: {output_path}")


def main():
    """Запускает плеер с мониторингом."""
    if len(sys.argv) < 2:
        print("Использование: python profiler.py <путь к MP4> [интервал сек] [выходной файл]")
        sys.exit(1)

    mp4_path = sys.argv[1]
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    output = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("metrics.jsonl")

    # Настройка базового логирования для самого профайлера
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s')
    logger.info(f"Запуск плеера: {mp4_path}")

    # Запускаем плеер как отдельный процесс
    cmd = [sys.executable, "main.py", mp4_path]
    try:
        proc = subprocess.Popen(cmd)
        pid = proc.pid
        logger.info(f"Плеер запущен (PID {pid}). Мониторинг с интервалом {interval} с.")
    except Exception as e:
        logger.error(f"Не удалось запустить плеер: {e}")
        sys.exit(1)

    # Ждём пока появится процесс (иногда нужна задержка)
    time.sleep(2)
    monitor(pid, interval, output)
    
    # Завершаем плеер
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    logger.info("Мониторинг завершён, плеер остановлен.")


if __name__ == "__main__":
    main()