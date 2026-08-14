"""
chunk_pipeline.py – трёхэтапный конвейер загрузки и декодирования чанков.
v2 – интеграция с MasterClock: AudioDecoderStage пушит аудио напрямую в MasterClock.
MultiTrackAudioBuffer больше не используется.

Исправления (включая тройной буфер, фильтрацию окон и эталонный PTS):
- Фильтрация аудиодорожек: только дорожки 2 и 3.
- VideoDecoderStage использует threading.Condition вместо активного ожидания.
- При остановке конвейера очищаются все внутренние очереди.
- Добавлен метод set_video_buffer для перенаправления вывода видео-декодера
  в другой буфер без остановки стадий (для тройного буфера).
- Добавлен метод flush() для очистки очередей при переключении буфера.
- VideoDecoderStage фильтрует пакеты по PTS текущего окна, чтобы
  отбрасывать устаревшие пакеты, полученные до перемотки.
- Добавлен эталонный PTS (seek_reference): все кадры с PTS меньше эталона
  отбрасываются на уровне вставки в буфер. Это исключает подмешивание старых кадров
  даже в момент гонки потоков.
"""

import time
import queue
import threading
import logging
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np

from buffer.frame_buffer import FrameRingBuffer
from decode.decoder import Decoder
from decode.audio_decoder import AudioDecoder
from file_io.win_sequential_reader import WinSequentialReader
from index.lazy_index import IndexWindow
from index.moov_builder import SEGMENT_SIZE
from pipeline.stream_scheduler import StreamScheduler, PlaybackMode
from pipeline.adaptive_chunk import AdaptiveChunkStrategy
from config.timebase import SAMPLES_PER_VIDEO_FRAME, FRAMES_PER_CHUNK

logger = logging.getLogger(__name__)

# --- Логгер для мониторинга seek (используется совместно с playback_engine) ---
monitor_logger = logging.getLogger("SeekMonitor")
monitor_logger.setLevel(logging.DEBUG)
if not monitor_logger.handlers:
    _mon_handler = logging.FileHandler("seek_monitor.log", encoding="utf-8")
    _mon_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    monitor_logger.addHandler(_mon_handler)
monitor_logger.propagate = False

VideoPacket = Tuple[bytes, int]                       # (data, pts)
AudioPacket = Tuple[int, bytes, bytes, int, bool]     # (track, data1, data2, pts, need_fade)

RAW_QUEUE_SIZE = 10
VIDEO_QUEUE_SIZE = 10
AUDIO_QUEUE_SIZE = 50


class Stage(threading.Thread):
    def __init__(self, name: str):
        super().__init__(daemon=True, name=name)
        self.stop_event = threading.Event()

    def stop(self):
        self.stop_event.set()


class ReaderStage(Stage):
    def __init__(self, mp4_path: Path, pipeline: 'ChunkPipeline',
                 scheduler: StreamScheduler, output_queue: queue.Queue,
                 adaptive: AdaptiveChunkStrategy):
        super().__init__("ReaderStage")
        self._reader = WinSequentialReader(mp4_path, rate_limit=0, overlapped=False)
        self._pipeline = pipeline
        self._scheduler = scheduler
        self._output_queue = output_queue
        self._adaptive = adaptive
        logger.info("ReaderStage инициализирован")

    def run(self):
        logger.info("ReaderStage запущен")
        while not self.stop_event.is_set():
            local_chunk = self._scheduler.get_next_chunk()
            if local_chunk is None:
                self.stop_event.wait(0.1)
                continue

            window = self._pipeline.get_window_snapshot()

            if local_chunk < 0 or local_chunk >= window.total_chunks:
                self._scheduler.mark_chunk_failed(local_chunk)
                continue

            first_local_frame = local_chunk * FRAMES_PER_CHUNK
            if first_local_frame >= len(window.cached_offsets):
                self._scheduler.mark_chunk_failed(local_chunk)
                continue

            chunk_start_offset = int(window.cached_offsets[first_local_frame])
            next_local_frame = min(first_local_frame + FRAMES_PER_CHUNK, len(window.video_records))
            if next_local_frame < len(window.cached_offsets):
                chunk_end_offset = int(window.cached_offsets[next_local_frame])
            else:
                chunk_end_offset = chunk_start_offset + window.chunk_sizes[local_chunk]

            size = max(0, chunk_end_offset - chunk_start_offset)
            size = int(size)
            if size <= 0:
                self._scheduler.mark_chunk_failed(local_chunk)
                continue

            logger.debug(f"Чтение чанка {local_chunk}, смещение={chunk_start_offset}, размер={size}")
            try:
                data = self._reader.read_sequential(chunk_start_offset, size)
                if data:
                    self._output_queue.put((local_chunk, data, window), timeout=2.0)
                    logger.debug(f"Чанк {local_chunk} прочитан и помещён в очередь")
                else:
                    self._scheduler.mark_chunk_failed(local_chunk)
            except queue.Full:
                logger.warning("raw_queue переполнена, повторная попытка для чанка %d", local_chunk)
                self.stop_event.wait(0.1)
            except Exception as e:
                logger.error(f"Ошибка чтения чанка {local_chunk}: {e}", exc_info=True)
                self._scheduler.mark_chunk_failed(local_chunk)

    def stop(self):
        super().stop()
        self._reader.close()
        logger.info("ReaderStage остановлен")


class DemuxerStage(Stage):
    def __init__(self, input_queue: queue.Queue,
                 video_queue: queue.Queue, audio_queue: queue.Queue):
        super().__init__("DemuxerStage")
        self._input_queue = input_queue
        self._video_queue = video_queue
        self._audio_queue = audio_queue
        logger.info("DemuxerStage инициализирован")

    def run(self):
        logger.info("DemuxerStage запущен")
        while not self.stop_event.is_set():
            try:
                local_chunk, raw_data, window_snapshot = self._input_queue.get(timeout=0.5)
                logger.debug(f"DemuxerStage получил чанк {local_chunk}")
            except queue.Empty:
                continue

            try:
                v_packets, a_packets = self._demux(local_chunk, raw_data, window_snapshot)
                if v_packets:
                    self._safe_put(self._video_queue, (local_chunk, v_packets), "video")
                if a_packets:
                    self._safe_put(self._audio_queue, (local_chunk, a_packets), "audio")
            except Exception as e:
                logger.error(f"Ошибка демукса чанка {local_chunk}: {e}", exc_info=True)

    def _safe_put(self, q: queue.Queue, item, qname: str, max_retries=5, retry_delay=0.2):
        for attempt in range(max_retries):
            try:
                q.put(item, timeout=1.0)
                return
            except queue.Full:
                logger.warning("Очередь %s переполнена (попытка %d/%d)", qname, attempt+1, max_retries)
                self.stop_event.wait(retry_delay)
        logger.error("Не удалось поместить в очередь %s после %d попыток, отбрасываю", qname, max_retries)

    def _demux(self, local_chunk: int, raw_data: bytes, window: IndexWindow) -> Tuple[List[VideoPacket], List[AudioPacket]]:
        video_packets = []
        audio_packets = []

        global_start_frame = (window.window_start_chunk + local_chunk) * FRAMES_PER_CHUNK
        first_local_frame = local_chunk * FRAMES_PER_CHUNK

        if first_local_frame >= len(window.cached_offsets):
            return video_packets, audio_packets
        chunk_start_offset = int(window.cached_offsets[first_local_frame])
        cached_offs = window.cached_offsets

        # ---------- видео ----------
        for i in range(FRAMES_PER_CHUNK):
            abs_idx = global_start_frame + i
            local_frame = first_local_frame + i
            if local_frame < 0 or local_frame >= len(window.video_records):
                continue
            if local_frame >= len(cached_offs):
                continue
            abs_off = int(cached_offs[local_frame])

            if i < FRAMES_PER_CHUNK - 1:
                next_local = local_frame + 1
                if next_local < len(cached_offs):
                    next_off = int(cached_offs[next_local])
                    size = next_off - abs_off
                else:
                    size = 0
            else:
                next_local = local_frame + 1
                if next_local < len(cached_offs):
                    chunk_end = int(cached_offs[next_local])
                else:
                    chunk_end = chunk_start_offset + len(raw_data)
                size = chunk_end - abs_off

            if size <= 0:
                continue
            rel_start = abs_off - chunk_start_offset
            if rel_start < 0 or rel_start + size > len(raw_data):
                continue

            sample = raw_data[rel_start:rel_start + size]
            pts = abs_idx * SAMPLES_PER_VIDEO_FRAME
            video_packets.append((sample, pts))

        # ---------- аудио ----------
        if local_chunk < len(window.audio_chunks):
            audio_chunk = window.audio_chunks[local_chunk]
            if audio_chunk:
                for track_id, entries in audio_chunk.items():
                    if track_id not in (2, 3):
                        continue
                    for entry in entries:
                        size2 = entry.get('size2', 0)
                        rel = entry['abs_offset'] - chunk_start_offset

                        d1 = b''
                        if 0 <= rel < len(raw_data):
                            d1 = raw_data[rel:rel + entry['size1']]

                        d2 = b''
                        if size2 > 0:
                            rel2 = rel + entry['size1']
                            if 0 <= rel2 < len(raw_data):
                                d2 = raw_data[rel2:rel2 + size2]

                        audio_packets.append((track_id, d1, d2, entry['pts'], False))

        return video_packets, audio_packets


class VideoDecoderStage(Stage):
    def __init__(self, decoder: Decoder, video_buffer: FrameRingBuffer,
                 input_queue: queue.Queue):
        super().__init__("VideoDecoderStage")
        self._decoder = decoder
        self._buffer = video_buffer
        self._input_queue = input_queue
        self._buffer_cond = threading.Condition()

        # Диапазон PTS активного окна
        self._pts_min = 0
        self._pts_max = 0

        # Эталонный PTS после seek (кадры с меньшим PTS отбрасываются)
        self._seek_pts_reference = 0

        logger.info("VideoDecoderStage инициализирован")

    def set_active_window(self, window: IndexWindow):
        """
        Устанавливает допустимый диапазон PTS на основе текущего окна.
        Пакеты с PTS вне этого диапазона будут отбрасываться.
        """
        start_pts = window.window_start_chunk * FRAMES_PER_CHUNK * SAMPLES_PER_VIDEO_FRAME
        end_pts = (window.window_start_chunk + window.total_chunks) * FRAMES_PER_CHUNK * SAMPLES_PER_VIDEO_FRAME
        with self._buffer_cond:
            self._pts_min = start_pts
            self._pts_max = end_pts
            monitor_logger.info(f"VIDEO_DECODER_WINDOW_SET min_pts={self._pts_min} max_pts={self._pts_max}")

    def set_seek_reference(self, pts: int):
        """
        Устанавливает эталонный PTS после seek.
        Все кадры с PTS меньше эталонного будут отброшены при вставке в буфер.
        """
        with self._buffer_cond:
            self._seek_pts_reference = pts
            monitor_logger.info(f"VIDEO_DECODER_SEEK_REF pts={pts}")

    def _is_pts_valid(self, pts: int) -> bool:
        """Проверяет, попадает ли PTS в активное окно."""
        return self._pts_min <= pts < self._pts_max

    def run(self):
        logger.info("VideoDecoderStage запущен")
        while not self.stop_event.is_set():
            try:
                local_chunk, packets = self._input_queue.get(timeout=0.5)
                logger.debug(f"VideoDecoderStage получил чанк {local_chunk}, пакетов: {len(packets)}")
            except queue.Empty:
                continue

            for data, pts in packets:
                # Отбрасываем пакеты, не принадлежащие активному окну
                if not self._is_pts_valid(pts):
                    monitor_logger.debug(f"PACKET_OUTSIDE_WINDOW pts={pts}")
                    continue

                try:
                    filtered = self._decoder.filter_avcc(data)
                    if not filtered:
                        continue
                    frames = self._decoder.decode_sample(filtered)
                    for frame in frames:
                        self._push_frame(frame, pts)
                except Exception as e:
                    logger.debug(f"Ошибка декодирования видео: {e}")

    def _push_frame(self, frame: np.ndarray, pts: int):
        # Жёсткая фильтрация: кадр с PTS меньше эталонного не должен попасть в буфер
        if pts < self._seek_pts_reference:
            monitor_logger.debug(f"FILTER_OUTDATED pts={pts} < ref={self._seek_pts_reference}")
            return

        with self._buffer_cond:
            while not self.stop_event.is_set():
                if self._buffer.try_push(frame, pts):
                    monitor_logger.debug(f"FRAME_ADDED pts={pts} buffer_count={self._buffer.count}")
                    return
                monitor_logger.debug(f"BUFFER_FULL pts={pts} buffer_count={self._buffer.count}")
                self._buffer_cond.wait(timeout=0.1)
        monitor_logger.debug(f"FRAME_DROPPED pts={pts}")
        logger.debug("Видеобуфер переполнен, кадр отброшен (pts=%d)", pts)

    def notify_buffer_available(self):
        """Вызывается, когда в буфере освобождается место."""
        with self._buffer_cond:
            self._buffer_cond.notify()

    def set_buffer(self, new_buffer: FrameRingBuffer):
        """
        Заменяет целевой буфер без остановки стадии.
        Безопасно вызывать из другого потока.
        """
        with self._buffer_cond:
            self._buffer = new_buffer
            monitor_logger.info(f"VIDEO_DECODER_BUFFER_SWITCH new_buffer={id(new_buffer)}")
            self._buffer_cond.notify_all()


class AudioDecoderStage(Stage):
    def __init__(self, decoders: List[AudioDecoder], master_clock,
                 input_queue: queue.Queue):
        super().__init__("AudioDecoderStage")
        self._decoders = decoders
        self._master_clock = master_clock
        self._input_queue = input_queue
        self._fade_len = 8
        logger.info("AudioDecoderStage инициализирован")

    def run(self):
        logger.info("AudioDecoderStage запущен")
        while not self.stop_event.is_set():
            try:
                local_chunk, packets = self._input_queue.get(timeout=0.5)
                logger.debug(f"AudioDecoderStage получил чанк {local_chunk}, пакетов: {len(packets)}")
            except queue.Empty:
                continue

            for track_id, d1, d2, pts, need_fade in packets:
                if track_id not in (2, 3):
                    continue
                decoder_idx = track_id - 2
                if decoder_idx >= len(self._decoders) or self._decoders[decoder_idx] is None:
                    continue

                decoder = self._decoders[decoder_idx]
                try:
                    pcm1 = decoder.decode(d1)
                    pcm2 = decoder.decode(d2)
                    pcm_block = np.concatenate([pcm1, pcm2])
                    logger.debug(f"Аудио декодировано: трек {track_id}, pts={pts}, сэмплов={len(pcm_block)}")
                except Exception:
                    continue

                if need_fade and len(pcm_block) >= self._fade_len:
                    pcm_block[:self._fade_len] *= np.linspace(0, 1, self._fade_len)

                # Отправляем напрямую в MasterClock
                if self._master_clock:
                    self._master_clock.push_audio(track_id, pcm_block)


class ChunkPipeline:
    def __init__(self, mp4_path: Path, window: IndexWindow,
                 video_decoder: Decoder, audio_decoders: List[AudioDecoder],
                 video_buffer: FrameRingBuffer, master_clock=None):
        self._mp4_path = mp4_path
        self._video_decoder = video_decoder
        self._audio_decoders = audio_decoders
        self._video_buffer = video_buffer
        self._master_clock = master_clock

        self._window_lock = threading.Lock()
        self._window = window

        self._scheduler = StreamScheduler(AdaptiveChunkStrategy())
        self._scheduler.set_buffer(self._video_buffer)

        self._raw_queue = queue.Queue(maxsize=RAW_QUEUE_SIZE)
        self._video_queue = queue.Queue(maxsize=VIDEO_QUEUE_SIZE)
        self._audio_queue = queue.Queue(maxsize=AUDIO_QUEUE_SIZE)

        self._stages: List[Stage] = []

    def set_master_clock(self, master_clock):
        """Подключает MasterClock после создания."""
        self._master_clock = master_clock
        for stage in self._stages:
            if isinstance(stage, AudioDecoderStage):
                stage._master_clock = master_clock

    def get_window_snapshot(self) -> IndexWindow:
        with self._window_lock:
            return self._window

    def update_window(self, new_window: IndexWindow):
        with self._window_lock:
            self._window = new_window
            self._scheduler.set_normal_mode(0, new_window.total_chunks)
        # Обновляем допустимый диапазон PTS для видео-декодера
        for stage in self._stages:
            if isinstance(stage, VideoDecoderStage):
                stage.set_active_window(new_window)

    def set_video_buffer(self, new_buffer: FrameRingBuffer):
        """
        Перенаправляет вывод VideoDecoderStage в новый буфер.
        Конвейер продолжает работать без остановки.
        """
        self._video_buffer = new_buffer
        # Обновляем планировщик
        self._scheduler.set_buffer(new_buffer)
        # Обновляем стадии (если они уже запущены)
        for stage in self._stages:
            if isinstance(stage, VideoDecoderStage):
                stage.set_buffer(new_buffer)

    def set_seek_reference(self, pts: int):
        """
        Устанавливает эталонный PTS для фильтрации старых кадров.
        Проксирует в VideoDecoderStage.
        """
        for stage in self._stages:
            if isinstance(stage, VideoDecoderStage):
                stage.set_seek_reference(pts)

    def flush(self):
        """Очищает все внутренние очереди, отбрасывая устаревшие пакеты."""
        for q in (self._raw_queue, self._video_queue, self._audio_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        logger.debug("Очереди конвейера очищены")

    def start(self, start_local_chunk: int):
        self._scheduler.set_normal_mode(start_local_chunk, self.get_window_snapshot().total_chunks)

        reader = ReaderStage(self._mp4_path, self, self._scheduler,
                             self._raw_queue, AdaptiveChunkStrategy())
        demuxer = DemuxerStage(self._raw_queue,
                               self._video_queue, self._audio_queue)
        video_dec = VideoDecoderStage(self._video_decoder, self._video_buffer,
                                      self._video_queue)
        audio_dec = AudioDecoderStage(self._audio_decoders, self._master_clock,
                                      self._audio_queue)

        # Устанавливаем окно для видео-декодера ДО запуска
        video_dec.set_active_window(self.get_window_snapshot())
        # Сбрасываем эталонный PTS (будет установлен заново при seek)
        video_dec.set_seek_reference(0)

        self._stages = [reader, demuxer, video_dec, audio_dec]
        for stage in self._stages:
            stage.start()
        logger.info("ChunkPipeline: все стадии запущены")

    def stop(self):
        logger.info("ChunkPipeline: остановка стадий")
        for stage in self._stages:
            stage.stop()
        for stage in self._stages:
            if isinstance(stage, VideoDecoderStage):
                stage.notify_buffer_available()
        for stage in self._stages:
            stage.join(timeout=2.0)
        self._stages.clear()

        # Очистка всех внутренних очередей
        self.flush()
        logger.info("ChunkPipeline: все стадии остановлены, очереди очищены")