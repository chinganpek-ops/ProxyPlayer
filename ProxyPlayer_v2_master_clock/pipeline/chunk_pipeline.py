"""
chunk_pipeline.py – трёхэтапный конвейер загрузки и декодирования чанков.
v2 – интеграция с MasterClock: AudioDecoderStage пушит аудио напрямую в MasterClock.
MultiTrackAudioBuffer больше не используется.

Исправления (исходные):
- Один AudioDecoderStage для обеих дорожек (синхронность левого/правого уха).
- Гистерезис аудиобуфера: остановка при 55%, возобновление при 45%.
- Скорость подачи аудио ограничена 1.25x в зоне 45–55%.
- Видео: тройной буфер, фильтрация по PTS, эталонный PTS после seek.
- Очистка очередей при остановке/смене окна.
- Логирование синхронизации (SYNC_AUDIO) в sync_monitor.log через SyncMonitor.

ИЗМЕНЕНИЯ (правки продакшен-ревью):
- ReaderStage: при queue.Full уже прочитанные с диска данные раньше
  молча терялись (I/O выполнен, а put в очередь — нет), и чанк не
  помечался ни успешным, ни неудачным. Теперь до 5 попыток отдать уже
  прочитанные данные в очередь, и только если не вышло — mark_chunk_failed,
  чтобы планировщик выдал этот чанк повторно.
- DemuxerStage: теперь получает ссылку на StreamScheduler и вызывает
  mark_chunk_failed(), если после исчерпания ретраев не удалось положить
  пакеты в video_queue/audio_queue (раньше это тихо терялось).
  _safe_put() возвращает bool вместо None.
- AudioDecoderStage: except Exception при декодировании аудио больше не
  проглатывается молча — добавлено сообщение с track/pts для диагностики
  (поведение — continue — не изменилось).
- ChunkPipeline: добавлен extend_window() — расширяет активное окно при
  росте live-файла, не сбрасывая текущую позицию чтения/декодирования
  (в отличие от update_window(), который предназначен для полной смены
  окна, например при seek, и всегда перематывает планировщик на начало).
- ChunkPipeline.start(): DemuxerStage теперь создаётся с scheduler.
- НОВОЕ: shift_window() — плавное переключение на скользящее окно
  (LazyIndex.build_slid_window()): в отличие от update_window(), не
  сбрасывает позицию, а пересчитывает её через
  StreamScheduler.shift_loaded(), сохраняя уже загруженные чанки; в
  отличие от extend_window(), умеет сдвигать не только конец окна, но и
  начало. Корректность при этом обеспечивается overlap'ом самого
  new_window (строит LazyIndex), а не логикой этого метода — см.
  докстринг shift_window().

Логика демукса, декодирования, гистерезиса и PTS-фильтрации не менялась.
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
from utils.sync_logger import sync_monitor_logger

logger = logging.getLogger(__name__)

# --- Логгер для мониторинга видео (seek) ---
monitor_logger = logging.getLogger("SeekMonitor")
monitor_logger.setLevel(logging.DEBUG)
if not monitor_logger.handlers:
    from logging.handlers import RotatingFileHandler
    _mon_handler = RotatingFileHandler(
        "seek_monitor.log", encoding="utf-8",
        maxBytes=50 * 1024 * 1024, backupCount=5,
    )
    _mon_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    monitor_logger.addHandler(_mon_handler)
monitor_logger.propagate = False

# --- Логгер для мониторинга аудио ---
audio_monitor_logger = logging.getLogger("AudioMonitor")
audio_monitor_logger.setLevel(logging.DEBUG)
if not audio_monitor_logger.handlers:
    from logging.handlers import RotatingFileHandler
    _audio_mon_handler = RotatingFileHandler(
        "audio_monitor.log", encoding="utf-8",
        maxBytes=50 * 1024 * 1024, backupCount=5,
    )
    _audio_mon_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    audio_monitor_logger.addHandler(_audio_mon_handler)
audio_monitor_logger.propagate = False

VideoPacket = Tuple[bytes, int, bool]                 # (data, pts, is_idr)
AudioPacket = Tuple[int, bytes, bytes, int, bool]     # (track, data1, data2, pts, need_fade)

RAW_QUEUE_SIZE = 10
VIDEO_QUEUE_SIZE = 10
AUDIO_QUEUE_SIZE = 50

# До этой скорости JKL декодирует чанк целиком (плавная картинка).
# Выше — только опорный кадр каждого чанка.
SCRUB_FULL_DECODE_SPEED = 2.0


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
        self._reader = WinSequentialReader(mp4_path)
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
                self._reader.seek(chunk_start_offset)
                data = self._reader.read(size)
            except Exception as e:
                logger.error(f"Ошибка чтения чанка {local_chunk}: {e}", exc_info=True)
                self._scheduler.mark_chunk_failed(local_chunk)
                continue

            if not data:
                self._scheduler.mark_chunk_failed(local_chunk)
                continue

            # Данные уже прочитаны с диска (I/O выполнен) — при переполнении
            # очереди не отбрасываем их, а пробуем повторно; теряем чанк
            # только если совсем не удалось его передать дальше.
            put_ok = False
            for attempt in range(5):
                try:
                    self._output_queue.put((local_chunk, data, window), timeout=2.0)
                    put_ok = True
                    break
                except queue.Full:
                    logger.warning("raw_queue переполнена, попытка %d/5 для чанка %d",
                                    attempt + 1, local_chunk)
                    if self.stop_event.is_set():
                        break
                    self.stop_event.wait(0.1)

            if put_ok:
                logger.debug(f"Чанк {local_chunk} прочитан и помещён в очередь")
            else:
                logger.error("Не удалось поместить чанк %d в очередь после повторных "
                             "попыток, помечаю как неудачный", local_chunk)
                self._scheduler.mark_chunk_failed(local_chunk)

    def refresh_file_size(self):
        """
        Просит ридер переспросить у ОС актуальный размер файла.

        Нужно для растущего MP4: ридер создаётся один раз на весь сеанс, а
        WinSequentialReader клампит чтение по размеру файла, определённому в
        момент открытия. Без этого обновления чтение у live-края упирается в
        устаревшую границу и молча возвращает пустые данные. Вызывается по
        событию роста индекса — см. ChunkPipeline.notify_file_grew().
        """
        try:
            self._reader.refresh_file_size()
        except Exception:
            logger.exception("ReaderStage: не удалось обновить размер файла")

    def stop(self):
        super().stop()
        self._reader.close()
        logger.info("ReaderStage остановлен")


class DemuxerStage(Stage):
    def __init__(self, input_queue: queue.Queue,
                 video_queue: queue.Queue,
                 audio_queue: queue.Queue,
                 scheduler: StreamScheduler):
        super().__init__("DemuxerStage")
        self._input_queue = input_queue
        self._video_queue = video_queue
        self._audio_queue = audio_queue
        self._scheduler = scheduler
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
                v_ok = True
                a_ok = True
                if v_packets:
                    v_ok = self._safe_put(self._video_queue, (local_chunk, v_packets), "video")
                if a_packets:
                    a_ok = self._safe_put(self._audio_queue, (local_chunk, a_packets), "audio")
                if not (v_ok and a_ok):
                    # Не удалось доставить пакеты дальше по конвейеру —
                    # сообщаем планировщику, чтобы чанк выдали повторно,
                    # а не считали молча "загруженным".
                    self._scheduler.mark_chunk_failed(local_chunk)
            except Exception as e:
                logger.error(f"Ошибка демукса чанка {local_chunk}: {e}", exc_info=True)
                self._scheduler.mark_chunk_failed(local_chunk)

    def _safe_put(self, q: queue.Queue, item, qname: str, max_retries=5, retry_delay=0.2) -> bool:
        """Возвращает True, если элемент удалось поместить в очередь, иначе False."""
        for attempt in range(max_retries):
            try:
                q.put(item, timeout=1.0)
                return True
            except queue.Full:
                logger.warning("Очередь %s переполнена (попытка %d/%d)", qname, attempt+1, max_retries)
                self.stop_event.wait(retry_delay)
        logger.error("Не удалось поместить в очередь %s после %d попыток, отбрасываю", qname, max_retries)
        return False

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
            # Помечаем опорные кадры (та же проверка, что в
            # moov_builder.get_idr_indices_from_mmap). Нужно для режима
            # перемотки: там чанки читаются с пропуском, состояние декодера
            # между ними не сохраняется, и начинать декодирование можно
            # ТОЛЬКО с IDR. GOP (15 кадров) не кратен чанку (12), поэтому
            # первый кадр чанка опорным обычно не является.
            try:
                f7 = int(window.video_records[local_frame]['f7'])
                is_idr = 27 <= f7 <= 30
            except Exception:
                is_idr = False
            video_packets.append((sample, pts, is_idr))

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

        self._pts_min = 0
        self._pts_max = 0
        self._seek_pts_reference = 0

        # --- режим скраба (JKL) ---
        # При ускоренной перемотке кадры НЕ идут в кольцевой буфер: он
        # построен на возрастающем PTS, а при движении назад кадры приходят
        # с убывающим — порядок и drop_until() ломаются. Вместо этого
        # последний декодированный кадр кладётся в одну ячейку, откуда его
        # забирает PlaybackEngine. Синхронизировать не с чем: звук при JKL
        # приглушён.
        self._scrub_mode = False
        self._scrub_speed = 1.0
        self._scrub_frame = None      # (pts, frame)
        self._scrub_lock = threading.Lock()

        logger.info("VideoDecoderStage инициализирован")

    def set_scrub_mode(self, enabled: bool, speed: float = 1.0):
        """
        Включает/выключает режим ускоренной перемотки.

        speed определяет, декодировать ли чанк целиком или только опорный
        кадр: на x2 нужна плавность (12 кадров чанка ≈ 25 изображений в
        секунду), на x4/x8 достаточно одного опорного кадра на чанк —
        частота смены и так получается 8-17 в секунду, а нагрузка на
        декодер кратно ниже.
        """
        with self._scrub_lock:
            self._scrub_mode = enabled
            self._scrub_speed = speed
            if not enabled:
                self._scrub_frame = None
        monitor_logger.info(f"SCRUB_MODE enabled={enabled} speed={speed}")

    def take_scrub_frame(self):
        """Возвращает последний кадр скраба (pts, frame) или None."""
        with self._scrub_lock:
            return self._scrub_frame

    def _decode_scrub(self, packets):
        """
        Декодирует пакеты чанка в режиме перемотки.

        Декодирование НАЧИНАЕТСЯ С ОПОРНОГО КАДРА. Это принципиально: в
        режиме перемотки чанки читаются с пропуском (stride), состояние
        декодера между ними не сохраняется, и кадр, зависящий от
        предыдущих, декодировать нельзя — получится мусор или ничего.

        Раньше здесь бралcя packets[:1] — просто ПЕРВЫЙ пакет чанка,
        в предположении, что опорный кадр стоит в начале каждого чанка.
        На реальном материале это неверно: GOP 15 кадров не кратен чанку
        в 12 кадров, поэтому чанк начинается с IDR лишь каждый пятый раз.
        В остальных случаях перемотка на скорости x4/x8 не показывала
        ничего.

        До SCRUB_FULL_DECODE_SPEED декодируем от IDR до конца чанка —
        картинка плавная. Выше — только сам опорный кадр.

        Если в чанке нет ни одного IDR (при GOP 15 и чанке 12 таких около
        20%), чанк пропускается целиком: лучше задержать предыдущий кадр,
        чем показать артефакты.
        """
        with self._scrub_lock:
            full = self._scrub_speed <= SCRUB_FULL_DECODE_SPEED

        idr_pos = None
        for i, pkt in enumerate(packets):
            if len(pkt) > 2 and pkt[2] and self._is_pts_valid(pkt[1]):
                idr_pos = i
                break

        if idr_pos is None:
            monitor_logger.debug("SCRUB_NO_IDR chunk пропущен")
            return

        selected = packets[idr_pos:] if full else packets[idr_pos:idr_pos + 1]

        last = None
        for pkt in selected:
            data, pts = pkt[0], pkt[1]
            if not self._is_pts_valid(pts):
                continue
            try:
                filtered = self._decoder.filter_avcc(data)
                if not filtered:
                    continue
                frames = self._decoder.decode_sample(filtered)
                for frame in frames:
                    last = (pts, frame)
            except Exception as e:
                logger.debug(f"Скраб: ошибка декодирования pts={pts}: {e}")

        if last is not None:
            with self._scrub_lock:
                self._scrub_frame = last
            monitor_logger.debug(
                f"SCRUB_FRAME pts={last[0]} full={full} idr_pos={idr_pos}")

    def set_active_window(self, window: IndexWindow):
        start_pts = window.window_start_chunk * FRAMES_PER_CHUNK * SAMPLES_PER_VIDEO_FRAME
        end_pts = (window.window_start_chunk + window.total_chunks) * FRAMES_PER_CHUNK * SAMPLES_PER_VIDEO_FRAME
        with self._buffer_cond:
            self._pts_min = start_pts
            self._pts_max = end_pts
            monitor_logger.info(f"VIDEO_DECODER_WINDOW_SET min_pts={self._pts_min} max_pts={self._pts_max}")

    def set_seek_reference(self, pts: int):
        with self._buffer_cond:
            self._seek_pts_reference = pts
            monitor_logger.info(f"VIDEO_DECODER_SEEK_REF pts={pts}")

    def _is_pts_valid(self, pts: int) -> bool:
        return self._pts_min <= pts < self._pts_max

    def run(self):
        logger.info("VideoDecoderStage запущен")
        while not self.stop_event.is_set():
            try:
                local_chunk, packets = self._input_queue.get(timeout=0.5)
                logger.debug(f"VideoDecoderStage получил чанк {local_chunk}, пакетов: {len(packets)}")
            except queue.Empty:
                continue

            # В режиме скраба идём другим путём: без кольцевого буфера.
            with self._scrub_lock:
                scrubbing = self._scrub_mode
            if scrubbing:
                self._decode_scrub(packets)
                continue

            for data, pts, _is_idr in packets:
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
        with self._buffer_cond:
            self._buffer_cond.notify()

    def set_buffer(self, new_buffer: FrameRingBuffer):
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

        # Диапазон PTS активного окна для аудио
        self._pts_min = 0
        self._pts_max = 0

        # Гистерезис: остановка при 55%, возобновление при 45%
        self._audio_high_watermark = 0.55
        self._audio_low_watermark = 0.45
        self._audio_pause = False

        # Коэффициент ограничения скорости подачи (1.25x)
        self._speed_factor = 1.25

        self._audio_cond = threading.Condition()

        logger.info("AudioDecoderStage инициализирован")

    def set_active_window(self, window: IndexWindow):
        start_pts = window.window_start_chunk * FRAMES_PER_CHUNK * SAMPLES_PER_VIDEO_FRAME
        end_pts = (window.window_start_chunk + window.total_chunks) * FRAMES_PER_CHUNK * SAMPLES_PER_VIDEO_FRAME
        with self._audio_cond:
            self._pts_min = start_pts
            self._pts_max = end_pts
            audio_monitor_logger.info(f"AUDIO_DECODER_WINDOW_SET min_pts={self._pts_min} max_pts={self._pts_max}")

    def _is_pts_valid(self, pts: int) -> bool:
        return self._pts_min <= pts < self._pts_max

    def _check_audio_queue(self) -> bool:
        """Проверяет заполненность аудиоочередей MasterClock."""
        if not self._master_clock:
            return True
        max_samples = getattr(self._master_clock, 'max_audio_queue_samples', 48000)
        current_samples = self._master_clock.get_audio_queue_samples()
        if current_samples >= max_samples * self._audio_high_watermark:
            self._audio_pause = True
            audio_monitor_logger.debug(f"AUDIO_QUEUE_HIGH samples={current_samples}/{max_samples}")
            return False
        elif current_samples <= max_samples * self._audio_low_watermark:
            self._audio_pause = False
            audio_monitor_logger.debug(f"AUDIO_QUEUE_LOW samples={current_samples}/{max_samples}")
        return not self._audio_pause

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

                if not self._is_pts_valid(pts):
                    audio_monitor_logger.debug(f"AUDIO_PACKET_OUTSIDE_WINDOW pts={pts}")
                    continue

                decoder_idx = track_id - 2
                if decoder_idx >= len(self._decoders) or self._decoders[decoder_idx] is None:
                    continue

                # Ожидание освобождения аудиоочереди
                while not self.stop_event.is_set():
                    if self._check_audio_queue():
                        break
                    with self._audio_cond:
                        self._audio_cond.wait(timeout=0.05)
                if self.stop_event.is_set():
                    break

                decoder = self._decoders[decoder_idx]
                try:
                    pcm1 = decoder.decode(d1)
                    pcm2 = decoder.decode(d2)
                    pcm_block = np.concatenate([pcm1, pcm2])
                    logger.debug(f"Аудио track {track_id} декодировано: pts={pts}, сэмплов={len(pcm_block)}")
                except Exception as e:
                    logger.debug(f"Ошибка декодирования аудио track={track_id} pts={pts}: {e}")
                    continue

                if need_fade and len(pcm_block) >= self._fade_len:
                    pcm_block[:self._fade_len] *= np.linspace(0, 1, self._fade_len)

                if self._master_clock:
                    # PTS обязателен: по нему MasterClock выравнивает
                    # дорожки по времени. Без него блоки укладывались бы
                    # подряд, и разрыв в одной дорожке сдвигал бы её
                    # относительно другой безвозвратно.
                    self._master_clock.push_audio(track_id, pcm_block, pts)
                    audio_monitor_logger.debug(f"AUDIO_PUSHED track={track_id} pts={pts} samples={len(pcm_block)}")

                    # Логирование синхронизации
                    clock = self._master_clock.get_audio_clock() if self._master_clock else 0
                    sync_monitor_logger.debug(
                        f"SYNC_AUDIO track={track_id} pts={pts} samples={len(pcm_block)} clock={clock}"
                    )

                    # Регулировка темпа: в зоне 45–55% ограничиваем скорость подачи
                    current_samples = self._master_clock.get_audio_queue_samples()
                    max_samples = getattr(self._master_clock, 'max_audio_queue_samples', 48000)
                    low_threshold = max_samples * self._audio_low_watermark
                    high_threshold = max_samples * self._audio_high_watermark
                    if low_threshold < current_samples < high_threshold:
                        duration = len(pcm_block) / 48000.0
                        time.sleep(duration / self._speed_factor)

    def notify_audio_available(self):
        with self._audio_cond:
            self._audio_cond.notify_all()


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
        self._master_clock = master_clock
        for stage in self._stages:
            if isinstance(stage, AudioDecoderStage):
                stage._master_clock = master_clock

    def set_normal_position(self, current_local_chunk: int, total_local_chunks: int):
        """Возврат планировщика в обычный режим с указанной позиции."""
        self._scheduler.set_normal_mode(current_local_chunk, total_local_chunks)

    def get_window_snapshot(self) -> IndexWindow:
        with self._window_lock:
            return self._window

    def update_window(self, new_window: IndexWindow, start_local_chunk: int = 0):
        """
        Полная смена активного окна (например, после seek в другую часть
        файла): планировщик перезапускается с указанного чанка нового окна.

        start_local_chunk обязателен для seek: раньше метод всегда ставил
        set_normal_mode(0, ...), а вызывающий отдельным вызовом чуть позже
        выставлял правильный чанк. В промежутке между этими двумя вызовами
        ReaderStage (работает непрерывно, независимо от seek) успевал
        прочитать чанк 0 нового окна — то есть заведомо не то место, куда
        метил seek. Теперь позиция передаётся сразу, и set_normal_mode
        вызывается ровно один раз.

        Для роста live-файла внутри уже открытого окна используйте
        extend_window() — он не сбрасывает текущую позицию.
        """
        with self._window_lock:
            self._window = new_window
            self._scheduler.set_normal_mode(start_local_chunk, new_window.total_chunks)
        # Обновляем диапазоны PTS для видео- и аудиостадий
        for stage in self._stages:
            if isinstance(stage, (VideoDecoderStage, AudioDecoderStage)):
                stage.set_active_window(new_window)

    # ------------------------------------------------------------------
    # Публичное состояние и управление планировщиком (этап 1.2)
    #
    # PlaybackEngine обращался к self._pipeline._scheduler напрямую —
    # то есть знал о внутреннем устройстве конвейера. Методы ниже делают
    # эту связь явной и проверяемой контрактным тестом.
    # ------------------------------------------------------------------
    def get_scheduler_state(self) -> dict:
        """Состояние планировщика — делегирует ему же."""
        return self._scheduler.get_state()

    def get_queue_sizes(self) -> dict:
        """Длины очередей между стадиями."""
        return {
            "raw": self._raw_queue.qsize(),
            "video": self._video_queue.qsize(),
            "audio": self._audio_queue.qsize(),
        }

    def get_stage_status(self) -> list:
        """Список стадий с признаком активности."""
        out = []
        for stage in self._stages:
            out.append({
                "name": getattr(stage, "name", "?"),
                "alive": bool(stage.is_alive()) if hasattr(stage, "is_alive") else None,
            })
        return out

    def get_reader_state(self) -> dict:
        """Состояние ридера ReaderStage: размер файла, позиция, статистика."""
        for stage in self._stages:
            reader = getattr(stage, "_reader", None)
            if reader is None:
                continue
            if hasattr(reader, "get_state"):
                return reader.get_state()
            return {}
        return {}

    def set_playback_position(self, local_chunk: int, total_chunks: int = None):
        """
        Переводит планировщик в обычный режим с указанной позиции.

        Заменяет обращение вида pipeline._scheduler.set_normal_mode(...)
        из PlaybackEngine.
        """
        if total_chunks is None:
            window = self.get_window_snapshot()
            total_chunks = window.total_chunks if window else 0
        self._scheduler.set_normal_mode(local_chunk, total_chunks)

    def set_fast_forward(self, direction: int, speed: float,
                         local_chunk: int, total_chunks: int = None):
        """
        Переводит планировщик в режим ускоренной перемотки.

        Заменяет обращение вида
        pipeline._scheduler.set_fast_forward_mode(...) из PlaybackEngine.
        """
        if total_chunks is None:
            window = self.get_window_snapshot()
            total_chunks = window.total_chunks if window else 0
        # Через публичный метод планировщика, а не set_fast_forward_mode:
        # цепочка делегирования должна быть публичной на всех звеньях,
        # иначе вынос интерфейса наверх остаётся половинчатым.
        self._scheduler.set_fast_forward(
            direction=direction, speed=speed,
            current_local_chunk=local_chunk, total_local_chunks=total_chunks,
        )

    def set_scrub_mode(self, enabled: bool, direction: int = 0, speed: float = 1.0,
                       current_local_chunk: int = 0):
        """
        Переводит конвейер в режим ускоренной перемотки (JKL) и обратно.

        Раньше JKL вообще не доходил до конвейера: PlaybackEngine.set_speed()
        менял только свои поля, а планировщик оставался в NORMAL. Из-за
        этого JKL просто вычерпывал уже накопленный видеобуфер (около 14 с),
        после чего картинка замирала, а назад не двигалась вовсе — в буфере
        лежат только кадры ВПЕРЁД от текущей позиции.

        Теперь планировщик переводится в FAST_FORWARD и выдаёт чанки с
        пропуском в нужную сторону (stride по скорости), а декодер
        складывает кадры в отдельную ячейку скраба.
        """
        window = self.get_window_snapshot()
        total = window.total_chunks if window else 0

        for stage in self._stages:
            if isinstance(stage, VideoDecoderStage):
                stage.set_scrub_mode(enabled, speed)

        if enabled:
            self._scheduler.set_fast_forward_mode(
                direction=direction, speed=speed,
                current_local_chunk=current_local_chunk,
                total_local_chunks=total,
            )
        else:
            self._scheduler.set_normal_mode(current_local_chunk, total)

        # Очереди содержат чанки, набранные для другого режима: при входе в
        # скраб это данные обычного воспроизведения, при выходе — разрежённые
        # чанки перемотки. И то и другое дальше только мешает.
        self.flush()
        logger.info("ChunkPipeline: скраб %s (direction=%d, speed=%.1f, chunk=%d)",
                    "включён" if enabled else "выключен",
                    direction, speed, current_local_chunk)

    def get_scrub_frame(self):
        """Последний кадр скраба (pts, frame) или None."""
        for stage in self._stages:
            if isinstance(stage, VideoDecoderStage):
                return stage.take_scrub_frame()
        return None

    def notify_file_grew(self):
        """
        Сигнал "исходный файл вырос" — прокидывается в ReaderStage, чтобы тот
        обновил размер файла у своего ридера.

        Вызывается из PlaybackEngine после того, как LazyIndex подтвердил
        появление новых записей в индексе (см. _refresh_growth_async). Сам
        LazyIndex этот метод не дёргает: он ничего не знает о конвейере и не
        должен — обновление собственных границ (mdat_end) он делает у себя, а
        доставку сигнала в конвейер обеспечивает владелец обоих компонентов.
        """
        for stage in self._stages:
            if isinstance(stage, ReaderStage):
                stage.refresh_file_size()

    def extend_window(self, new_window: IndexWindow):
        """
        Расширяет активное окно вперёд при росте live-файла.

        В отличие от update_window(), НЕ сбрасывает текущую позицию
        чтения/декодирования — только поднимает верхнюю границу
        total_chunks у планировщика и PTS-диапазоны видео/аудио стадий.
        Предполагается, что new_window.window_start_frame совпадает со
        start_frame текущего окна (ровно то, что делает
        LazyIndex.expand_window()/refresh_from_disk()).
        """
        with self._window_lock:
            self._window = new_window
            self._scheduler.extend_total_chunks(new_window.total_chunks)
        for stage in self._stages:
            if isinstance(stage, (VideoDecoderStage, AudioDecoderStage)):
                stage.set_active_window(new_window)

    def shift_window(self, new_window: IndexWindow):
        """
        Плавное переключение на скользящее окно, построенное
        LazyIndex.build_slid_window() (оба края окна сдвинуты вперёд,
        в отличие от extend_window(), где сдвигается только конец).

        В отличие от update_window(), НЕ сбрасывает текущую позицию
        чтения/декодирования — пересчитывает её в новой локальной системе
        координат через scheduler.shift_loaded(), сохраняя уже загруженные
        чанки, которые попадают в новое окно.

        Порядок вызова важен для отсутствия гонки с очередями конвейера:
        1) new_window должен быть построен с overlap, покрывающим глубину
           очередей (см. DEFAULT_SLIDE_OVERLAP_CHUNKS в lazy_index.py) —
           это гарантирует, что пакеты, уже лежащие в raw_queue/video_queue/
           audio_queue со старыми глобальными PTS на момент вызова, всё
           ещё попадают в диапазон [pts_min, pts_max) нового окна и не
           будут отброшены PTS-фильтром сразу после свопа.
        2) Вызывающий (PlaybackEngine) обязан вызвать
           LazyIndex.commit_window(new_window) ТОЛЬКО после успешного
           возврата из этого метода — так self._window в LazyIndex и
           активное окно в ChunkPipeline не могут разойтись.
        """
        if new_window is None:
            logger.error("shift_window: new_window is None")
            return
        try:
            with self._window_lock:
                old_window = self._window
                chunk_shift = new_window.window_start_chunk - old_window.window_start_chunk
                self._window = new_window
                self._scheduler.shift_loaded(chunk_shift, new_window.total_chunks)
            # Обновляем диапазоны PTS для видео- и аудиостадий — благодаря
            # overlap в new_window новый pts_min не может быть больше PTS
            # пакетов, уже поставленных в очередь до переключения.
            for stage in self._stages:
                if isinstance(stage, (VideoDecoderStage, AudioDecoderStage)):
                    stage.set_active_window(new_window)
            logger.info(
                "ChunkPipeline: окно сдвинуто (chunk_shift=%d), новые кадры %d-%d",
                chunk_shift, new_window.window_start_frame, new_window.window_end_frame,
            )
        except Exception as e:
            logger.exception(f"Ошибка в shift_window: {e}")    

    def set_video_buffer(self, new_buffer: FrameRingBuffer):
        self._video_buffer = new_buffer
        self._scheduler.set_buffer(new_buffer)
        for stage in self._stages:
            if isinstance(stage, VideoDecoderStage):
                stage.set_buffer(new_buffer)

    def set_seek_reference(self, pts: int):
        for stage in self._stages:
            if isinstance(stage, VideoDecoderStage):
                stage.set_seek_reference(pts)

    def flush(self):
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
                               self._video_queue,
                               self._audio_queue,
                               self._scheduler)
        video_dec = VideoDecoderStage(self._video_decoder, self._video_buffer,
                                      self._video_queue)
        audio_dec = AudioDecoderStage(self._audio_decoders, self._master_clock,
                                      self._audio_queue)

        # Устанавливаем окно до запуска
        video_dec.set_active_window(self.get_window_snapshot())
        audio_dec.set_active_window(self.get_window_snapshot())
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
            elif isinstance(stage, AudioDecoderStage):
                stage.notify_audio_available()
        for stage in self._stages:
            stage.join(timeout=2.0)
        self._stages.clear()

        self.flush()
        logger.info("ChunkPipeline: все стадии остановлены, очереди очищены")
