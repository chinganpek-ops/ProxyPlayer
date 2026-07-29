"""
chunk_reader.py – загрузчик чанков (статический режим, v5-совместимый).
Работает с готовыми video_records, chunk_offsets, chunk_sizes, audio_chunks.
Встроен отладочный коллбэк для мониторинга (UDP).
Добавлены флаги и методы для синхронизации с потребителем кадров.
"""

import time
import threading
import logging
from collections import deque
from pathlib import Path
from typing import Optional
import numpy as np

from frame_buffer import FrameRingBuffer
from audio_buffer import MultiTrackAudioBuffer
from decoder import Decoder as VideoDecoder
from audio_decoder import AudioDecoder
from moov_builder import SEGMENT_SIZE
from win_sequential_reader import WinSequentialReader
from utils import get_real_size

logger = logging.getLogger(__name__)

SAMPLES_PER_AAC_FRAME = 1024
SAMPLES_PER_C9 = SAMPLES_PER_AAC_FRAME * 2
SAMPLES_PER_VIDEO_FRAME = 1920
FRAMES_PER_CHUNK = 12
MAX_CONSECUTIVE_AAC_ERRORS = 5
FADE_IN_LEN = 8

MAX_CONCURRENT_DECODES = 3
_decode_semaphore = threading.Semaphore(MAX_CONCURRENT_DECODES)


class TokenBucket:
    def __init__(self, rate, capacity):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill = time.monotonic()
        self.lock = threading.Lock()

    def consume(self, n):
        with self.lock:
            now = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + (now - self.last_refill) * self.rate)
            self.last_refill = now
            if self.tokens >= n:
                self.tokens -= n
                return True
            return False


class ChunkReader:
    def __init__(self, mp4_path: Path, video_records, chunk_offsets, chunk_sizes,
                 audio_chunks, video_buffer, audio_buffers: MultiTrackAudioBuffer,
                 video_decoder, audio_decoders,
                 asc=b'\x11\x88', fps=25.0, group_chunks=20, max_retries=5,
                 free_slots_required=25, normal_playback_speed_factor=1.25,
                 rate_limit=0, bitrate=0.0, mdat_end=0,
                 mm_c9=None, video_records_mmap=None,
                 dynamic_chunks=False, mm_193=None,
                 use_mmap_reader=False,
                 debug_callback=None):
        self.mp4_path = mp4_path
        self.video_records = video_records
        self.chunk_offsets = chunk_offsets
        self.chunk_sizes = chunk_sizes
        self.audio_chunks = audio_chunks
        self.video_buffer = video_buffer
        self.audio_buffers = audio_buffers
        self.video_decoder = video_decoder
        self.audio_decoders = audio_decoders
        self.asc = asc
        self.fps = fps
        self.group_chunks = group_chunks
        self.max_retries = max_retries
        self.free_slots_required = free_slots_required
        self.normal_playback_speed_factor = normal_playback_speed_factor
        self.bitrate = bitrate
        self.rate_limit = rate_limit
        self.mdat_end = mdat_end
        self._mdat_end = mdat_end
        self._last_size_check = 0.0
        self.dynamic_chunks = False
        self.mm_193 = None
        self.mm_c9 = None
        self.use_mmap_reader = use_mmap_reader

        self._reader = WinSequentialReader(mp4_path, 0, False)
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._next_chunk = 0
        self._progressive_chunks = 1
        self._soft_reset_requested = False
        self._soft_reset_chunk = 0
        self._token_bucket = TokenBucket(rate_limit, rate_limit*3) if rate_limit > 0 else None
        self._download_times = deque(maxlen=5)
        self._current_speed = 0.0

        # === Синхронизация с потребителем ===
        self._audio_write_enabled = True                 # запись аудио разрешена
        self._video_write_suspended = threading.Event()  # приостановка загрузки чанков
        self._video_write_suspended.set()                # изначально загрузка разрешена

        self._audio_error_counters = [0] * len(audio_decoders)
        self.dec2 = AudioDecoder(asc)
        self.dec3 = AudioDecoder(asc)
        self._err_count_2 = 0
        self._err_count_3 = 0

        # Отладочный коллбэк
        self._debug_cb = debug_callback

    # ============ Новые методы синхронизации ============
    def pause_loading(self):
        """Приостановить загрузку чанков (когда видеобуфер близок к переполнению)."""
        self._video_write_suspended.clear()
        logger.debug("Загрузка чанков приостановлена")

    def resume_loading(self):
        """Возобновить загрузку чанков."""
        self._video_write_suspended.set()
        logger.debug("Загрузка чанков возобновлена")

    def disable_audio_write(self):
        """Временно запретить запись аудио (например, во время перемотки)."""
        self._audio_write_enabled = False
        logger.debug("Запись аудио в чанк-ридере приостановлена")

    def enable_audio_write(self):
        """Снова разрешить запись аудио."""
        self._audio_write_enabled = True
        logger.debug("Запись аудио в чанк-ридере возобновлена")

    def reset_audio_errors(self):
        """Сброс счётчиков ошибок аудиодекодеров."""
        self._err_count_2 = 0
        self._err_count_3 = 0
        self._audio_error_counters = [0] * len(self._audio_error_counters)

    # ============ Существующие методы (с правками) ============
    def _ensure_decoder(self, idx):
        if self._audio_error_counters[idx] >= MAX_CONSECUTIVE_AAC_ERRORS:
            self.audio_decoders[idx] = AudioDecoder(self.asc)
            self._audio_error_counters[idx] = 0
            logger.info("Пересоздан декодер дорожки %d после %d ошибок", idx + 2, MAX_CONSECUTIVE_AAC_ERRORS)
        if self.audio_decoders[idx] is None:
            self.audio_decoders[idx] = AudioDecoder(self.asc)
        return self.audio_decoders[idx]

    def _decode_aac_frame(self, decoder, data, track_id=None):
        if not data:
            self._increase_error(track_id)
            return np.zeros(SAMPLES_PER_AAC_FRAME, dtype=np.float64), False

        try:
            pcm = decoder.decode(data)
            if pcm is not None and not np.all(pcm == 0):
                self._reset_errors(track_id)
                return pcm, False
        except Exception:
            pass

        self._increase_error(track_id)
        need_fade = False
        if track_id == 2 and self._err_count_2 >= MAX_CONSECUTIVE_AAC_ERRORS:
            self.dec2 = AudioDecoder(self.asc)
            self._err_count_2 = 0
            need_fade = True
            logger.info("Пересоздан декодер дорожки 2")
        elif track_id == 3 and self._err_count_3 >= MAX_CONSECUTIVE_AAC_ERRORS:
            self.dec3 = AudioDecoder(self.asc)
            self._err_count_3 = 0
            need_fade = True
            logger.info("Пересоздан декодер дорожки 3")
        elif track_id is not None and track_id >= 4:
            idx = track_id - 2
            if self._audio_error_counters[idx] >= MAX_CONSECUTIVE_AAC_ERRORS:
                self.audio_decoders[idx] = AudioDecoder(self.asc)
                self._audio_error_counters[idx] = 0
                need_fade = True
                logger.info("Пересоздан декодер дорожки %d", track_id)

        return np.zeros(SAMPLES_PER_AAC_FRAME, dtype=np.float64), need_fade

    def _increase_error(self, track_id):
        if track_id == 2:
            self._err_count_2 += 1
        elif track_id == 3:
            self._err_count_3 += 1
        elif track_id is not None and track_id >= 4:
            idx = track_id - 2
            if idx < len(self._audio_error_counters):
                self._audio_error_counters[idx] += 1

    def _reset_errors(self, track_id):
        if track_id == 2:
            self._err_count_2 = 0
        elif track_id == 3:
            self._err_count_3 = 0
        elif track_id is not None and track_id >= 4:
            idx = track_id - 2
            if idx < len(self._audio_error_counters):
                self._audio_error_counters[idx] = 0

    def start(self, start_chunk=0):
        self.stop()
        self._stop_event.clear()
        self._next_chunk = start_chunk
        self._progressive_chunks = 1
        self.reset_audio_errors()   # сброс ошибок при старте
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
        if self._debug_cb:
            self._debug_cb("CHUNK_START", f"chunk={start_chunk}")

    def stop(self):
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        self._worker_thread = None
        if self._debug_cb:
            self._debug_cb("CHUNK_STOP", "")

    def soft_reset(self, start_chunk):
        with self._lock:
            self._soft_reset_requested = True
            self._soft_reset_chunk = start_chunk
            self._progressive_chunks = 1
            self.reset_audio_errors()   # сброс ошибок при мягком сбросе

    def flush_audio_queue(self):
        """Очищает очередь аудиоданных и сбрасывает прогрессивную загрузку (используется при seek)."""
        with self._lock:
            self._progressive_chunks = 1
        logger.debug("Очередь аудиоданных очищена")

    def update_data(self, vr, co, cs, ac, mdat_end=None):
        with self._lock:
            self.video_records = vr
            self.chunk_offsets = co
            self.chunk_sizes = cs
            self.audio_chunks = ac
            if mdat_end is not None:
                self._mdat_end = mdat_end

    def _worker(self):
        while not self._stop_event.is_set():
            # Ждём разрешения на загрузку (видеобуфер не переполнен)
            self._video_write_suspended.wait()

            with self._lock:
                if self._soft_reset_requested:
                    self.video_buffer.clear()
                    # audio_buffers НЕ очищаем, чтобы не разрушать синхронизацию
                    self._next_chunk = self._soft_reset_chunk
                    self._soft_reset_requested = False
                    self._progressive_chunks = 1

            if self.video_buffer.free_slots < self.free_slots_required:
                time.sleep(0.05)
                continue

            with self._lock:
                remaining = len(self.chunk_offsets) - self._next_chunk
                if remaining <= 0:
                    time.sleep(0.05)
                    continue

                near_end = remaining < 3
                chunks_to_load = min(self._progressive_chunks, 2) if near_end else min(self._progressive_chunks, self.group_chunks)

                start = self._next_chunk
                end = start + chunks_to_load

            for c in range(start, end):
                if self._stop_event.is_set():
                    break
                self._load_and_decode_chunk(c)

            with self._lock:
                self._next_chunk = end
            if self._progressive_chunks < self.group_chunks and not near_end:
                self._progressive_chunks = min(self._progressive_chunks * 2, self.group_chunks)

    def _get_chunk_offsets(self, chunk_idx):
        if chunk_idx >= len(self.chunk_offsets):
            return 0, 0
        return int(self.chunk_offsets[chunk_idx]), int(self.chunk_sizes[chunk_idx])

    def _load_and_decode_chunk(self, chunk_idx):
        try:
            self._load_and_decode_chunk_impl(chunk_idx)
        except Exception as e:
            logger.exception("Ошибка чанка %d: %s", chunk_idx, e)
            if self._debug_cb:
                self._debug_cb("CHUNK_ERROR", f"chunk={chunk_idx} error={e}")
            with self._lock:
                if chunk_idx >= self._next_chunk:
                    self._next_chunk = chunk_idx + 1

    def _load_and_decode_chunk_impl(self, chunk_idx):
        vs, chunk_size = self._get_chunk_offsets(chunk_idx)
        if chunk_size <= 0:
            return
        ve = vs + chunk_size

        ch = self.audio_chunks[chunk_idx] if chunk_idx < len(self.audio_chunks) else {}

        ab_start = ve
        ab_end = ve
        if ch:
            offs = []
            for lst in ch.values():
                for e in lst:
                    offs.append(e['abs_offset'])
                    offs.append(e['abs_offset'] + e['size1'] + e.get('size2', 0))
            if offs:
                ab_start = min(offs)
                ab_end = max(offs)

        read_start = min(vs, ab_start)
        read_end = max(ve, ab_end) + 256

        if chunk_idx == len(self.chunk_offsets) - 1:
            now = time.monotonic()
            if now - self._last_size_check > 1.0:
                new_size = get_real_size(str(self.mp4_path))
                if new_size > 0:
                    self._mdat_end = new_size
                self._last_size_check = now
        read_end = min(read_end, self._mdat_end)

        read_size = read_end - read_start
        if read_size <= 0:
            return

        if self._token_bucket:
            while not self._token_bucket.consume(read_size) and not self._stop_event.is_set():
                time.sleep(0.05)
            if self._stop_event.is_set():
                return

        data = None
        for _ in range(self.max_retries):
            if self._stop_event.is_set():
                return
            try:
                t0 = time.monotonic()
                data = self._reader.read_sequential(read_start, read_size)
                if data:
                    self._download_times.append((time.monotonic()-t0, len(data)))
                break
            except Exception as e:
                logger.warning("Чтение чанка %d: %s", chunk_idx, e)
                time.sleep(0.5)
        if not data:
            return

        if len(self._download_times) >= 2:
            avg_t = sum(t for t,_ in self._download_times) / len(self._download_times)
            avg_b = sum(b for _,b in self._download_times) / len(self._download_times)
            self._current_speed = avg_b / avg_t if avg_t > 0 else 0.0

        _decode_semaphore.acquire()
        try:
            self._decode_chunk_data(chunk_idx, data, vs, chunk_size, read_start, ch)
        finally:
            _decode_semaphore.release()

        if self._debug_cb:
            self._debug_cb("CHUNK_DECODED", f"chunk={chunk_idx}")

    def _decode_chunk_data(self, chunk_idx, data, video_start, chunk_size, read_start, audio_data):
        video_recs = self.video_records
        start_frame = chunk_idx * FRAMES_PER_CHUNK
        for i in range(FRAMES_PER_CHUNK):
            abs_idx = start_frame + i
            if abs_idx >= len(video_recs):
                break
            rec = video_recs[abs_idx]
            off = int(rec['f1']) - 4 + int(rec['f2']) * SEGMENT_SIZE
            rel = off - video_start
            if rel < 0 or rel >= len(data):
                continue
            if i < FRAMES_PER_CHUNK - 1 and (abs_idx + 1) < len(video_recs):
                next_off = int(video_recs[abs_idx+1]['f1']) - 4 + int(video_recs[abs_idx+1]['f2']) * SEGMENT_SIZE
                size = next_off - off
            else:
                size = (video_start + chunk_size) - off
            size = max(0, min(size, len(data) - rel))
            if size <= 0:
                continue
            sample = data[rel:rel+size]
            filtered = self.video_decoder.filter_avcc(sample)
            if not filtered:
                continue
            try:
                frames = self.video_decoder.decode_sample(filtered)
            except Exception as e:
                logger.error("Ошибка видео: %s", e)
                if self._debug_cb:
                    self._debug_cb("VIDEO_DECODE_ERROR", f"frame={abs_idx} error={e}")
                continue
            for frame in frames:
                pts = abs_idx * SAMPLES_PER_VIDEO_FRAME
                while not self._stop_event.is_set():
                    if self.video_buffer.try_push(frame, pts):
                        break
                    time.sleep(0.01)

        # Обрабатываем аудио, только если запись разрешена
        if audio_data and self._audio_write_enabled:
            for track_id, entries in audio_data.items():
                if track_id == 0 or track_id > 3:
                    continue

                if track_id == 2:
                    decoder = self.dec2
                elif track_id == 3:
                    decoder = self.dec3
                else:
                    dec_idx = track_id - 2
                    if dec_idx < 0 or dec_idx >= len(self.audio_decoders):
                        continue
                    decoder = self._ensure_decoder(dec_idx)

                for entry in entries:
                    size2 = entry.get('size2', 0)
                    rel = entry['abs_offset'] - read_start

                    d1 = b''
                    if 0 <= rel and rel + entry['size1'] <= len(data):
                        d1 = data[rel:rel+entry['size1']]
                    pcm1, need_fade1 = self._decode_aac_frame(decoder, d1, track_id)

                    d2 = b''
                    if size2 > 0:
                        rel2 = rel + entry['size1']
                        if rel2 >= 0 and rel2 + size2 <= len(data):
                            d2 = data[rel2:rel2+size2]
                    pcm2, need_fade2 = self._decode_aac_frame(decoder, d2, track_id)

                    pcm_block = np.concatenate([pcm1, pcm2])

                    # Проверка на тишину в декодированном PCM
                    if self._debug_cb and np.max(np.abs(pcm_block)) < 1e-10:
                        self._debug_cb("AUDIO_SILENCE_DECODED", f"track={track_id} pts={entry['pts']}")

                    if need_fade1 or need_fade2:
                        if FADE_IN_LEN > 0 and len(pcm_block) >= FADE_IN_LEN:
                            pcm_block[:FADE_IN_LEN] *= np.linspace(0, 1, FADE_IN_LEN, dtype=np.float64)
                        if self._debug_cb:
                            self._debug_cb("AUDIO_DECODER_RECREATED", f"track={track_id}")

                    if not self.audio_buffers.try_write(track_id, pcm_block, entry['pts']):
                        if self._debug_cb:
                            self._debug_cb("AUDIO_WRITE_FAILED", f"track={track_id} pts={entry['pts']} samples={len(pcm_block)}")
                    else:
                        if self._debug_cb:
                            self._debug_cb("AUDIO_WRITE", f"track={track_id} pts={entry['pts']} samples={len(pcm_block)}")

    @property
    def network_speed(self):
        return self._current_speed

    def close(self):
        self.stop()
        for dec in self.audio_decoders:
            if dec is not None:
                try:
                    dec.codec.decode(None)
                except Exception:
                    pass
        if self._reader:
            self._reader.close()