"""
master_clock.py – единый тактовый генератор на основе sounddevice.
Одна звуковая карта = Master Clock + вывод аудио.
Поддерживает две моно-дорожки (2 → левый канал, 3 → правый канал).
Управление: включение/отключение дорожек, общий mute.
Добавлено управление заполненностью аудиоочередей:
- max_audio_queue_samples ограничивает суммарный размер буферов.
- get_audio_queue_samples() возвращает текущее количество сэмплов в очередях.
- flush_audio() очищает очереди без сброса тактового счётчика (для seek).
Логирование аудио-событий в audio_monitor.log через AudioMonitor.

ИЗМЕНЕНИЯ (продакшен-ревью, диагностика "нет звука при старте"):
- push_audio(): убрана проверка `if not self._active: return`. Она
  отбрасывала ВСЁ декодированное аудио, поступавшее до вызова start()
  (который происходит только при нажатии Play в resume()) — а конвейер
  (ReaderStage/AudioDecoderStage) запускается заметно раньше, во время
  начальной буферизации в PlaybackEngine.start_playback(). Поскольку
  выброшенные сэмплы не увеличивали get_audio_queue_samples(), гистерезис
  AudioDecoderStage (45%/55%) не видел реального заполнения и не
  тормозил декодирование аудио — в отличие от видео, которое корректно
  тормозится своим буфером (buffer_size кадров). Когда видеобуфер
  заполнялся, общий для видео и аудио StreamScheduler останавливал
  чтение чанков целиком — то есть к моменту нажатия Play звук для уже
  прочитанного видео (до ~buffer_size кадров вперёд) был безвозвратно
  потерян, а новое чтение возобновлялось только после того, как
  воспроизведение растратит буфер ниже 60% — то есть реальный, иногда
  многосекундный провал звука в начале.
  Теперь push_audio() всегда кладёт сэмплы в очередь независимо от
  self._active — вывод (потребление очереди) по-прежнему происходит
  только в _callback(), который запускается лишь пока поток реально
  активен (sounddevice сам не дёргает callback, пока stream не
  запущен/после stop()), так что тишина до нажатия Play сохраняется
  ровно как раньше — просто теперь за счёт того, что поток не запущен
  и никто не потребляет очередь, а не за счёт отбрасывания данных на
  входе. Гистерезис в AudioDecoderStage теперь получает реальные цифры
  уже во время начальной буферизации и тормозит декодирование аудио
  симметрично видео — как и было задумано.
"""

import threading
import logging
from collections import deque
import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

# --- Логгер для мониторинга аудио ---
audio_monitor_logger = logging.getLogger("AudioMonitor")
audio_monitor_logger.setLevel(logging.DEBUG)
if not audio_monitor_logger.handlers:
    _audio_mon_handler = logging.FileHandler("audio_monitor.log", encoding="utf-8")
    _audio_mon_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    audio_monitor_logger.addHandler(_audio_mon_handler)
audio_monitor_logger.propagate = False


class MasterClock:
    def __init__(self, sample_rate: int = 48000, buffer_size: int = 1024,
                 max_audio_queue_samples: int = 48000):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size

        # Максимальное количество аудиосэмплов в очередях (гистерезис)
        self.max_audio_queue_samples = max_audio_queue_samples

        # Счётчик воспроизведённых семплов (монотонный)
        self._samples_played = 0
        self._clock_lock = threading.Lock()

        # Раздельные очереди для дорожек 2 и 3 (float32)
        self._queue2 = deque()
        self._queue3 = deque()
        self._queue_lock = threading.Lock()

        # Управление дорожками
        self._track2_enabled = True   # дорожка 2 (левый канал)
        self._track3_enabled = True   # дорожка 3 (правый канал)
        self._muted = False           # общий mute

        # Задержка аудио относительно видео (в секундах)
        self.audio_delay = 0.0

        # Поток и управление
        self._stream: sd.OutputStream = None
        self._active = False

        # Статистика
        self._underruns = 0
        self._max_queue_len = 0

        # Предел опережения блока относительно текущего времени (сэмплы).
        # Блок дальше этого считается устаревшим и выбрасывается — см.
        # _mix_channel. Секунда с запасом отделяет настоящие разрывы
        # записи (десятки миллисекунд) от данных со старой позиции.
        self._max_future_lead = sample_rate * 1
        self._stale_dropped = 0

        # Статистика качества подаваемого звука по дорожкам.
        #
        # Собирается здесь, а не внешним наблюдателем, потому что блок PCM
        # живёт от декодера до очереди и нигде больше не сохраняется:
        # измерить его снаружи можно было только подменой push_audio, а
        # такая подмена ломается молча при любом изменении сигнатуры.
        #
        # Что именно считается и зачем:
        #   peak        — амплитуда вне диапазона -1..+1 означает, что
        #                 декодер отдаёт целочисленные сэмплы, а поток
        #                 ожидает float32: слышно как громкий треск;
        #   clipped     — доля сэмплов за пределами диапазона;
        #   non_finite  — NaN/inf от повреждённых пакетов, в звуке щелчки;
        #   min/max_block — разброс размера блока выдаёт обрезанные пакеты.
        self._track_stats = {
            t: {"blocks": 0, "samples": 0, "dtype": None, "peak": 0.0,
                "clipped": 0, "non_finite": 0, "rms_sum": 0.0, "rms_n": 0,
                "min_block": None, "max_block": 0}
            for t in (2, 3)
        }

        # Флаг доступности устройства
        self._device_available = True

    # ------------------------------------------------------------------
    # Время
    # ------------------------------------------------------------------
    @property
    def samples_played(self) -> int:
        """Общее количество воспроизведённых семплов."""
        with self._clock_lock:
            return self._samples_played

    def get_time(self) -> float:
        """Текущее время в секундах от начала воспроизведения."""
        return self.samples_played / self.sample_rate

    def get_audio_clock(self) -> int:
        """Возвращает текущий audio_clock (в семплах) для синхронизации видео."""
        return self.samples_played + int(self.audio_delay * self.sample_rate)

    def set_clock(self, pts: int):
        """Принудительно устанавливает счётчик (при перемотке)."""
        with self._clock_lock:
            self._samples_played = pts
        logger.debug("MasterClock счётчик установлен на %d", pts)

    def reset(self):
        """Сбрасывает счётчик и очищает очереди (при остановке/перемотке)."""
        self.set_clock(0)
        self.flush_audio()
        self._underruns = 0
        logger.debug("MasterClock сброшен")

    # ------------------------------------------------------------------
    # Управление дорожками
    # ------------------------------------------------------------------
    def set_track_enabled(self, track_id: int, enabled: bool):
        """Включает или отключает дорожку (2 или 3)."""
        with self._queue_lock:
            if track_id == 2:
                self._track2_enabled = enabled
                if not enabled:
                    self._queue2.clear()
            elif track_id == 3:
                self._track3_enabled = enabled
                if not enabled:
                    self._queue3.clear()
        logger.debug("Дорожка %d %s", track_id, "включена" if enabled else "отключена")

    def set_muted(self, muted: bool):
        """Общий mute для всех дорожек."""
        self._muted = muted
        logger.debug("Mute: %s", "включен" if muted else "выключен")

    def is_track_enabled(self, track_id: int) -> bool:
        """Возвращает True, если дорожка включена."""
        if track_id == 2:
            return self._track2_enabled
        elif track_id == 3:
            return self._track3_enabled
        return False

    # ------------------------------------------------------------------
    # Очереди аудио
    # ------------------------------------------------------------------
    def push_audio(self, track_id: int, samples: np.ndarray, pts: int = None):
        """
        Добавляет декодированные аудиосемплы (float64, моно) в очередь дорожки.
        Вызывается из AudioDecoderStage.

        ПРАВКА: раньше здесь был ранний выход `if not self._active: return`,
        из-за которого всё аудио, декодированное до первого вызова start()
        (то есть до нажатия Play), терялось молча — см. докстринг модуля.
        Очередь принимает данные всегда; воспроизведение (потребление
        очереди) всё равно происходит только в _callback(), который
        sounddevice вызывает исключительно пока поток запущен — тишина до
        нажатия Play сохраняется, просто не ценой потери данных на входе.
        """
        block = samples.astype(np.float32)
        self._collect_track_stats(track_id, samples, block)

        # PTS блока. Если вызывающий его не передал, продолжаем прежнюю
        # последовательную укладку: блок помечается временем, следующим за
        # концом уже накопленного. Так старый вызов остаётся рабочим, а
        # новый получает точное позиционирование.
        with self._queue_lock:
            queue = self._queue2 if track_id == 2 else self._queue3
            enabled = self._track2_enabled if track_id == 2 else self._track3_enabled
            if track_id not in (2, 3) or not enabled:
                return

            if pts is None:
                pts = (queue[-1][0] + len(queue[-1][1])) if queue else self.samples_played

            queue.append((int(pts), block))
            self._max_queue_len = max(self._max_queue_len,
                                      len(self._queue2) + len(self._queue3))

        # Логирование в аудио-монитор
        audio_monitor_logger.debug(f"AUDIO_PUSH track={track_id} samples={len(samples)}")

    def get_queue_size(self) -> int:
        """Возвращает количество элементов (блоков) в очередях."""
        with self._queue_lock:
            return len(self._queue2) + len(self._queue3)

    def get_audio_queue_samples(self) -> int:
        """Возвращает суммарное количество аудиосэмплов во всех очередях."""
        with self._queue_lock:
            total = sum(len(item[1]) for item in self._queue2)
            total += sum(len(item[1]) for item in self._queue3)
            return total

    def _collect_track_stats(self, track_id: int, raw, block) -> None:
        """
        Измеряет качество блока PCM. Вызывается из push_audio до укладки
        в очередь. Ошибки подавляются: диагностика не должна мешать
        воспроизведению.
        """
        st = self._track_stats.get(track_id)
        if st is None:
            return
        try:
            n = int(block.size)
            st["blocks"] += 1
            st["samples"] += n
            st["dtype"] = str(getattr(raw, "dtype", ""))
            if not n:
                return
            st["min_block"] = n if st["min_block"] is None else min(st["min_block"], n)
            st["max_block"] = max(st["max_block"], n)

            finite = np.isfinite(block)
            bad = int(n - int(finite.sum()))
            st["non_finite"] += bad
            if bad >= n:
                return
            vals = block[finite]
            peak = float(np.abs(vals).max())
            if peak > st["peak"]:
                st["peak"] = peak
            st["clipped"] += int((np.abs(vals) > 1.0).sum())
            st["rms_sum"] += float(np.sqrt(np.mean(vals.astype(np.float64) ** 2)))
            st["rms_n"] += 1
        except Exception:
            pass

    def get_underruns(self) -> int:
        """
        Число случаев, когда данных не хватило на очередной вызов вывода.

        Отдельный метод, а не поле _underruns: это самая частая метрика в
        инструментах замера, и запрашивать ради неё полный get_stats()
        (который трогает состояние устройства и очередей) избыточно.
        """
        return self._underruns

    def get_track_quality(self, track_id: int) -> dict:
        """
        Качество подаваемого звука по дорожке: амплитуда, клиппинг,
        NaN/inf, разброс размера блока.

        Заменяет подмену push_audio внешним наблюдателем.
        """
        st = self._track_stats.get(track_id)
        if st is None:
            return {}
        out = {
            "blocks": st["blocks"],
            "samples": st["samples"],
            "dtype": st["dtype"],
            "peak": round(st["peak"], 4),
            "clipped": st["clipped"],
            "non_finite": st["non_finite"],
            "min_block": st["min_block"],
            "max_block": st["max_block"],
        }
        if st["rms_n"]:
            out["rms_avg"] = round(st["rms_sum"] / st["rms_n"], 5)
        return out

    def get_audio_diagnostics(self) -> dict:
        """
        Полная диагностика звука: качество и очереди по обеим дорожкам
        плюс расхождение между ними.

        Расхождение подачи — прямой признак того, что дорожки разъезжаются:
        именно по нему обнаружилось расхождение в 3.4 секунды до перехода
        на выравнивание по PTS.
        """
        out = {"tracks": {}, "underruns": self._underruns,
               "stale_dropped": self._stale_dropped}
        pushed = {}
        for track_id in (2, 3):
            q = self.get_track_state(track_id)
            q.update(self.get_track_quality(track_id))
            out["tracks"][str(track_id)] = q
            pushed[track_id] = q.get("samples", 0)

        out["pushed_drift_samples"] = pushed.get(2, 0) - pushed.get(3, 0)
        q2 = self.get_track_queue_samples(2)
        q3 = self.get_track_queue_samples(3)
        out["queue_drift_samples"] = q2 - q3
        out["queue_drift_ms"] = round((q2 - q3) / (self.sample_rate / 1000.0), 1)
        return out

    def get_track_queue_samples(self, track_id: int) -> int:
        """Сэмплы в очереди ОДНОЙ дорожки — для диагностики расхождения."""
        with self._queue_lock:
            queue = self._queue2 if track_id == 2 else self._queue3
            return sum(len(item[1]) for item in queue)

    def get_track_state(self, track_id: int) -> dict:
        """
        Состояние одной дорожки: блоки, сэмплы, диапазон PTS, включена ли.

        Заменяет чтение _queue2/_queue3 телеметрией и инструментами
        замера. Диапазон PTS полезен для диагностики: по нему видно, какой
        участок звука лежит в очереди относительно текущих часов.
        """
        with self._queue_lock:
            queue = self._queue2 if track_id == 2 else self._queue3
            enabled = self._track2_enabled if track_id == 2 else self._track3_enabled
            blocks = len(queue)
            samples = sum(len(item[1]) for item in queue)
            first_pts = int(queue[0][0]) if blocks else None
            last_pts = int(queue[-1][0] + len(queue[-1][1])) if blocks else None
        return {
            "track": track_id,
            "enabled": enabled,
            "blocks": blocks,
            "samples": samples,
            "first_pts": first_pts,
            "last_pts": last_pts,
        }

    def get_audio_state(self) -> dict:
        """Сводное состояние звука: часы, дорожки, underrun, mute."""
        return {
            "clock": self.get_audio_clock(),
            "samples_played": self.samples_played,
            "underruns": self._underruns,
            "muted": self._muted,
            "active": self._active,
            "device_available": self._device_available,
            "max_queue_samples": self.max_audio_queue_samples,
            "tracks": {
                "2": self.get_track_state(2),
                "3": self.get_track_state(3),
            },
        }

    def get_tracks_state(self) -> dict:
        """Состояние обеих дорожек и их расхождение в сэмплах."""
        t2 = self.get_track_state(2)
        t3 = self.get_track_state(3)
        return {
            "2": t2,
            "3": t3,
            "drift_samples": t2["samples"] - t3["samples"],
            "underruns": self._underruns,
            "muted": self._muted,
        }

    def flush_audio(self):
        """
        Очищает аудиоочереди, не меняя счётчик воспроизведённых семплов.
        Используется после seek для удаления старых аудиоданных.
        """
        with self._queue_lock:
            self._queue2.clear()
            self._queue3.clear()
        audio_monitor_logger.info("AUDIO_FLUSH")
        logger.debug("MasterClock: аудиоочереди очищены (flush_audio)")

    # ------------------------------------------------------------------
    # Callback звуковой карты
    # ------------------------------------------------------------------
    def _callback(self, outdata: np.ndarray, frames: int, time_info, status):
        if status:
            logger.debug("Sounddevice status: %s", status)

        with self._clock_lock:
            block_start_pts = self._samples_played      # время начала блока
            self._samples_played += frames

        outdata.fill(0.0)
        if self._muted:
            return

        # Обе дорожки выравниваются по ОДНОМУ И ТОМУ ЖЕ времени, поэтому
        # разойтись не могут даже при неравномерной подаче.
        if self._track2_enabled:
            self._mix_channel(outdata, 0, self._queue2, frames, block_start_pts)
        if self._track3_enabled:
            self._mix_channel(outdata, 1, self._queue3, frames, block_start_pts)

        if self._muted:
            outdata.fill(0.0)

    def _mix_channel(self, outdata, channel_idx, queue, frames, block_start_pts):
        """
        Заполняет канал данными из очереди, ВЫРАВНИВАЯ их по времени.

        Раньше блоки укладывались подряд, в порядке поступления, без учёта
        того, какому моменту они соответствуют. Пока подача обеих дорожек
        шла ровно, это работало; но стоило одной дорожке недосчитаться
        пакетов — а на реальных записях в индексе встречаются разрывы —
        каналы сдвигались друг относительно друга, и ДОГНАТЬ БЫЛО НЕЧЕМ:
        каждый канал просто продолжал играть свою очередь по порядку.
        Расхождение фиксировалось навсегда и только росло.

        Теперь позиция каждого блока определяется его PTS:

        - блок целиком в прошлом  -> выбрасывается (опоздал);
        - блок начинается позже   -> до его начала выводится тишина,
                                     то есть разрыв в записи слышен как
                                     пауза, а не сдвигает всё последующее;
        - блок перекрывает точку  -> берётся с нужного смещения внутрь.

        Обе дорожки выравниваются по одному и тому же block_start_pts,
        поэтому расходиться не могут в принципе, а после разрыва каждая
        самостоятельно возвращается на своё место.
        """
        written = 0
        dropped_late = 0
        dropped_stale = 0
        silence_gap = 0

        while written < frames:
            with self._queue_lock:
                if not queue:
                    break
                pts, chunk = queue[0]

                target_pts = block_start_pts + written
                chunk_end = pts + len(chunk)

                if chunk_end <= target_pts:
                    # Блок целиком в прошлом: воспроизводить его уже поздно.
                    queue.popleft()
                    dropped_late += 1
                    continue

                if pts - target_pts > self._max_future_lead:
                    # Блок опережает текущее время на недопустимую
                    # величину — это не разрыв в записи, а УСТАРЕВШИЕ
                    # данные со старой позиции.
                    #
                    # Возникает при перемотке НАЗАД: пока выполняется
                    # переключение, непрерывно работающий декодер звука
                    # успевает дослать пакеты прежней позиции. Их PTS
                    # оказывается далеко впереди новых часов. Без этой
                    # проверки такой блок считался бы "будущим разрывом":
                    # микшер ждал бы его наступления, писал тишину и не
                    # забирал блок из очереди. Новые данные вставали бы в
                    # очередь ЗА ним и не доходили никогда. Очередь
                    # переполнялась, декодер звука вставал по гистерезису,
                    # следом блокировался демуксер, и через 15-20 секунд
                    # останавливалось видео — при бегущем таймкоде.
                    #
                    # Настоящие разрывы в записи составляют доли секунды
                    # (на проверенных файлах — 1024 сэмпла, 21 мс), поэтому
                    # порог в секунду отделяет их от мусора с запасом.
                    queue.popleft()
                    dropped_stale += 1
                    continue

                if pts > target_pts:
                    # Данные этого момента ещё не наступили — разрыв.
                    # Оставляем тишину ровно на его длину.
                    gap = min(frames - written, pts - target_pts)
                    written += gap
                    silence_gap += gap
                    continue

                skip = target_pts - pts               # >= 0
                take = min(len(chunk) - skip, frames - written)
                outdata[written:written + take, channel_idx] = chunk[skip:skip + take]

                if skip + take >= len(chunk):
                    queue.popleft()
                else:
                    # Остаток блока остаётся в очереди со сдвинутым PTS.
                    queue[0] = (pts + skip + take, chunk[skip + take:])

            written += take

        if silence_gap:
            audio_monitor_logger.debug(
                f"AUDIO_GAP channel={channel_idx} samples={silence_gap}")
        if dropped_late:
            audio_monitor_logger.debug(
                f"AUDIO_LATE channel={channel_idx} blocks={dropped_late}")
        if dropped_stale:
            self._stale_dropped += dropped_stale
            audio_monitor_logger.debug(
                f"AUDIO_STALE channel={channel_idx} blocks={dropped_stale}")
        if written < frames:
            self._underruns += 1
            audio_monitor_logger.debug(f"AUDIO_UNDERRUN channel={channel_idx} missing={frames - written}")
        return written

    # ------------------------------------------------------------------
    # Управление потоком
    # ------------------------------------------------------------------
    def start(self):
        """Запускает звуковой поток."""
        if self._active:
            return
        try:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=2,
                callback=self._callback,
                blocksize=self.buffer_size,
                latency='low',
                dtype='float32'
            )
            self._stream.start()
            self._active = True
            self._device_available = True
            logger.info("MasterClock запущен: %d Hz, stereo, buffer=%d",
                        self.sample_rate, self.buffer_size)
        except Exception as e:
            logger.error("Не удалось запустить аудиоустройство: %s", e)
            self._active = False
            self._device_available = False

    def stop(self):
        """Останавливает звуковой поток."""
        self._active = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        logger.info("MasterClock остановлен")

    def close(self):
        """Останавливает и очищает ресурсы."""
        self.stop()
        self.flush_audio()
        logger.info("MasterClock закрыт")

    # ------------------------------------------------------------------
    # Диагностика
    # ------------------------------------------------------------------
    def get_stats(self) -> dict:
        return {
            'samples_played': self.samples_played,
            'queue_size': self.get_queue_size(),
            'audio_queue_samples': self.get_audio_queue_samples(),
            'max_audio_queue_samples': self.max_audio_queue_samples,
            'max_queue_len': self._max_queue_len,
            'underruns': self._underruns,
            'audio_delay': self.audio_delay,
            'track2': self._track2_enabled,
            'track3': self._track3_enabled,
            'muted': self._muted,
            'device_available': self._device_available,
        }
