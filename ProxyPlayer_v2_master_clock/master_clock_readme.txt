План интеграции MasterClock
1. Установить sounddevice
bash

pip install sounddevice

2. Заменить AudioOutput на MasterClock в StreamController

В core/stream_controller.py:

    Удалить from output.audio_output import AudioOutput

    Добавить from core.master_clock import MasterClock

    В _background_init() убрать создание AudioOutput, вместо этого создать MasterClock (но не запускать)

    В start_playback() создать и подключить MasterClock к PlaybackEngine

    В resume() и pause() вызывать master_clock.start() / master_clock.stop()

3. Адаптировать PlaybackEngine

    PlaybackEngine больше не использует AudioOutput, а получает MasterClock

    audio_clock теперь берётся из master_clock.get_audio_clock() в get_display_frame()

    При resume() вызывается master_clock.start(), при pause() — master_clock.stop()

4. Адаптировать AudioDecoderStage

    Вместо записи в MultiTrackAudioBuffer, AudioDecoderStage будет вызывать master_clock.push_audio(pcm_block)

    MultiTrackAudioBuffer и AudioRingBuffer больше не нужны для вывода (можно оставить для тестов)

5. Синхронизация видео

    В SyncManager.get_display_frame() использовать master_clock.get_audio_clock() как опорный audio_clock

    Убрать всю коррекцию дрейфа (она больше не нужна)

    1. Что мы берём из v1 без изменений

Эти модули уже отлажены и не зависят от способа вывода аудио:

    index/ — lazy_index.py, moov_builder.py, idx_cache.py

    file_io/ — win_sequential_reader.py

    decode/ — decoder.py, audio_decoder.py

    config/ — config.py, timebase.py, logger.py (можно чуть адаптировать)

    utils/ — все утилиты

    buffer/frame_buffer.py

    pipeline/stream_scheduler.py, adaptive_chunk.py

    seek/seek_engine.py

2. Что мы удаляем или полностью заменяем

    output/audio_output.py — полностью заменяется MasterClock

    buffer/audio_buffer.py — MultiTrackAudioBuffer / AudioRingBuffer больше не нужны для вывода, аудио идёт напрямую в MasterClock

    core/sync_manager.py — упрощается: дрейф-коррекция не нужна, весь audio_clock приходит из MasterClock

    core/playback_engine.py — адаптируется под MasterClock (без AudioOutput)

    core/stream_controller.py — создаёт и подключает MasterClock

    pipeline/chunk_pipeline.py — AudioDecoderStage теперь пушит данные прямо в MasterClock, а не в MultiTrackAudioBuffer

3. Архитектура с MasterClock
text

MasterClock (sounddevice)
   ├── callback: обновляет _samples_played, микширует аудио из очереди
   ├── push_audio(samples) – вызывается из AudioDecoderStage
   ├── get_audio_clock() – единый источник времени для SyncManager
   ├── start() / stop() – управление потоком
   └── reset() – при перемотке

PlaybackEngine
   ├── владеет MasterClock
   ├── get_display_frame() → SyncManager.get_display_frame(clock=master.get_audio_clock())
   ├── resume() → master.start()
   ├── pause() → master.stop()
   └── seek() → master.reset() и т.д.

AudioDecoderStage
   └── декодированные сэмплы напрямую отправляются в master.push_audio(pcm_block)

Ключевые изменения:

    SyncManager больше не хранит модель дрейфа — он просто получает актуальный audio_clock из MasterClock и выбирает кадр.

    PlaybackEngine не использует отдельный AudioOutput, а работает с MasterClock.

    StreamController создаёт MasterClock в главном потоке при start_playback() и передаёт его в PlaybackEngine.

4. Первые практические шаги
4.1 Создайте структуру v2

Скопируйте проект в новую папку ProxyPlayer_v2_master_clock, удалите неактуальное (output/audio_output.py, buffer/audio_buffer.py).
4.2 Установите sounddevice
bash

pip install sounddevice

4.3 Напишите тестовый скрипт для MasterClock

В корне проекта создайте test_master_clock.py:

    Создайте MasterClock, заполните очередь тестовым синусом, запустите start() и через 3 секунды stop().

    Убедитесь, что в наушниках слышен тон, а samples_played монотонно растёт.

4.4 Интегрируйте MasterClock в AudioDecoderStage

    Временно отключите видео-декодер, оставьте только аудио-конвейер.

    AudioDecoderStage._push_audio() пусть вызывает master_clock.push_audio(pcm_block).

    Проверьте, что слышен звук из файла (без видео).

4.5 Подключите видео и синхронизацию

    В SyncManager принимайте audio_clock из MasterClock (без drift-коррекции).

    Убедитесь, что видео идёт плавно, таймкод бежит, звук синхронен.

4.6 Полная интеграция и стресс-тесты

    Верните все компоненты (seek, JKL, live-рост) и прогоните те же 6-часовые файлы, которые выявляли проблемы в v1.

5. Что мы выигрываем

    Надёжный аудиовыход без привязки к Qt main thread.

    Точная синхронизация — видео всегда следует за железным клоком звуковой карты.

    Упрощение — уходят AudioRingBuffer, MultiTrackAudioBuffer, коррекция дрейфа, проблемы с read_pos.

    Масштабируемость — можно легко добавить многоканальный LTC или дополнительные звуковые дорожки.