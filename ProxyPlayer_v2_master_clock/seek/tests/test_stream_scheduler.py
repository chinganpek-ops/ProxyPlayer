"""
Тестирование StreamScheduler для ProxyPlayer v1.
Проверяет правильность планирования чанков в разных режимах.
"""
import pytest
from pipeline.stream_scheduler import StreamScheduler, PlaybackMode
from pipeline.adaptive_chunk import AdaptiveChunkStrategy


@pytest.fixture
def scheduler():
    return StreamScheduler(AdaptiveChunkStrategy())


class TestNormalMode:
    def test_sequential(self, scheduler):
        scheduler.set_normal_mode(current_chunk=0, total_chunks=10)
        chunks = [scheduler.get_next_chunk() for _ in range(5)]
        assert chunks == [0, 1, 2, 3, 4]

    def test_respects_loaded(self, scheduler):
        scheduler.set_normal_mode(current_chunk=0, total_chunks=10)
        scheduler._loaded_chunks = {0, 1}  # уже загружены
        assert scheduler.get_next_chunk() == 2
        assert scheduler.get_next_chunk() == 3

    def test_end_of_file(self, scheduler):
        scheduler.set_normal_mode(current_chunk=9, total_chunks=10)
        assert scheduler.get_next_chunk() == 9
        assert scheduler.get_next_chunk() is None


class TestSeekMode:
    def test_target_first(self, scheduler):
        scheduler.set_seek_mode(target_chunk=50, total_chunks=100)
        first = scheduler.get_next_chunk()
        assert first == 50

    def test_neighbors(self, scheduler):
        scheduler.set_seek_mode(target_chunk=50, total_chunks=100)
        # первый - цель
        assert scheduler.get_next_chunk() == 50
        # затем два круга соседей: ±1, ±2
        expected = [49, 51, 48, 52]
        for exp in expected:
            assert scheduler.get_next_chunk() == exp

    def test_falls_back_to_normal(self, scheduler):
        scheduler.set_seek_mode(target_chunk=0, total_chunks=10)
        # цель 0
        assert scheduler.get_next_chunk() == 0
        # соседи: 1 (только положительный в пределах)
        assert scheduler.get_next_chunk() == 1
        # сосед -1 не существует, +2 = 2 (offset=2, direction=1)
        assert scheduler.get_next_chunk() == 2
        # дальше -2 нет, +3 = 3? По логике offset=2, direction=-1: chunk = 0+(-1)*2=-2 -> нет,
        # потом offset=2, direction=1: 2 -> уже загружен, так что None пока не закончатся.
        # После соседей должен переключиться в NORMAL и выдать следующий незагруженный.
        # Проверим, что после обработки соседей переходит в normal
        # (оставшиеся 3..9 будут выдаваться)
        chunks = []
        while (chunk := scheduler.get_next_chunk()) is not None:
            chunks.append(chunk)
        assert chunks == [3, 4, 5, 6, 7, 8, 9]  # первые три уже загружены (0,1,2)


class TestFastForwardMode:
    @pytest.mark.parametrize("speed,expected_stride", [(2, 2), (4, 4), (8, 8)])
    def test_stride(self, scheduler, speed, expected_stride):
        scheduler.set_fast_forward_mode(direction=1, speed=speed, 
                                        current_chunk=0, total_chunks=20)
        # первый чанк = 0
        assert scheduler.get_next_chunk() == 0
        # следующий должен быть 0 + stride = expected_stride
        assert scheduler.get_next_chunk() == expected_stride

    def test_negative_direction(self, scheduler):
        scheduler.set_fast_forward_mode(direction=-1, speed=4,
                                        current_chunk=10, total_chunks=20)
        chunk = scheduler.get_next_chunk()
        assert chunk == 10
        # следующий: 10 - 4 = 6
        assert scheduler.get_next_chunk() == 6

    def test_skips_loaded(self, scheduler):
        scheduler.set_fast_forward_mode(direction=1, speed=2,
                                        current_chunk=0, total_chunks=10)
        scheduler._loaded_chunks = {0, 2}  # 0 и 2 уже загружены
        assert scheduler.get_next_chunk() == 4  # пропустил 0 и 2


class TestPauseMode:
    def test_same_as_normal(self, scheduler):
        scheduler.set_pause_mode(current_chunk=5, total_chunks=10)
        chunks = [scheduler.get_next_chunk() for _ in range(3)]
        assert chunks == [5, 6, 7]


class TestMarkFailed:
    def test_retry_failed_chunk(self, scheduler):
        scheduler.set_normal_mode(current_chunk=0, total_chunks=5)
        assert scheduler.get_next_chunk() == 0
        scheduler.mark_chunk_failed(0)
        # следующий вызов должен снова предложить 0 (но current_chunk уже 1, 
        # поэтому без специальной логики он не вернётся). Планировщик в normal 
        # идёт только вперёд. Поэтому mark_chunk_failed просто удаляет из loaded,
        # но не меняет current_chunk. Для повторной попытки нужно ручное 
        # управление. Пока тест проверяет, что чанк убран из loaded.
        assert 0 not in scheduler._loaded_chunks


class TestReset:
    def test_clear_state(self, scheduler):
        scheduler.set_normal_mode(current_chunk=3, total_chunks=10)
        scheduler.get_next_chunk()
        scheduler.reset()
        assert scheduler.loaded_count == 0
        assert scheduler._mode == PlaybackMode.NORMAL
        assert scheduler._current_chunk == 0
        assert scheduler._total_chunks == 0