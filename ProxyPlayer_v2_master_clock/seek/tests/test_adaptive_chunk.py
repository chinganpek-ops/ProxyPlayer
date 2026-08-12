"""
Тестирование AdaptiveChunkStrategy.
Запуск: python -m pytest tests/test_adaptive_chunk.py -v
"""

from pipeline.adaptive_chunk import AdaptiveChunkStrategy


def test_chunk_size_normal():
    strategy = AdaptiveChunkStrategy()
    assert strategy.get_chunk_size(1.0) == 12

def test_chunk_size_fast():
    strategy = AdaptiveChunkStrategy()
    assert strategy.get_chunk_size(2.0) == 24
    assert strategy.get_chunk_size(4.0) == 24
    assert strategy.get_chunk_size(4.1) == 48
    assert strategy.get_chunk_size(8.0) == 48

def test_stride():
    strategy = AdaptiveChunkStrategy()
    assert strategy.get_fast_forward_stride(1.0) == 1
    assert strategy.get_fast_forward_stride(2.0) == 2
    assert strategy.get_fast_forward_stride(4.0) == 4
    assert strategy.get_fast_forward_stride(8.0) == 8

def test_lookahead():
    strategy = AdaptiveChunkStrategy()
    assert strategy.get_lookahead_chunks(1.0, 600) > 0
    assert strategy.get_lookahead_chunks(2.0, 600) == 2