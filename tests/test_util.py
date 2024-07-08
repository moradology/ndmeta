from ndmeta.util import format_mem_size, analyze_chunking_strategy


def test_format_mem_size():
    assert(format_mem_size(1000) == '1000.00B')
    assert(format_mem_size(1000000) == '976.56KB')
    assert(format_mem_size(100000000) == '95.37MB')


def test_analyze_chunking_strategy():
    dim_size = 1000
    proposed_chunk_size = 10
    estimated_object_size = 64
    proposed_chunk_mem = analyze_chunking_strategy(
        dim_size,
        proposed_chunk_size,
        estimated_object_size
    )
    assert(proposed_chunk_mem == 640)

