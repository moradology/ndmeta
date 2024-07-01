def format_mem_size(bytes, suffix="B"):
    """Scale bytes to its proper format"""
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor

def analyze_chunking_strategy(dim_size, proposed_chunk_size, estimated_object_size):
    print(f"\nAnalyzing chunking strategy for dimension size {dim_size} and proposed chunk size {proposed_chunk_size}:")
    num_chunks = dim_size // proposed_chunk_size
    remainder = dim_size % proposed_chunk_size
    proposed_chunk_mem = estimated_object_size * proposed_chunk_size
    proposed_formatted_mem_size = format_mem_size(proposed_chunk_mem)
    print(f"  - {num_chunks} chunks of size {proposed_chunk_size}, with a remainder of {remainder} @{proposed_formatted_mem_size} per chunk.")

    print("\nAlternative chunk sizes based on divisors:")
    divisors = [i for i in range(1, dim_size + 1) if dim_size % i == 0 and i != 1 and i != dim_size]
    alternative_chunk_sizes = sorted(divisors, key=lambda x: abs(x - proposed_chunk_size))

    for alternative_chunk_size in alternative_chunk_sizes[:5]:
        alternative_chunk_mem = estimated_object_size * alternative_chunk_size
        alternative_formatted_mem_size = format_mem_size(alternative_chunk_mem)
        print(f"  - Chunk size of {alternative_chunk_size} evenly divides the dimension @{alternative_formatted_mem_size} per chunk.")

    return proposed_chunk_mem