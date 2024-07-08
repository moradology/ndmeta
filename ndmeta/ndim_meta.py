from dataclasses import dataclass
import functools
from itertools import product
import logging
import operator
from typing import Dict, Optional

import numpy as np
import objsize
import xarray as xr

from .array_meta import ArrayMeta
from . import util


logger = logging.getLogger(__name__)

@dataclass
class NDimMeta:
    array_meta: Dict[str, ArrayMeta]
    concat_dim: Optional[str]

    def to_chunks(self, chunk_sizes: dict):
        for var_name, array_meta in self.array_meta.items():
            dimension_chunks = {}
            
            # Calculate chunk indices for each dimension based on the provided chunk sizes
            for dim, full_size in zip(array_meta.attributes['dimension_names'], array_meta.shape):
                chunk_size = chunk_sizes.get(dim, full_size)  # Use full size if not specified
                num_full_chunks = full_size // chunk_size
                last_chunk_size = full_size % chunk_size
                
                # Create a list of slice objects for each chunk in this dimension
                dimension_chunks[dim] = [slice(i * chunk_size, (i + 1) * chunk_size) for i in range(num_full_chunks)]
                if last_chunk_size > 0:
                    dimension_chunks[dim].append(slice(num_full_chunks * chunk_size, full_size))
            
            # Generate all possible chunk combinations using Cartesian product
            for chunk_combination in product(*dimension_chunks.values()):
                # Map each dimension to its corresponding slice from the combination
                chunk_definition = {dim: slc for dim, slc in zip(dimension_chunks.keys(), chunk_combination)}
                yield var_name, chunk_definition

    def chunk_coverage(self, dataset, chunk_sizes, file_index):
        covered_results = {
            'full_coverage': [],
            'partial_coverage': []
        }
        
        for var_name, chunk_ranges in self.to_chunks(chunk_sizes):
            if var_name in dataset.variables:
                fully_covered = True
                partially_covered = True
                actual_covered_ranges = {}

                for dim, slice_range in chunk_ranges.items():
                    if dim in dataset.variables and dim == self.concat_dim:
                        chunk_size = chunk_sizes[dim]

                        # Calculate the start and end indices of the current file
                        file_start = file_index * dataset.sizes[dim]
                        file_end = file_start + dataset.sizes[dim]

                        chunk_start = slice_range.start
                        chunk_end = slice_range.stop

                        # Calculate the end index of the first chunk within the current file
                        file_chunk_offset = file_start % chunk_size

                        # Not covered at all, move on
                        if chunk_end < file_start or chunk_start > file_end:
                            fully_covered = False
                            partially_covered = False
                            break

                        # Check if the chunk starts before the current file
                        if chunk_start < file_start:
                            fully_covered = False
                            if chunk_end <= file_end:
                                # The chunk ends within the current file
                                actual_covered_ranges[dim] = slice(0, chunk_size - file_chunk_offset)
                            else:
                                # The chunk ends beyond the current file
                                actual_covered_ranges[dim] = slice(0, dataset.sizes[dim])
                        # Check if the chunk starts within the current file
                        elif chunk_start < file_end:
                            if chunk_end <= file_end:
                                # The chunk is fully contained within the current file
                                actual_covered_ranges[dim] = slice(chunk_start - file_start, chunk_end - file_start)
                            else:
                                # The chunk extends beyond the current file
                                actual_covered_ranges[dim] = slice(chunk_start - file_chunk_offset, dataset.sizes[dim])
                                fully_covered = False
                    else:
                        actual_covered_ranges[dim] = slice_range

                if fully_covered:
                    logger.info(f"FULLY COVERED: {var_name=}, {actual_covered_ranges=}")
                    logger.info("---")
                    covered_results['full_coverage'].append((var_name, actual_covered_ranges))
                elif partially_covered and actual_covered_ranges:
                    logger.info(f"PARTIALLY COVERED: {var_name=}, {actual_covered_ranges=}")
                    logger.info("---")
                    covered_results['partial_coverage'].append((var_name, actual_covered_ranges))
                else:
                    logger.info(f"NOT COVERED: {var_name=}, {chunk_ranges=}")
                    logger.info("---")

        return covered_results

    def merge_with(self, other: 'NDimMeta', concat_dim: str) -> 'NDimMeta':
        # Perform initial checks to ensure all variables and dimensions are present in both metadata sets
        self_keys = set(self.array_meta.keys())
        other_keys = set(other.array_meta.keys())
        
        if self_keys != other_keys:
            missing_in_self = other_keys - self_keys
            missing_in_other = self_keys - other_keys
            error_message = f"Metadata mismatch: missing in self {missing_in_self}, missing in other {missing_in_other}."
            raise ValueError(error_message)

        # Create a new dictionary to store merged metadata
        new_metadata_dict = {}

        # Only add keys that are present in both metadata dictionaries
        for key in self_keys:
            if concat_dim in self.array_meta[key].attributes['dimension_names'] and \
                concat_dim in other.array_meta[key].attributes['dimension_names']:
                # Merge metadata and add it to the new dictionary
                new_metadata_dict[key] = self.array_meta[key].merge_with(other.array_meta[key], concat_dim)
            else:
                # Check for equality for dimensions that do not include the concat dimension
                if self.array_meta[key] == other.array_meta[key]:
                    new_metadata_dict[key] = self.array_meta[key]
                else:
                    raise ValueError(f"Non-merging variable '{key}' differs between metadata sets.")

        return NDimMeta(
            array_meta=new_metadata_dict,
            concat_dim=concat_dim
        )

    @classmethod
    def from_xarray(cls, ds: xr.Dataset):
        metadata_dict = {}
        visited_dims = set()

        for var_name, var in ds.variables.items():
            dimension_ranges = {}
            for dim in var.dims:
                dimension_data = ds[dim]
                first_value = dimension_data.isel({dim: 0}).values.item()
                last_value = dimension_data.isel({dim: -1}).values.item()
                dimension_ranges[dim] = (first_value, last_value)
                if dim not in visited_dims:
                    logger.info(f"Dimension {dim} ranges from {first_value} to {last_value}")
                    visited_dims.add(dim)


            if hasattr(var.data, 'chunks') and var.data.chunks:
                chunk_sizes = tuple(map(len, var.data.chunks))
            else:
                chunk_sizes = var.shape  # Default to one chunk per dimension if not chunked

            attributes = var.attrs.copy()
            attributes['dimension_names'] = var.dims

            if var.dtype == object:
                # Rough estimation: Average size of a few sampled elements
                sample_size = min(10, var.shape[0])
                sample_indices = np.random.choice(range(var.shape[0]), size=sample_size)
                average_size = np.mean([objsize.get_exclusive_deep_size(var.values[index]) for index in sample_indices])
                estimated_item_size = average_size
            else:
                estimated_item_size = var.dtype.itemsize

            is_data_var = var_name in ds.data_vars
            
            metadata = ArrayMeta(
                shape=var.shape,
                fill_value=var.attrs.get('_FillValue', None),
                dtype=var.dtype,
                chunk_grid=chunk_sizes,
                attributes=attributes,
                dimension_ranges=dimension_ranges,
                estimated_obj_size=estimated_item_size,
                is_data_var=is_data_var
            )
            metadata_dict[var_name] = metadata

        return cls(array_meta=metadata_dict, concat_dim=None)
    
    @property
    def is_merged(self):
        return self.concat_dim is not None

    @property    
    def data_vars(self):
        return {dim: meta for dim, meta in self.array_meta.items() if meta.is_data_var}

    def analyze_chunking_strategy(self, chunk_sizes):
        logger.info("Starting chunking strategy analysis for each proposed dimension:")
        proposed_chunk_mem = {}
        for dim, chunk_size in chunk_sizes.items():
            logger.info(f"\n=== Dimension: {dim} ===")
            dim_size = self.array_meta[dim].shape[0]
            estimated_object_size = self.array_meta[dim].estimated_obj_size
            proposed_chunk_mem[dim] = util.analyze_chunking_strategy(dim_size, chunk_size, estimated_object_size)

        chunk_sizes['bnds'] = 2
        logger.info("With the proposed chunking strategy, data variable chunks will have roughly the following sizes:")
        for var_name, meta in self.data_vars.items():
            data_var_indices = meta.attributes['dimension_names']
            data_var_chunk_shape = [chunk_sizes[dim] for dim in data_var_indices]
            data_var_chunk_mem = meta.estimated_obj_size * functools.reduce(operator.mul, data_var_chunk_shape)
            formatted_mem = util.format_mem_size(data_var_chunk_mem)
            logger.info(f"Data Variable {var_name}")
            logger.info(f"  - Chunk shape {tuple(data_var_chunk_shape)} @{formatted_mem} per chunk.")
