from dataclasses import dataclass, field
from itertools import product
from pprint import pprint
from typing import Any, Dict, Tuple

import cftime
import xarray as xr
import numpy as np

@dataclass
class ArrayMeta:
    shape: tuple
    fill_value: any
    dtype: np.dtype
    chunk_grid: tuple
    attributes: dict
    dimension_ranges: Dict[str, Tuple[Any, Any]]
    estimated_obj_size: int
    is_data_var: bool

    def __post_init__(self):
        self.ndim = len(self.shape)

    def to_dict(self):
        return {
            "shape": self.shape,
            "fill_value": str(self.fill_value),
            "dtype": str(self.dtype),
            "chunk_grid": self.chunk_grid,
            "attributes": self.attributes,
            "dimension_ranges": self.dimension_ranges,
            "estimated_obj_size": int,
            "is_data_var": self.is_data_var
        }

    def merge_with(self, other: 'ArrayMeta', concat_dim: str):
        if self.dtype != other.dtype:
            raise ValueError("Data types do not match")
        if self.fill_value != other.fill_value:
            raise ValueError("Fill values do not match")

        try:
            concat_index = self.attributes['dimension_names'].index(concat_dim)
        except ValueError:
            raise ValueError(f"Concatenation dimension {concat_dim} not found in metadata for '{self.attributes.get('standard_name', 'unknown variable')}'.")

        new_shape = list(self.shape)
        new_shape[concat_index] += other.shape[concat_index]

        new_chunk_grid = self.chunk_grid  # Assuming chunk grids remain the same for simplicity
        new_attributes = {**self.attributes, **other.attributes}

        # Dimension range merging logic
        new_dimension_ranges = {}
        for dim in set(self.dimension_ranges.keys()).union(other.dimension_ranges.keys()):
            if dim in self.dimension_ranges and dim in other.dimension_ranges:
                if dim == concat_dim:
                    # For the concatenation dimension, extend the range
                    new_dimension_ranges[dim] = (self.dimension_ranges[dim][0], other.dimension_ranges[dim][1])
                else:
                    # For non-concatenation dimensions, find the min and max of the ranges
                    start = min(self.dimension_ranges[dim][0], other.dimension_ranges[dim][0])
                    end = max(self.dimension_ranges[dim][1], other.dimension_ranges[dim][1])
                    new_dimension_ranges[dim] = (start, end)
            elif dim in self.dimension_ranges:
                new_dimension_ranges[dim] = self.dimension_ranges[dim]
            else:
                new_dimension_ranges[dim] = other.dimension_ranges[dim]

        average_obj_size = (self.estimated_obj_size + other.estimated_obj_size) / 2

        return ArrayMeta(
            shape=tuple(new_shape),
            fill_value=self.fill_value,
            dtype=self.dtype,
            chunk_grid=new_chunk_grid,
            attributes=new_attributes,
            dimension_ranges=new_dimension_ranges,
            estimated_obj_size=average_obj_size,
            is_data_var=self.is_data_var
        )
