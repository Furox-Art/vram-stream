"""Plan the three device buffers used by a blocked matrix product."""

from collections.abc import Sequence
from dataclasses import dataclass
from operator import index

import numpy as np
from numpy.typing import DTypeLike

MIB = 1 << 20
DEFAULT_BUDGET_BYTES = 256 * MIB


def matrix_dtype(dtype: DTypeLike) -> np.dtype:
    result = np.dtype(dtype)
    if result not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("matrices must use native-endian float32 or float64")
    return result


def _integer(value: int, label: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be an integer")
    try:
        number = index(value)
    except TypeError as error:
        raise TypeError(f"{label} must be an integer") from error
    if number < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return number


def _shape(value: Sequence[int], name: str) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"{name} must be two-dimensional")
    return _integer(value[0], f"{name} rows", 0), _integer(
        value[1], f"{name} columns", 0
    )


@dataclass(frozen=True, slots=True)
class TilePlan:
    """Matrix shapes, tile shapes and the byte count of the three data buffers."""

    left_shape: tuple[int, int]
    right_shape: tuple[int, int]
    dtype: str
    tile_rows: int
    tile_columns: int
    tile_inner: int
    tile_bytes: int
    budget_bytes: int
    full_matrix_bytes: int

    @property
    def output_shape(self) -> tuple[int, int]:
        return self.left_shape[0], self.right_shape[1]


def plan_matmul(
    left_shape: Sequence[int],
    right_shape: Sequence[int],
    *,
    dtype: DTypeLike = "float32",
    budget_bytes: int = DEFAULT_BUDGET_BYTES,
) -> TilePlan:
    """Find the largest balanced tile within a byte budget, without allocating arrays."""
    rows, inner = _shape(left_shape, "left matrix")
    right_inner, columns = _shape(right_shape, "right matrix")
    if inner != right_inner:
        raise ValueError(f"matrix inner dimensions differ ({inner} and {right_inner})")
    scalar = matrix_dtype(dtype)
    budget = _integer(budget_bytes, "budget_bytes", 1)

    def size(r: int, c: int, k: int) -> int:
        return (r * k + k * c + r * c) * scalar.itemsize

    full = size(rows, columns, inner)
    if min(rows, columns, inner) == 0:
        tile_rows = tile_columns = tile_inner = tile_bytes = 0
    else:
        minimum = size(1, 1, 1)
        if budget < minimum:
            raise ValueError(f"budget_bytes must be at least {minimum} for this dtype")
        low, high = 1, max(rows, columns, inner)
        while low < high:
            edge = (low + high + 1) // 2
            required = size(min(rows, edge), min(columns, edge), min(inner, edge))
            if required <= budget:
                low = edge
            else:
                high = edge - 1
        tile_rows = min(rows, low)
        tile_columns = min(columns, low)
        tile_inner = min(inner, low)
        tile_bytes = size(tile_rows, tile_columns, tile_inner)
    return TilePlan(
        (rows, inner),
        (inner, columns),
        scalar.name,
        tile_rows,
        tile_columns,
        tile_inner,
        tile_bytes,
        budget,
        full,
    )
