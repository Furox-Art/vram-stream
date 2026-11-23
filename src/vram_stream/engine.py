"""Run a matrix product without placing the full matrices on a GPU."""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from .planner import DEFAULT_BUDGET_BYTES, TilePlan, matrix_dtype, plan_matmul


class DeviceUnavailableError(RuntimeError):
    """The requested GPU backend or device is unavailable."""


@dataclass(frozen=True, slots=True)
class MatmulResult:
    """The host result and the tile plan used to compute it."""

    output: np.ndarray
    plan: TilePlan
    device: str
    elapsed_seconds: float
    oom_retries: int


def _cuda_device(device: str):
    if not isinstance(device, str) or not re.fullmatch(
        r"auto|cpu|cuda(?::[0-9]+)?", device
    ):
        raise ValueError("device must be auto, cpu, cuda or cuda:<index>")
    if device == "cpu":
        return None
    try:
        torch = importlib.import_module("torch")
    except (ImportError, OSError) as error:
        if device == "auto":
            return None
        raise DeviceUnavailableError(
            "Install a GPU-enabled build of PyTorch."
        ) from error
    if not torch.cuda.is_available():
        if device == "auto":
            return None
        raise DeviceUnavailableError("PyTorch could not find a CUDA or ROCm device.")
    index = (
        int(device.split(":", 1)[1]) if ":" in device else torch.cuda.current_device()
    )
    if index >= torch.cuda.device_count():
        raise DeviceUnavailableError(f"GPU index {index} is unavailable.")
    return torch, index


def _mapped_path(array: np.ndarray) -> Path | None:
    while isinstance(array, np.ndarray):
        if isinstance(array, np.memmap) and array.filename is not None:
            return Path(array.filename).resolve()
        array = array.base
    return None


def _check_output(
    out: np.ndarray, left: np.ndarray, right: np.ndarray, plan: TilePlan
) -> None:
    if not isinstance(out, np.ndarray):
        raise TypeError("out must be a NumPy array")
    if out.shape != plan.output_shape or out.dtype != left.dtype:
        raise ValueError("out must match the result shape and dtype")
    if not out.flags.writeable or not (
        out.flags.c_contiguous or out.flags.f_contiguous
    ):
        raise ValueError("out must be writable and contiguous")
    mapped = _mapped_path(out)
    for source in (left, right):
        if np.may_share_memory(out, source):
            raise ValueError("out must not share memory with either input")
        path = _mapped_path(source) if mapped is not None else None
        if path is not None:
            try:
                same_file = mapped == path or mapped.samefile(path)
            except OSError:
                same_file = mapped == path
            if same_file:
                raise ValueError("out must not map a file used as input")


def _blocks(length: int, width: int):
    for start in range(0, length, width):
        yield slice(start, min(start + width, length))


def _numpy_matmul(
    left: np.ndarray, right: np.ndarray, out: np.ndarray, plan: TilePlan
) -> None:
    buffer = np.empty((plan.tile_rows, plan.tile_columns), dtype=left.dtype)
    for rows in _blocks(out.shape[0], plan.tile_rows):
        for columns in _blocks(out.shape[1], plan.tile_columns):
            dest = out[rows, columns]
            dest.fill(0)
            product = buffer[: dest.shape[0], : dest.shape[1]]
            for inner in _blocks(left.shape[1], plan.tile_inner):
                np.matmul(left[rows, inner], right[inner, columns], out=product)
                np.add(dest, product, out=dest)


def _torch_matmul(torch, device, left, right, out, plan: TilePlan) -> None:
    shapes = (
        (plan.tile_rows, plan.tile_inner),
        (plan.tile_inner, plan.tile_columns),
        (plan.tile_rows, plan.tile_columns),
    )
    # Staging buffers accept strided or read-only input arrays.
    host_left, host_right, host_result = (
        np.empty(shape, dtype=left.dtype) for shape in shapes
    )
    cpu_left, cpu_right, cpu_result = (
        torch.from_numpy(buffer) for buffer in (host_left, host_right, host_result)
    )
    gpu_left, gpu_right, gpu_result = (
        torch.empty(shape, device=device, dtype=cpu_left.dtype) for shape in shapes
    )
    for rows in _blocks(out.shape[0], plan.tile_rows):
        for columns in _blocks(out.shape[1], plan.tile_columns):
            row_count = rows.stop - rows.start
            column_count = columns.stop - columns.start
            result_tile = gpu_result[:row_count, :column_count]
            result_tile.zero_()
            for inner in _blocks(left.shape[1], plan.tile_inner):
                inner_count = inner.stop - inner.start
                a = (slice(row_count), slice(inner_count))
                b = (slice(inner_count), slice(column_count))
                np.copyto(host_left[a], left[rows, inner])
                np.copyto(host_right[b], right[inner, columns])
                gpu_left[a].copy_(cpu_left[a])
                gpu_right[b].copy_(cpu_right[b])
                result_tile.addmm_(gpu_left[a], gpu_right[b])
            result = (slice(row_count), slice(column_count))
            cpu_result[result].copy_(result_tile)
            np.copyto(out[rows, columns], host_result[result])


def _cuda_matmul(
    torch, index, left, right, out, plan: TilePlan
) -> tuple[TilePlan, int]:
    minimum = 3 * left.dtype.itemsize
    with torch.cuda.device(index), torch.inference_mode():
        free, _ = torch.cuda.mem_get_info(index)
        budget = min(plan.budget_bytes, free * 4 // 5)
        if budget < minimum:
            raise RuntimeError(
                f"GPU {index} has insufficient free memory for three values."
            )
        device = torch.device("cuda", index)
        retries = 0
        while True:
            plan = plan_matmul(
                left.shape, right.shape, dtype=left.dtype, budget_bytes=budget
            )
            try:
                _torch_matmul(torch, device, left, right, out, plan)
            except torch.cuda.OutOfMemoryError:
                if plan.tile_bytes == minimum:
                    raise
            else:
                return plan, retries
            # The failed call's tensor references are released before retrying.
            # Every output block, including any already written, is recomputed.
            torch.cuda.empty_cache()
            budget = max(minimum, plan.tile_bytes // 2)
            retries += 1


def matmul(
    left: np.ndarray,
    right: np.ndarray,
    *,
    out: np.ndarray | None = None,
    device: str = "auto",
    budget_bytes: int = DEFAULT_BUDGET_BYTES,
) -> MatmulResult:
    """Multiply two host arrays using CPU or GPU tiles.

    Inputs must be two-dimensional NumPy arrays with the same float32 or
    float64 dtype. A supplied output must be independent of both inputs.
    The byte budget covers the three planned GPU data buffers.
    """
    if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
        raise TypeError("left and right must be NumPy arrays")
    if matrix_dtype(left.dtype) != matrix_dtype(right.dtype):
        raise ValueError("left and right must have the same dtype")
    plan = plan_matmul(
        left.shape, right.shape, dtype=left.dtype, budget_bytes=budget_bytes
    )
    if out is not None:
        _check_output(out, left, right, plan)
    gpu = _cuda_device(device)
    result = np.empty(plan.output_shape, dtype=left.dtype) if out is None else out
    start = perf_counter()
    retries = 0
    used_device = "cpu"
    if plan.tile_bytes == 0:
        result.fill(0)
    elif gpu is None:
        _numpy_matmul(left, right, result, plan)
    else:
        torch, index = gpu
        plan, retries = _cuda_matmul(torch, index, left, right, result, plan)
        used_device = f"cuda:{index}"
    return MatmulResult(result, plan, used_device, perf_counter() - start, retries)
