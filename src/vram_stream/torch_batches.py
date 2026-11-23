"""Feed contiguous host arrays to PyTorch through the CUDA read cache."""

from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np

from .cuda_cache import CacheStats, open_cuda_cache
from .engine import DeviceUnavailableError, _cuda_device
from .planner import _integer

_TORCH_DTYPES = {
    np.dtype(name)
    for name in (
        "bool",
        "uint8",
        "int8",
        "int16",
        "int32",
        "int64",
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    )
}


class _CudaArray:
    def __init__(self, pointer, shape, dtype, stream):
        self.__cuda_array_interface__ = {
            "shape": shape,
            "typestr": dtype.str,
            "data": (pointer, False),
            "strides": None,
            "version": 3,
            "stream": stream,
        }


class TorchBatchSource:
    """A reusable batch iterable whose GPU cache lives inside open_torch_batches."""

    def __init__(self, torch, cache, device, source):
        self._torch = torch
        self._cache = cache
        self._device = device
        self._shape = source.shape[1:]
        self._dtype = source.dtype
        self._torch_dtype = torch.from_numpy(np.empty(0, dtype=source.dtype)).dtype
        self._row_bytes = source.nbytes // len(source)
        self._stage = torch.cuda.ExternalStream(cache.compute_stream, device=device)

    @property
    def stats(self) -> CacheStats:
        """Transfers and repeated reads across all iterations."""
        return self._cache.stats

    def __iter__(self):
        for index in range(self._cache.block_count):
            with self._cache.read(index) as block:
                shape = (block.size_bytes // self._row_bytes, *self._shape)
                consumer = self._torch.cuda.current_stream(self._device)
                batch = self._torch.empty(
                    shape, dtype=self._torch_dtype, device=self._device
                )
                view = _CudaArray(
                    block.device_pointer, shape, self._dtype, block.stream
                )
                # Allocate on the PyTorch stream and return an independent Tensor.
                self._stage.wait_stream(consumer)
                try:
                    with self._torch.cuda.stream(self._stage):
                        batch.copy_(self._torch.as_tensor(view, device=self._device))
                finally:
                    consumer.wait_stream(self._stage)
                if self._cache.slot_count > 1 and index + 1 < self._cache.block_count:
                    self._cache.prefetch(index + 1)
            yield batch


@contextmanager
def open_torch_batches(
    source: np.ndarray,
    *,
    batch_rows: int,
    slots: int = 2,
    device: int = 0,
    vram_budget_bytes: int | None = None,
    pinned_budget_bytes: int | None = None,
) -> Iterator[TorchBatchSource]:
    """Yield an iterable of owned CUDA tensors from a NumPy dataset.

    Each iteration streams batches in row order. Keep the dataset values
    stable until the context exits. The cache budget is slots times one
    batch in each memory tier. Training and retained tensors use additional
    PyTorch memory.
    """
    batch_rows = _integer(batch_rows, "batch_rows", 1)
    slots = _integer(slots, "slots", 1)
    device = _integer(device, "device", 0)
    if not isinstance(source, np.ndarray) or not source.ndim or not source.size:
        raise ValueError("source must be a non-empty NumPy array with rows")
    if not source.flags.c_contiguous or source.dtype not in _TORCH_DTYPES:
        raise ValueError(
            "source must be C-contiguous with a native PyTorch-compatible dtype"
        )
    torch, _ = _cuda_device(f"cuda:{device}")
    if not torch.version.cuda:
        raise DeviceUnavailableError(
            "the batch cache requires an NVIDIA CUDA build of PyTorch"
        )
    row_bytes = source.nbytes // len(source)
    with open_cuda_cache(
        source,
        block_bytes=batch_rows * row_bytes,
        slots=slots,
        device=device,
        vram_budget_bytes=vram_budget_bytes,
        pinned_budget_bytes=pinned_budget_bytes,
    ) as cache:
        yield TorchBatchSource(torch, cache, torch.device(f"cuda:{device}"), source)
