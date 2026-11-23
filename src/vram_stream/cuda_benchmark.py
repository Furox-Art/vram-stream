"""Compare repeated mapped-RAM reads and bounded VRAM caching."""

import ctypes as ct
import sys
from contextlib import ExitStack
from dataclasses import asdict
from importlib.resources import files
from time import perf_counter

import numpy as np

from .cuda_cache import _stream, open_cuda_cache
from .cuda_workspace import CudaWorkspaceError, _allocate
from .planner import MIB, _integer

_WORD_BYTES = np.dtype("uint32").itemsize
_THREADS = 256


def _timed_sum(cuda, stream, output, words, enqueue):
    ptx = files("vram_stream").joinpath("add_words.ptx").read_bytes()
    module = ct.c_void_p()
    cuda.call("cuModuleLoadData", ct.byref(module), ptx)
    with ExitStack() as cleanup:
        cleanup.callback(cuda.call, "cuModuleUnload", module)
        cleanup.callback(cuda.call, "cuStreamSynchronize", stream)
        kernel = ct.c_void_p()
        cuda.call("cuModuleGetFunction", ct.byref(kernel), module, b"add_words")

        def launch(source, count=words):
            grid = max(1, min(4096, (count + _THREADS - 1) // _THREADS))
            args = (ct.c_uint64(source), ct.c_uint64(output), ct.c_uint64(count))
            params = (ct.c_void_p * len(args))(*(ct.addressof(arg) for arg in args))
            cuda.call(
                "cuLaunchKernel",
                kernel,
                grid,
                1,
                1,
                _THREADS,
                1,
                1,
                0,
                stream,
                params,
                None,
            )

        # Initialize the result and warm the kernel without reading either input path.
        cuda.call("cuMemsetD32_v2", output, 0, words)
        cuda.call("cuCtxSynchronize")
        launch(output, 0)
        cuda.call("cuStreamSynchronize", stream)
        start = perf_counter()
        enqueue(launch)
        cuda.call("cuStreamSynchronize", stream)
        elapsed = perf_counter() - start
        values = np.empty(words, dtype=np.uint32)
        cuda.call("cuMemcpyDtoH_v2", values.ctypes.data, output, values.nbytes)
    return elapsed, values


def _mapped_sum(source, block_bytes, blocks, passes, device):
    with (
        _allocate(block_bytes, source.nbytes, device) as (cuda, workspace),
        ExitStack() as cleanup,
    ):
        stream = _stream(cuda, cleanup)
        ct.memmove(workspace.ram_host_pointer, source.ctypes.data, source.nbytes)

        def enqueue(launch):
            for _ in range(passes):
                for index in range(blocks):
                    launch(workspace.ram_device_pointer + index * block_bytes)

        return _timed_sum(
            cuda,
            stream,
            workspace.vram_device_pointer,
            block_bytes // _WORD_BYTES,
            enqueue,
        )


def _cached_sum(source, block_bytes, slots, passes, device):
    with open_cuda_cache(
        source, block_bytes=block_bytes, slots=slots, device=device
    ) as cache:
        cuda = cache._cuda
        output = ct.c_uint64()
        cuda.call("cuMemAlloc_v2", ct.byref(output), block_bytes)
        with ExitStack() as cleanup:
            cleanup.callback(cuda.call, "cuMemFree_v2", output)

            def enqueue(launch):
                for _ in range(passes):
                    for index in range(cache.block_count):
                        with cache.read(index) as block:
                            launch(block.device_pointer)
                            if cache.slot_count > 1 and index + 1 < cache.block_count:
                                cache.prefetch(index + 1)

            elapsed, values = _timed_sum(
                cuda,
                cache.compute_stream,
                output.value,
                block_bytes // _WORD_BYTES,
                enqueue,
            )
        return elapsed, values, cache.stats


def benchmark_cuda_cache(
    *,
    block_bytes: int = 16 * MIB,
    blocks: int = 2,
    slots: int = 2,
    passes: int = 4,
    device: int = 0,
) -> dict:
    """Time both paths and check every output word against a NumPy reference."""
    block_bytes = _integer(block_bytes, "block_bytes", _WORD_BYTES)
    blocks = _integer(blocks, "blocks", 1)
    slots = _integer(slots, "slots", 1)
    passes = _integer(passes, "passes", 1)
    device = _integer(device, "device", 0)
    if block_bytes % _WORD_BYTES:
        raise ValueError("block_bytes must be a multiple of four")
    if block_bytes * blocks > sys.maxsize:
        raise ValueError("benchmark source is too large")
    words = block_bytes // _WORD_BYTES
    source = np.arange(words * blocks, dtype=np.uint32)

    mapped_seconds, mapped = _mapped_sum(source, block_bytes, blocks, passes, device)
    cached_seconds, cached, stats = _cached_sum(
        source, block_bytes, slots, passes, device
    )
    expected = source.reshape(blocks, words).sum(axis=0, dtype=np.uint32)
    expected *= np.uint32(passes % (1 << 32))
    if not (np.array_equal(mapped, expected) and np.array_equal(cached, expected)):
        raise CudaWorkspaceError(
            "the mapped and cached results did not both match NumPy"
        )

    return {
        "device": f"cuda:{device}",
        "block_bytes": block_bytes,
        "blocks": blocks,
        "passes": passes,
        "logical_read_bytes": source.nbytes * passes,
        "direct_mapped": {
            "seconds": mapped_seconds,
            "pinned_ram_bytes": source.nbytes,
            "vram_bytes": block_bytes,
        },
        "staged_vram": {
            "seconds": cached_seconds,
            **asdict(stats),
            "vram_bytes": stats.vram_bytes + block_bytes,
        },
        "verified": True,
    }
