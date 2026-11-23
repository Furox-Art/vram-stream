"""Stage frequently read host blocks into a bounded CUDA cache."""

import ctypes as ct
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from threading import get_ident

import numpy as np

from .cuda_workspace import CudaWorkspaceError, _allocate
from .planner import MIB, _integer

_NON_BLOCKING_STREAM = 1
_DISABLE_EVENT_TIMING = 2
_BLOCKING_EVENT_SYNC = 1


def _stream(cuda, cleanup: ExitStack) -> ct.c_void_p:
    stream = ct.c_void_p()
    cuda.call("cuStreamCreate", ct.byref(stream), _NON_BLOCKING_STREAM)
    cleanup.callback(cuda.call, "cuStreamDestroy_v2", stream)
    return stream


def _event(cuda, cleanup: ExitStack, flags: int) -> ct.c_void_p:
    event = ct.c_void_p()
    cuda.call("cuEventCreate", ct.byref(event), flags)
    cleanup.callback(cuda.call, "cuEventDestroy_v2", event)
    return event


@dataclass(frozen=True, slots=True)
class CachedBlock:
    """A read-only GPU view valid during its read lease."""

    index: int
    device_pointer: int
    size_bytes: int
    stream: int


@dataclass(frozen=True, slots=True)
class CacheStats:
    """Successful transfers and reads since the cache was opened."""

    h2d_copies: int
    h2d_bytes: int
    reads: int
    reused_reads: int
    vram_bytes: int
    pinned_ram_bytes: int


@dataclass(slots=True)
class _Slot:
    device_pointer: int
    host_pointer: int
    ready: ct.c_void_p
    consumed: ct.c_void_p
    uploaded: bool = False
    used: bool = False
    read_current_block: bool = False
    leases: int = 0


class CudaReadCache:
    """A host-thread-bound CUDA cache created by open_cuda_cache."""

    def __init__(
        self, cuda, workspace, source, block_bytes, slots, copy_stream, compute_stream
    ):
        self._cuda = cuda
        self._source = source
        self._block_bytes = block_bytes
        self._block_count = (source.nbytes + block_bytes - 1) // block_bytes
        self._blocks: OrderedDict[int, _Slot] = OrderedDict()
        self._free = slots
        self._copy = copy_stream
        self._compute = compute_stream
        self._thread = get_ident()
        self._closed = False
        self._failed = False
        self._uploads = self._bytes = self._reads = self._reuses = 0
        self._vram_bytes = workspace.vram_bytes
        self._pinned_bytes = workspace.ram_bytes
        self._slot_count = len(slots)

    @property
    def block_count(self) -> int:
        return self._block_count

    @property
    def slot_count(self) -> int:
        return self._slot_count

    @property
    def compute_stream(self) -> int:
        return self._compute.value

    @property
    def stats(self) -> CacheStats:
        return CacheStats(
            self._uploads,
            self._bytes,
            self._reads,
            self._reuses,
            self._vram_bytes,
            self._pinned_bytes,
        )

    def _check_active(self) -> None:
        if get_ident() != self._thread:
            raise RuntimeError("use the CUDA cache on the thread that opened it")
        if self._closed or self._failed:
            raise CudaWorkspaceError("the CUDA cache is no longer active")

    def _call(self, name, *arguments) -> None:
        try:
            self._cuda.call(name, *arguments)
        except CudaWorkspaceError:
            self._failed = True
            raise

    def _load(self, index: int) -> tuple[_Slot, int]:
        self._check_active()
        index = _integer(index, "block index", 0)
        if index >= self.block_count:
            raise IndexError(f"block {index} is outside the CUDA cache source")
        offset = index * self._block_bytes
        size = min(self._block_bytes, self._source.nbytes - offset)
        if index in self._blocks:
            self._blocks.move_to_end(index)
            return self._blocks[index], size

        if self._free:
            slot = self._free.pop()
        else:
            evicted = next(
                (key for key, value in self._blocks.items() if not value.leases), None
            )
            if evicted is None:
                raise RuntimeError("all CUDA cache slots have active read leases")
            slot = self._blocks.pop(evicted)

        if slot.uploaded:
            # The CPU can refill this pinned buffer after its previous DMA completes.
            self._call("cuEventSynchronize", slot.ready)
        ct.memmove(slot.host_pointer, self._source.ctypes.data + offset, size)
        if slot.used:
            # The DMA may overwrite VRAM only after a previous consumer has finished.
            self._call("cuStreamWaitEvent", self._copy, slot.consumed, 0)
        self._call(
            "cuMemcpyHtoDAsync_v2",
            slot.device_pointer,
            slot.host_pointer,
            size,
            self._copy,
        )
        self._call("cuEventRecord", slot.ready, self._copy)
        slot.uploaded = True
        slot.read_current_block = False
        self._blocks[index] = slot
        self._uploads += 1
        self._bytes += size
        return slot, size

    def prefetch(self, index: int) -> None:
        """Queue a block upload on the copy stream if it is missing."""
        self._load(index)

    @contextmanager
    def read(self, index: int) -> Iterator[CachedBlock]:
        """Lease a block and order its copy before work on its compute stream."""
        slot, size = self._load(index)
        self._call("cuStreamWaitEvent", self._compute, slot.ready, 0)
        self._reads += 1
        self._reuses += int(slot.read_current_block)
        slot.read_current_block = True
        slot.leases += 1
        try:
            yield CachedBlock(index, slot.device_pointer, size, self.compute_stream)
        finally:
            slot.leases -= 1
            self._call("cuEventRecord", slot.consumed, self._compute)
            slot.used = True

    def synchronize(self) -> None:
        """Wait for queued copies and consumers in this cache."""
        self._check_active()
        self._call("cuStreamSynchronize", self._copy)
        self._call("cuStreamSynchronize", self._compute)

    def invalidate(self) -> None:
        """Discard cached blocks before modifying the CPU source in place."""
        self._check_active()
        if any(slot.leases for slot in self._blocks.values()):
            raise RuntimeError(
                "finish active read leases before invalidating the cache"
            )
        self._free.extend(self._blocks.values())
        self._blocks.clear()


@contextmanager
def open_cuda_cache(
    source, *, block_bytes: int = 16 * MIB, slots: int = 2, device: int = 0
) -> Iterator[CudaReadCache]:
    """Open a cache over a contiguous CPU buffer such as a NumPy array or memmap.

    Submit GPU reads on the stream supplied by each lease. Keep the CPU source
    stable between invalidations and finish the lease before retaining any data
    in a separate allocation.
    """
    block_bytes = _integer(block_bytes, "block_bytes", 1)
    slots = _integer(slots, "slots", 1)
    try:
        data = np.frombuffer(source, dtype=np.uint8)
    except (TypeError, BufferError, ValueError) as error:
        raise ValueError("the source must expose a contiguous CPU buffer") from error
    if not data.nbytes:
        raise ValueError("the CUDA cache source must contain data")
    block_bytes = min(block_bytes, data.nbytes)
    count = min(slots, (data.nbytes + block_bytes - 1) // block_bytes)
    allocated = count * block_bytes

    with (
        _allocate(allocated, allocated, device) as (cuda, workspace),
        ExitStack() as cleanup,
    ):
        copy_stream = _stream(cuda, cleanup)
        compute_stream = _stream(cuda, cleanup)
        buffers = [
            _Slot(
                workspace.vram_device_pointer + index * block_bytes,
                workspace.ram_host_pointer + index * block_bytes,
                _event(cuda, cleanup, _DISABLE_EVENT_TIMING | _BLOCKING_EVENT_SYNC),
                _event(cuda, cleanup, _DISABLE_EVENT_TIMING),
            )
            for index in range(count)
        ]
        cache = CudaReadCache(
            cuda, workspace, data, block_bytes, buffers, copy_stream, compute_stream
        )
        cleanup.callback(cuda.call, "cuCtxSynchronize")
        try:
            yield cache
        finally:
            cache._closed = True
            cache._source = None
