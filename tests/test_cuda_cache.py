import ctypes as ct
import os
import weakref
from contextlib import contextmanager
from functools import partial
from itertools import count
from types import SimpleNamespace

import numpy as np
import pytest
from test_cuda_workspace import SimulatedDriver, _set_pointer

from vram_stream import (
    CudaWorkspaceError,
    cuda_workspace,
    open_cuda_cache,
    torch_batches,
)
from vram_stream.cuda_benchmark import benchmark_cuda_cache


class _Queue:
    def __init__(self):
        self.tasks = []
        self.completed = 0
        self.running = False

    def flush(self, end=None):
        limit = len(self.tasks) if end is None else end
        while self.completed < limit:
            assert not self.running, "cyclic CUDA stream dependency"
            self.running = True
            try:
                self.tasks[self.completed]()
                self.completed += 1
            finally:
                self.running = False

    def wait_for(self, other):
        self.tasks.append(partial(other.flush, len(other.tasks)))


def _value(handle):
    return handle.value if hasattr(handle, "value") else handle


class _AsyncDriver(SimulatedDriver):
    """Defer DMA and kernel work until the relevant CUDA event completes."""

    free_bytes = 2**30
    pinned_limit = 2**30

    def __init__(self):
        super().__init__()
        self.handles = count(1)
        self.queues = {}
        self.events = {}
        self.allocations = {}

    def cuMemAlloc_v2(self, output, size):
        allocation = ct.create_string_buffer(size)
        address = ct.addressof(allocation)
        self.allocations[address] = allocation
        _set_pointer(output, ct.c_uint64, address)

    def cuMemFree_v2(self, pointer):
        del self.allocations[_value(pointer)]

    def cuMemGetInfo_v2(self, free, total):
        _set_pointer(free, ct.c_size_t, self.free_bytes)
        _set_pointer(total, ct.c_size_t, 2**30)

    def cuMemHostAlloc(self, output, size, flags):
        if size > self.pinned_limit:
            raise cuda_workspace.CudaOutOfMemoryError(
                "simulated pinned allocation failure"
            )
        super().cuMemHostAlloc(output, size, flags)

    def cuStreamCreate(self, output, flags):
        assert flags == 1
        handle = next(self.handles)
        self.queues[handle] = _Queue()
        _set_pointer(output, ct.c_void_p, handle)

    def cuStreamDestroy_v2(self, stream):
        queue = self.queues.pop(_value(stream))
        assert queue.completed == len(queue.tasks)

    def cuEventCreate(self, output, flags):
        assert flags & 2
        handle = next(self.handles)
        self.events[handle] = None
        _set_pointer(output, ct.c_void_p, handle)

    def cuEventDestroy_v2(self, event):
        del self.events[_value(event)]

    def cuEventRecord(self, event, stream):
        queue = self.queues[_value(stream)]
        self.events[_value(event)] = partial(queue.flush, len(queue.tasks))

    def cuEventSynchronize(self, event):
        self.events[_value(event)]()

    def cuStreamWaitEvent(self, stream, event, flags):
        assert flags == 0
        self.queues[_value(stream)].tasks.append(self.events[_value(event)])

    def cuStreamSynchronize(self, stream):
        self.queues[_value(stream)].flush()

    def cuCtxSynchronize(self):
        for queue in self.queues.values():
            queue.flush()

    def cuMemcpyHtoDAsync_v2(self, device, host, size, stream):
        self.queues[_value(stream)].tasks.append(
            partial(ct.memmove, device, host, size)
        )

    def cuMemsetD32_v2(self, device_pointer, value, length):
        span = (ct.c_uint32 * length).from_address(device_pointer)
        np.ctypeslib.as_array(span).fill(value)

    def cuModuleLoadData(self, output, ptx):
        assert b".entry add_words(" in ptx
        _set_pointer(output, ct.c_void_p, 19)

    def cuModuleGetFunction(self, output, module, name):
        assert name == b"add_words"
        _set_pointer(output, ct.c_void_p, 20)

    def cuLaunchKernel(
        self, function, gx, gy, gz, bx, by, bz, shared, stream, params, extra
    ):
        source, output, words = [ct.c_uint64.from_address(p).value for p in params]

        def add_words():
            address = source
            if address >= self.ram_device_address:
                address += ct.addressof(self.ram) - self.ram_device_address
            array_type = ct.c_uint32 * words
            left = np.ctypeslib.as_array(array_type.from_address(address))
            right = np.ctypeslib.as_array(array_type.from_address(output))
            np.add(left, right, out=right)

        self.queues[_value(stream)].tasks.append(add_words)

    def read_later(self, block, values):
        def consume():
            values.append(ct.string_at(block.device_pointer, block.size_bytes))

        self.queues[block.stream].tasks.append(consume)


@pytest.fixture
def async_drivers(monkeypatch):
    drivers = []

    def create():
        driver = _AsyncDriver()
        drivers.append(driver)
        return driver

    monkeypatch.setattr(cuda_workspace, "_CudaDriver", create)
    return drivers


def test_prefetch_respects_slots_and_reuses_resident_data(async_drivers):
    source = np.arange(11, dtype=np.uint8)
    seen = []
    with open_cuda_cache(source, block_bytes=4, slots=2) as cache:
        driver = async_drivers[0]
        with cache.read(0) as block:
            driver.read_later(block, seen)
            cache.prefetch(1)
        for index in (1, 0, 2):
            with cache.read(index) as block:
                driver.read_later(block, seen)
        cache.synchronize()
        assert cache.stats.h2d_copies == 3
        assert cache.stats.h2d_bytes == 11
        assert cache.stats.reused_reads == 1
        assert cache.stats.vram_bytes == cache.stats.pinned_ram_bytes == 8
    assert seen == [
        source[:4].tobytes(),
        source[4:8].tobytes(),
        source[:4].tobytes(),
        source[8:].tobytes(),
    ]
    assert not driver.allocations and not driver.events and not driver.queues
    with pytest.raises(CudaWorkspaceError, match="no longer active"):
        cache.prefetch(0)


def test_single_slot_waits_for_prior_dma_and_kernel(async_drivers):
    source = np.arange(12, dtype=np.uint8)
    seen = []
    with open_cuda_cache(source, block_bytes=4, slots=1) as cache:
        driver = async_drivers[0]
        for index in range(3):
            with cache.read(index) as block:
                driver.read_later(block, seen)
                if index == 0:
                    with pytest.raises(RuntimeError, match="active read leases"):
                        cache.prefetch(1)
        assert not seen or seen == [source[:4].tobytes()]
    assert seen == [chunk.tobytes() for chunk in source.reshape(3, 4)]


def test_invalidation_refreshes_source_safely(async_drivers):
    source = np.array([1, 2, 3, 4], dtype=np.uint8)
    seen = []
    with open_cuda_cache(source, block_bytes=4, slots=1) as cache:
        with cache.read(0) as block:
            async_drivers[0].read_later(block, seen)
            with pytest.raises(RuntimeError, match="active read leases"):
                cache.invalidate()
        cache.invalidate()
        source[:] = 7
        with cache.read(0) as block:
            async_drivers[0].read_later(block, seen)
    assert seen == [bytes([1, 2, 3, 4]), bytes([7, 7, 7, 7])]


def test_driver_error_prevents_reuse_and_cleans_up(async_drivers):
    with open_cuda_cache(b"data", slots=1) as cache:
        driver = async_drivers[0]
        driver.failure = "cuEventRecord"
        with pytest.raises(CudaWorkspaceError, match="simulated"):
            cache.prefetch(0)
        with pytest.raises(CudaWorkspaceError, match="no longer active"):
            cache.prefetch(0)
    assert not driver.allocations and not driver.events and not driver.queues


def test_closed_cache_releases_source_buffer(async_drivers):
    source = np.arange(4, dtype=np.uint8)
    lifetime = weakref.ref(source)
    with open_cuda_cache(source, block_bytes=4) as cache:
        cache.prefetch(0)
    del source
    assert lifetime() is None
    assert cache.block_count == 1


def test_cache_caps_slots_by_free_vram_and_pinned_budget(async_drivers, monkeypatch):
    monkeypatch.setattr(_AsyncDriver, "free_bytes", 100)
    data = np.arange(256, dtype=np.uint8)
    with open_cuda_cache(
        data, block_bytes=16, slots=10, pinned_budget_bytes=64
    ) as cache:
        assert cache.slot_count == 4
        assert cache.stats.vram_bytes == 64
        assert cache.block_count == 16
    with open_cuda_cache(data, block_bytes=16, slots=10) as cache:
        assert cache.slot_count == 5
    with (
        pytest.raises(cuda_workspace.CudaOutOfMemoryError, match="cache budget"),
        open_cuda_cache(data, block_bytes=128),
    ):
        pytest.fail("the requested block exceeds free VRAM")


def test_retries_release_partial_allocation_and_keep_block_size(
    async_drivers, monkeypatch
):
    monkeypatch.setattr(_AsyncDriver, "pinned_limit", 16)
    with open_cuda_cache(bytes(range(64)), block_bytes=16, slots=4) as cache:
        driver = async_drivers[0]
        assert len(driver.allocations) == 1
        assert cache.stats.oom_retries == 2
        assert cache.stats.pinned_ram_bytes == 16
        assert cache.block_count == 4
        with cache.read(1) as block:
            assert block.size_bytes == 16
    assert not driver.allocations


@pytest.mark.parametrize(("slots", "copied", "reused"), [(1, 384, 0), (2, 128, 4)])
def test_benchmark_agrees_with_numpy(async_drivers, slots, copied, reused):
    report = benchmark_cuda_cache(block_bytes=64, blocks=2, slots=slots, passes=3)
    assert report["verified"]
    assert report["logical_read_bytes"] == 384
    assert report["staged_vram"]["h2d_bytes"] == copied
    assert report["staged_vram"]["reused_reads"] == reused
    assert report["staged_vram"]["vram_bytes"] == (slots + 1) * 64


class _TorchStream:
    def __init__(self, queue):
        self.queue = queue

    def wait_stream(self, other):
        self.queue.wait_for(other.queue)


class _TorchSimulation:
    def __init__(self, drivers):
        self.drivers = drivers
        self.consumer = _TorchStream(_Queue())
        self.current = self.consumer
        self.cuda = SimpleNamespace(
            ExternalStream=self.external_stream,
            current_stream=lambda _: self.current,
            stream=self.stream,
        )
        self.version = SimpleNamespace(cuda="simulated")

    def external_stream(self, handle, device):
        driver = self.drivers[0]
        driver.queues[999] = self.consumer.queue
        return _TorchStream(driver.queues[handle])

    @contextmanager
    def stream(self, selected):
        previous = self.current
        self.current = selected
        try:
            yield
        finally:
            self.current = previous

    @staticmethod
    def device(name):
        return name

    @staticmethod
    def from_numpy(array):
        return array

    @staticmethod
    def as_tensor(view, *, device):
        return view.__cuda_array_interface__

    def empty(self, shape, *, dtype, device):
        assert self.current is self.consumer
        array = np.empty(shape, dtype=dtype)

        def copy_from(interface):
            assert self.current is not self.consumer
            pointer = interface["data"][0]
            assert shape == interface["shape"]
            self.current.queue.tasks.append(
                partial(ct.memmove, array.ctypes.data, pointer, array.nbytes)
            )

        return SimpleNamespace(array=array, copy_=copy_from)


def test_torch_adapter_keeps_batches_after_eviction(async_drivers, monkeypatch):
    torch = _TorchSimulation(async_drivers)
    monkeypatch.setattr(torch_batches, "_cuda_device", lambda _: (torch, 0))
    source = np.arange(30, dtype=np.float32).reshape(10, 3)
    with torch_batches.open_torch_batches(source, batch_rows=4, slots=2) as batches:
        kept = list(batches)
        assert batches.stats.h2d_copies == 3
    assert [batch.array.shape for batch in kept] == [(4, 3), (4, 3), (2, 3)]
    np.testing.assert_array_equal(
        np.concatenate([batch.array for batch in kept]), source
    )


@pytest.mark.skipif(
    os.environ.get("VRAM_STREAM_TEST_CUDA") != "1",
    reason="set VRAM_STREAM_TEST_CUDA=1 to run this on a CUDA device",
)
def test_benchmark_on_cuda_hardware():
    report = benchmark_cuda_cache(block_bytes=2**20, blocks=3, slots=2, passes=2)
    assert report["verified"]


@pytest.mark.skipif(
    os.environ.get("VRAM_STREAM_TEST_CUDA") != "1",
    reason="set VRAM_STREAM_TEST_CUDA=1 to run this on a CUDA device",
)
def test_torch_training_on_cuda_hardware():
    torch = pytest.importorskip("torch")
    torch.manual_seed(3)
    source = np.random.default_rng(3).normal(size=(11, 4)).astype(np.float32)
    model = torch.nn.Linear(3, 1).cuda()
    baseline = torch.nn.Linear(3, 1).cuda()
    baseline.load_state_dict(model.state_dict())

    def train(net, batches):
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
        for batch in batches:
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(net(batch[:, :3]), batch[:, 3:])
            loss.backward()
            optimizer.step()

    train(baseline, torch.from_numpy(source).cuda().split(4))
    with torch_batches.open_torch_batches(source, batch_rows=4, slots=2) as batches:
        train(model, batches)
    for actual, expected in zip(model.parameters(), baseline.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)
