import ctypes as ct
import json
import os

import pytest

from vram_stream import cli, cuda_workspace


def _set_pointer(output, pointer_type, value):
    ct.cast(output, ct.POINTER(pointer_type)).contents.value = value


class SimulatedDriver:
    """Exercise CUDA call ordering and distinct CPU and GPU RAM addresses."""

    ram_device_address = 2**48

    def __init__(self):
        self.calls = []
        self.failure = None
        self.can_map = 1
        self.write_results = True

    def call(self, name, *args):
        self.calls.append(name)
        if name == self.failure:
            raise cuda_workspace.CudaWorkspaceError(f"simulated {name} failure")
        handler = getattr(self, name, None)
        if handler:
            handler(*args)

    def cuDeviceGet(self, output, ordinal):
        _set_pointer(output, ct.c_int, ordinal)

    def cuDeviceGetAttribute(self, output, attribute, device):
        assert attribute == 19
        _set_pointer(output, ct.c_int, self.can_map)

    def cuDevicePrimaryCtxRetain(self, output, device):
        _set_pointer(output, ct.c_void_p, 1)

    def cuMemAlloc_v2(self, output, size):
        self.vram = ct.create_string_buffer(size)
        _set_pointer(output, ct.c_uint64, ct.addressof(self.vram))

    def cuMemHostAlloc(self, output, size, flags):
        assert flags == 0x02
        self.ram = ct.create_string_buffer(size)
        _set_pointer(output, ct.c_void_p, ct.addressof(self.ram))

    def cuMemHostGetDevicePointer_v2(self, output, host, flags):
        assert host.value == ct.addressof(self.ram)
        assert flags == 0
        _set_pointer(output, ct.c_uint64, self.ram_device_address)

    def cuModuleLoadData(self, output, ptx):
        assert b".visible .entry verify_memory(" in ptx
        _set_pointer(output, ct.c_void_p, 2)

    def cuModuleGetFunction(self, output, module, name):
        assert name == b"verify_memory"
        _set_pointer(output, ct.c_void_p, 3)

    def cuMemsetD32_v2(self, device_pointer, value, count):
        assert count == 1
        ct.c_uint32.from_address(device_pointer).value = value

    def cuLaunchKernel(
        self, function, gx, gy, gz, bx, by, bz, shared, stream, params, extra
    ):
        assert (gx, gy, gz, by, bz, shared) == (1, 1, 1, 1, 1, 0)
        if self.write_results:
            pointers = [ct.c_uint64.from_address(address).value for address in params]
            for lane in range(bx):
                ram_address = (
                    ct.addressof(self.ram)
                    + pointers[2 * lane + 1]
                    - self.ram_device_address
                )
                host = ct.c_uint32.from_address(ram_address)
                device = ct.c_uint32.from_address(pointers[2 * lane])
                host.value, device.value = (
                    host.value + device.value,
                    host.value ^ device.value,
                )

    def cuMemcpyDtoH_v2(self, output, device_pointer, size):
        ct.memmove(output, device_pointer, size)


@pytest.fixture
def driver(monkeypatch):
    simulated = SimulatedDriver()
    monkeypatch.setattr(cuda_workspace, "_CudaDriver", lambda: simulated)
    return simulated


def test_workspace_uses_two_addressable_regions(driver):
    with cuda_workspace.open_cuda_workspace(
        vram_bytes=40, ram_bytes=61, device=2
    ) as memory:
        assert memory.device == 2
        assert memory.gpu_accessible_bytes == 101
        assert memory.vram_device_pointer == ct.addressof(driver.vram)
        assert memory.ram_host_pointer == ct.addressof(driver.ram)
        assert memory.ram_device_pointer == driver.ram_device_address
    assert driver.calls[-5:] == [
        "cuCtxSynchronize",
        "cuMemFreeHost",
        "cuMemFree_v2",
        "cuCtxPopCurrent_v2",
        "cuDevicePrimaryCtxRelease_v2",
    ]


def test_mapping_failure_releases_both_allocations(driver):
    driver.failure = "cuMemHostGetDevicePointer_v2"
    with (
        pytest.raises(cuda_workspace.CudaWorkspaceError, match="simulated"),
        cuda_workspace.open_cuda_workspace(vram_bytes=16, ram_bytes=16),
    ):
        pytest.fail("mapping failed before either pointer could be used")
    assert driver.calls[-4:] == [
        "cuMemFreeHost",
        "cuMemFree_v2",
        "cuCtxPopCurrent_v2",
        "cuDevicePrimaryCtxRelease_v2",
    ]


def test_device_without_mapping_is_rejected_before_allocating(driver):
    driver.can_map = 0
    with (
        pytest.raises(cuda_workspace.CudaWorkspaceError, match="mapped host memory"),
        cuda_workspace.open_cuda_workspace(vram_bytes=16, ram_bytes=16),
    ):
        pytest.fail("the device cannot map RAM")
    assert driver.calls[-1] == "cuDeviceGetAttribute"


def test_zero_ram_budget_is_rejected_before_cuda(driver):
    with (
        pytest.raises(ValueError, match="ram_bytes"),
        cuda_workspace.open_cuda_workspace(vram_bytes=16, ram_bytes=0),
    ):
        pytest.fail("the RAM budget is invalid")
    assert driver.calls == []


def test_memory_command_verifies_writes_to_both_regions(driver, capsys):
    assert (
        cli.main(["memory", "--vram-mib", "1", "--ram-mib", "2", "--device", "2"]) == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report == {
        "device": "cuda:2",
        "vram_bytes": 2**20,
        "mapped_ram_bytes": 2 * 2**20,
        "gpu_accessible_bytes": 3 * 2**20,
        "words_verified_per_region": 2,
        "verified": True,
    }
    assert driver.calls.count("cuLaunchKernel") == 1
    assert driver.calls.count("cuMemcpyDtoH_v2") == 2


def test_probe_rejects_unchanged_words(driver):
    driver.write_results = False
    with pytest.raises(cuda_workspace.CudaWorkspaceError, match="did not match"):
        cuda_workspace.verify_cuda_workspace(vram_bytes=31, ram_bytes=40)
    assert "cuModuleUnload" in driver.calls
    assert "cuMemFreeHost" in driver.calls


@pytest.mark.skipif(
    os.environ.get("VRAM_STREAM_TEST_CUDA") != "1",
    reason="set VRAM_STREAM_TEST_CUDA=1 to run this on a CUDA device",
)
def test_mapped_ram_on_cuda_hardware():
    report = cuda_workspace.verify_cuda_workspace(vram_bytes=2**20, ram_bytes=2**20)
    assert report["verified"]
    assert report["words_verified_per_region"] == 2
