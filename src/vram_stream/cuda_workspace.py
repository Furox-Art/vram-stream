"""Allocate VRAM and GPU-mapped system RAM in the same CUDA context."""

import ctypes as ct
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from ctypes import POINTER, c_char_p, c_int, c_size_t, c_uint, c_void_p
from dataclasses import dataclass
from importlib.resources import files

from .planner import _integer

_CAN_MAP_HOST_MEMORY = 19
_HOST_ALLOC_DEVICE_MAP = 0x02
_DEVICE_PTR = ct.c_uint64

_CUDA_ARGUMENTS = {
    "cuInit": (c_uint,),
    "cuGetErrorString": (c_int, POINTER(c_char_p)),
    "cuDeviceGet": (POINTER(c_int), c_int),
    "cuDeviceGetAttribute": (POINTER(c_int), c_int, c_int),
    "cuDevicePrimaryCtxRetain": (POINTER(c_void_p), c_int),
    "cuDevicePrimaryCtxRelease_v2": (c_int,),
    "cuCtxPushCurrent_v2": (c_void_p,),
    "cuCtxPopCurrent_v2": (POINTER(c_void_p),),
    "cuCtxSynchronize": (),
    "cuStreamCreate": (POINTER(c_void_p), c_uint),
    "cuStreamDestroy_v2": (c_void_p,),
    "cuStreamSynchronize": (c_void_p,),
    "cuStreamWaitEvent": (c_void_p, c_void_p, c_uint),
    "cuEventCreate": (POINTER(c_void_p), c_uint),
    "cuEventDestroy_v2": (c_void_p,),
    "cuEventRecord": (c_void_p, c_void_p),
    "cuEventSynchronize": (c_void_p,),
    "cuMemAlloc_v2": (POINTER(_DEVICE_PTR), c_size_t),
    "cuMemFree_v2": (_DEVICE_PTR,),
    "cuMemHostAlloc": (POINTER(c_void_p), c_size_t, c_uint),
    "cuMemHostGetDevicePointer_v2": (POINTER(_DEVICE_PTR), c_void_p, c_uint),
    "cuMemFreeHost": (c_void_p,),
    "cuModuleLoadData": (POINTER(c_void_p), c_char_p),
    "cuModuleGetFunction": (POINTER(c_void_p), c_void_p, c_char_p),
    "cuModuleUnload": (c_void_p,),
    "cuMemsetD32_v2": (_DEVICE_PTR, c_uint, c_size_t),
    "cuLaunchKernel": (
        c_void_p,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_void_p,
        POINTER(c_void_p),
        POINTER(c_void_p),
    ),
    "cuMemcpyDtoH_v2": (c_void_p, _DEVICE_PTR, c_size_t),
    "cuMemcpyHtoDAsync_v2": (_DEVICE_PTR, c_void_p, c_size_t, c_void_p),
}


class CudaWorkspaceError(RuntimeError):
    """The CUDA driver could not create or verify the memory workspace."""


class _CudaDriver:
    def __init__(self):
        if ct.sizeof(c_void_p) != 8:
            raise CudaWorkspaceError("CUDA workspaces require a 64-bit Python process")
        try:
            if sys.platform == "win32":
                library = ct.WinDLL("nvcuda.dll")
            elif sys.platform.startswith("linux"):
                library = ct.CDLL("libcuda.so.1")
            else:
                raise CudaWorkspaceError("CUDA workspaces support Linux and Windows")
        except OSError as error:
            raise CudaWorkspaceError(
                "The NVIDIA CUDA driver could not be loaded"
            ) from error
        self._calls = {}
        for name, arguments in _CUDA_ARGUMENTS.items():
            try:
                function = getattr(library, name)
            except AttributeError as error:
                raise CudaWorkspaceError(
                    f"The CUDA driver is missing {name}"
                ) from error
            function.argtypes = arguments
            function.restype = c_int
            self._calls[name] = function

    def call(self, name: str, *arguments) -> None:
        status = self._calls[name](*arguments)
        if status != 0:
            message = c_char_p()
            self._calls["cuGetErrorString"](status, ct.byref(message))
            detail = (
                message.value.decode(errors="replace")
                if message.value
                else "unknown CUDA error"
            )
            raise CudaWorkspaceError(f"{name} failed ({status}): {detail}")


@dataclass(frozen=True, slots=True)
class CudaWorkspace:
    """Addresses and sizes valid inside an open_cuda_workspace context."""

    device: int
    vram_bytes: int
    ram_bytes: int
    vram_device_pointer: int
    ram_device_pointer: int
    ram_host_pointer: int

    @property
    def gpu_accessible_bytes(self) -> int:
        """The combined size of the two GPU-addressable regions."""
        return self.vram_bytes + self.ram_bytes


@contextmanager
def _allocate(
    vram_bytes: int, ram_bytes: int, device: int
) -> Iterator[tuple[_CudaDriver, CudaWorkspace]]:
    vram_bytes = _integer(vram_bytes, "vram_bytes", 1)
    ram_bytes = _integer(ram_bytes, "ram_bytes", 1)
    device = _integer(device, "device", 0)
    for name, value, maximum in (
        ("vram_bytes", vram_bytes, sys.maxsize),
        ("ram_bytes", ram_bytes, sys.maxsize),
        ("device", device, 2**31 - 1),
    ):
        if value > maximum:
            raise ValueError(f"{name} must be at most {maximum}")

    cuda = _CudaDriver()
    cuda.call("cuInit", 0)
    selected = c_int()
    cuda.call("cuDeviceGet", ct.byref(selected), device)
    can_map = c_int()
    cuda.call("cuDeviceGetAttribute", ct.byref(can_map), _CAN_MAP_HOST_MEMORY, selected)
    if not can_map.value:
        raise CudaWorkspaceError(f"cuda:{device} does not support mapped host memory")

    with ExitStack() as cleanup:
        context = c_void_p()
        cuda.call("cuDevicePrimaryCtxRetain", ct.byref(context), selected)
        cleanup.callback(cuda.call, "cuDevicePrimaryCtxRelease_v2", selected)
        cuda.call("cuCtxPushCurrent_v2", context)
        cleanup.callback(cuda.call, "cuCtxPopCurrent_v2", ct.byref(c_void_p()))

        vram = _DEVICE_PTR()
        cuda.call("cuMemAlloc_v2", ct.byref(vram), vram_bytes)
        cleanup.callback(cuda.call, "cuMemFree_v2", vram)
        host = c_void_p()
        cuda.call("cuMemHostAlloc", ct.byref(host), ram_bytes, _HOST_ALLOC_DEVICE_MAP)
        cleanup.callback(cuda.call, "cuMemFreeHost", host)
        mapped_ram = _DEVICE_PTR()
        cuda.call("cuMemHostGetDevicePointer_v2", ct.byref(mapped_ram), host, 0)

        # Complete kernels before releasing either region and restore the caller's context.
        cleanup.callback(cuda.call, "cuCtxSynchronize")
        yield (
            cuda,
            CudaWorkspace(
                device, vram_bytes, ram_bytes, vram.value, mapped_ram.value, host.value
            ),
        )


@contextmanager
def open_cuda_workspace(
    *, vram_bytes: int, ram_bytes: int, device: int = 0
) -> Iterator[CudaWorkspace]:
    """Open two GPU-accessible regions backed by VRAM and pinned system RAM.

    The selected CUDA primary context is current on the calling thread inside
    the block. Synchronize GPU access before reading or updating the RAM region
    on the host. The addresses expire when the block exits.
    """
    with _allocate(vram_bytes, ram_bytes, device) as (_, workspace):
        yield workspace


def verify_cuda_workspace(*, vram_bytes: int, ram_bytes: int, device: int = 0) -> dict:
    """Verify GPU reads and writes to both regions in one kernel launch."""
    vram_bytes = _integer(vram_bytes, "vram_bytes", 4)
    ram_bytes = _integer(ram_bytes, "ram_bytes", 4)
    with _allocate(vram_bytes, ram_bytes, device) as (cuda, workspace):
        vram_tail = (vram_bytes - 4) // 4 * 4
        ram_tail = (ram_bytes - 4) // 4 * 4
        offsets = [(0, 0)]
        if vram_tail and ram_tail:
            offsets.append((vram_tail, ram_tail))

        for vram_offset, ram_offset in offsets:
            cuda.call(
                "cuMemsetD32_v2", workspace.vram_device_pointer + vram_offset, 11, 1
            )
            ct.c_uint32.from_address(workspace.ram_host_pointer + ram_offset).value = 7

        module = c_void_p()
        ptx = files("vram_stream").joinpath("verify_memory.ptx").read_bytes()
        cuda.call("cuModuleLoadData", ct.byref(module), ptx)
        with ExitStack() as cleanup:
            cleanup.callback(cuda.call, "cuModuleUnload", module)
            kernel = c_void_p()
            cuda.call("cuModuleGetFunction", ct.byref(kernel), module, b"verify_memory")
            argument_values = (
                _DEVICE_PTR(workspace.vram_device_pointer),
                _DEVICE_PTR(workspace.ram_device_pointer),
                _DEVICE_PTR(workspace.vram_device_pointer + vram_tail),
                _DEVICE_PTR(workspace.ram_device_pointer + ram_tail),
            )
            parameters = (c_void_p * 4)(
                *(ct.addressof(value) for value in argument_values)
            )
            cuda.call(
                "cuLaunchKernel",
                kernel,
                1,
                1,
                1,
                len(offsets),
                1,
                1,
                0,
                None,
                parameters,
                None,
            )
            cuda.call("cuCtxSynchronize")
            for vram_offset, ram_offset in offsets:
                vram_value = ct.c_uint32()
                cuda.call(
                    "cuMemcpyDtoH_v2",
                    ct.byref(vram_value),
                    workspace.vram_device_pointer + vram_offset,
                    4,
                )
                ram_value = ct.c_uint32.from_address(
                    workspace.ram_host_pointer + ram_offset
                )
                if (vram_value.value, ram_value.value) != (11 ^ 7, 11 + 7):
                    raise CudaWorkspaceError(
                        "GPU values did not match in both memory regions"
                    )

        return {
            "device": f"cuda:{workspace.device}",
            "vram_bytes": workspace.vram_bytes,
            "mapped_ram_bytes": workspace.ram_bytes,
            "gpu_accessible_bytes": workspace.gpu_accessible_bytes,
            "words_verified_per_region": len(offsets),
            "verified": True,
        }
