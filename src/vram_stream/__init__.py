"""CUDA workspaces backed by GPU VRAM and mapped system RAM."""

from .cuda_cache import CachedBlock, CacheStats, CudaReadCache, open_cuda_cache
from .cuda_workspace import CudaWorkspace, CudaWorkspaceError, open_cuda_workspace
from .engine import DeviceUnavailableError, MatmulResult, matmul
from .planner import TilePlan, plan_matmul
from .torch_batches import TorchBatchSource, open_torch_batches

__all__ = [
    "CacheStats",
    "CachedBlock",
    "CudaReadCache",
    "CudaWorkspace",
    "CudaWorkspaceError",
    "DeviceUnavailableError",
    "MatmulResult",
    "TilePlan",
    "TorchBatchSource",
    "matmul",
    "open_cuda_cache",
    "open_cuda_workspace",
    "open_torch_batches",
    "plan_matmul",
]
