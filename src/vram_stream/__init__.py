"""CUDA workspaces backed by GPU VRAM and mapped system RAM."""

from .cuda_cache import CachedBlock, CacheStats, CudaReadCache, open_cuda_cache
from .cuda_workspace import (
    CudaOutOfMemoryError,
    CudaWorkspace,
    CudaWorkspaceError,
    open_cuda_workspace,
)
from .dataloader import prefetch_dataloader
from .engine import DeviceUnavailableError, MatmulResult, matmul
from .planner import TilePlan, plan_matmul
from .torch_batches import TorchBatchSource, open_torch_batches
from .training import (
    create_offload_trainer,
    offload_activations,
    offload_model,
    zero3_cpu_offload_config,
)

__all__ = [
    "CacheStats",
    "CachedBlock",
    "CudaOutOfMemoryError",
    "CudaReadCache",
    "CudaWorkspace",
    "CudaWorkspaceError",
    "DeviceUnavailableError",
    "MatmulResult",
    "TilePlan",
    "TorchBatchSource",
    "create_offload_trainer",
    "matmul",
    "offload_activations",
    "offload_model",
    "open_cuda_cache",
    "open_cuda_workspace",
    "open_torch_batches",
    "plan_matmul",
    "prefetch_dataloader",
    "zero3_cpu_offload_config",
]
