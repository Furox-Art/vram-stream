# vram-stream

Open a CUDA workspace with a device-memory region and a system-RAM region.
Both regions are addressable by the same CUDA kernel. The system-RAM region
is pinned and mapped into the GPU's address space so kernel loads and stores
can reach it over the CPU–GPU link.

For repeated reads, the VRAM cache transfers input blocks in advance and keeps
recently used blocks on the GPU. A PyTorch adapter supplies cached training
batches and a benchmark measures both CUDA memory paths.

## Install

Use 64-bit Python 3.11 or later on Linux or Windows. The memory workspace
requires an NVIDIA driver and a CUDA device with mapped host memory support.

```bash
git clone https://github.com/Furox-Art/vram-stream.git
cd vram-stream
python -m pip install .
```

## Verify the memory connection

```bash
vram-stream memory --device 0 --vram-mib 16 --ram-mib 16
```

The command allocates both regions in the selected CUDA context. It writes
initial values and launches one GPU kernel that combines values from VRAM
and system RAM and writes results back to both. After synchronization it
checks the results and reports each region's size and their total size.
The check covers the first and last aligned 32-bit words of each region.
The CUDA driver compiles the bundled PTX kernel when the command runs.

## Use the workspace from Python

```python
from vram_stream import open_cuda_workspace

with open_cuda_workspace(vram_bytes=16 * 2**20, ram_bytes=16 * 2**20) as memory:
    vram_address = memory.vram_device_pointer
    ram_address = memory.ram_device_pointer
    # Pass both addresses and their lengths to a kernel in the current context.
    print(memory.gpu_accessible_bytes)
```

Each region has its own device address. Use the selected CUDA context on
the thread that opened the workspace. The CPU can access the mapped RAM
through `ram_host_pointer`. Synchronize the CUDA stream before touching that
RAM on the CPU. Both regions are released on exit and the calling thread's
previous CUDA context is restored.

Mapped RAM traffic crosses PCIe or NVLink, with the bandwidth and latency
of that connection. The driver and operating system limit how much RAM can
be pinned. [NVIDIA's mapped memory documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#mapped-memory)
describes the access and synchronization rules.

## Cache read-mostly data in VRAM

`open_cuda_cache` reads a C-contiguous buffer in fixed-size blocks. It
allocates one pinned RAM buffer and one VRAM buffer per slot. Copies and
kernels run in separate CUDA streams. CUDA events protect the pinned buffer
until its copy completes and the VRAM buffer until its last consumer finishes.
Blocks are evicted in least-recently-used order when space is needed.

```python
import numpy as np
from vram_stream import open_cuda_cache

source = np.arange(262144, dtype=np.float32)
with open_cuda_cache(source, block_bytes=256 * 1024, slots=2) as cache:
    with cache.read(0) as block:
        # Queue your kernel with block.device_pointer on block.stream.
        cache.prefetch(1)
    print(cache.stats)
```

Queue every access to a leased address on its supplied stream within
`cache.read()`. Keep source values stable or call `cache.invalidate()` before
editing them. Each slot uses at most `block_bytes` in each memory tier.
The source can also be a contiguous NumPy memory map.

## Prefetch PyTorch training batches

Use an NVIDIA CUDA build of [PyTorch](https://pytorch.org/get-started/locally/).
`open_torch_batches` feeds a dense, contiguous NumPy dataset in row order,
including the final partial batch. The cache can be reused across epochs.
Its adapter uses a device-to-device copy to give PyTorch ownership of each
returned batch. Training parameters, activations and optimizer state follow
the placement configured by the training program.

```python
import numpy as np
import torch
from vram_stream import open_torch_batches

features = np.random.default_rng(0).normal(size=(256, 8)).astype(np.float32)
dataset = np.column_stack((features, features.sum(axis=1)))
model = torch.nn.Linear(8, 1).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

with open_torch_batches(dataset, batch_rows=64, slots=2) as batches:
    for _ in range(2):
        for batch in batches:
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(batch[:, :-1]), batch[:, -1:])
            loss.backward()
            optimizer.step()
    print(batches.stats)
```

The two-tier cache budget covers its slots. PyTorch separately allocates
memory for returned batches and the training program. A training set smaller
than the cache can remain resident across passes. A larger set can benefit
from overlapping the next input transfer with work on the current batch.

## Measure the two CUDA paths

```bash
vram-stream benchmark --block-mib 16 --blocks 2 --slots 2 --passes 4
```

Both paths launch the same accumulation kernel on the same data. The tool
compares every output word with NumPy. It reports elapsed wall time through
GPU completion, bytes submitted for CPU-to-GPU copies, reused reads and
buffer sizes. The timer includes cache uploads and excludes initialization,
kernel warm-up and result readback. The direct path reads mapped RAM and
reports its logical workload size in the shared `logical_read_bytes` field.
That field describes requested reads, with hardware caching and interconnect
traffic left to GPU profiling. Results characterize this workload. Actual
training throughput depends on the model, reuse, memory budget and device.

### Comparison with related tools

| Tool | Documented scope or constraint | Choice in vram-stream |
| --- | --- | --- |
| DeepSpeed ZeRO | Parameter offload is a ZeRO stage 3 feature. Pinned parameter storage can encounter host memory-lock limits. [DeepSpeed configuration](https://www.deepspeed.ai/docs/config-json/#parameter-offloading) | Allocate a fixed number of pinned slots and provide CUDA access independently of a distributed training configuration. |
| PyTorch FSDP CPU offload | FSDP's CPU-offload path documents a gradient-accumulation constraint outside `no_sync()`. [FSDP limitations](https://docs.pytorch.org/docs/stable/fsdp.html#limitations) | Keep the adapter at the training input boundary and leave gradient accumulation to the training loop. |
| Hugging Face Accelerate Big Model Inference | Its CPU and disk weight-offload path documents bringing in weights as they are needed. [Big Model Inference limits](https://huggingface.co/docs/accelerate/main/en/concept_guides/big_model_inference#limits-and-further-development) | Let the caller prefetch the next block and reuse resident data within a fixed VRAM budget. |

## Blocked matrix multiplication

The matrix commands multiply host arrays or memory-mapped `.npy` files in
bounded tiles. To run them on a GPU, install the [PyTorch build](https://pytorch.org/get-started/locally/)
for your CUDA or ROCm setup. The default `auto` device selects an available GPU
or falls back to NumPy on the CPU. `--device cuda:0` selects a specific GPU,
including when using PyTorch on ROCm.

```bash
vram-stream info
vram-stream plan --left 80000 40000 --right 40000 60000 --budget-mib 64
```

`plan` reports the tile dimensions, bytes needed by the three tile buffers
and bytes needed to hold the full inputs and result. It works from shapes
alone and skips device discovery.

To run a small example, create two `.npy` inputs and multiply them.

```bash
python -c "import numpy as np; np.save('left.npy', np.arange(12, dtype=np.float32).reshape(3, 4)); np.save('right.npy', np.ones((4, 2), dtype=np.float32))"
vram-stream matmul left.npy right.npy result.npy --budget-mib 64
python -c "import numpy as np; print(np.load('result.npy'))"
```

The command maps both input files for reading and creates the result in the
output directory. It moves the completed result into place and returns a JSON
report with the device, tile plan, elapsed compute time and GPU memory retries.
Use `--overwrite` to replace an existing result. Creating a new result without
that option requires the output filesystem to support hard links.

### Python matrix API

```python
import numpy as np
from vram_stream import matmul

left = np.arange(12, dtype=np.float32).reshape(3, 4)
right = np.ones((4, 2), dtype=np.float32)
result = matmul(left, right, budget_bytes=64 * 2**20)

print(result.output)
print(result.device, result.plan.tile_bytes)
```

The inputs must be two-dimensional arrays with the same native-endian
`float32` or `float64` dtype. Pass an independent, writable contiguous array
or memory map as `out` to provide the destination. An exception can leave a
partially written `out` array.

### Matrix buffer sizing

For tile dimensions `r`, `k` and `c`, the GPU buffers use
`(r*k + k*c + r*c) * dtype.itemsize` bytes. The runtime uses at most 80%
of currently free VRAM for those buffers and reduces the tile size if
PyTorch reports a GPU allocation failure. Host staging uses another set of
the same size. The GPU driver, allocator and BLAS workspace have their own
memory costs.

File-backed arrays use the operating system's page cache and need disk space
for the inputs and the full output. Tile size controls the data buffers. Host
memory usage also depends on arrays held by the caller, the filesystem cache
and the numerical library. Smaller tiles can increase transfer and compute
overhead. Floating-point results can vary with dtype, device and tile size.
