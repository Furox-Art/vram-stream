# vram-stream

Multiply large NumPy matrices using system RAM and a bounded set of GPU
buffers. The inputs and result stay in host arrays or memory-mapped `.npy`
files. The GPU reads and multiplies tiles that fit the configured VRAM budget.

## Install

Use Python 3.11 or later.

```bash
git clone https://github.com/Furox-Art/vram-stream.git
cd vram-stream
python -m pip install .
```

For GPU execution, install the [PyTorch build](https://pytorch.org/get-started/locally/)
for your CUDA or ROCm setup. The default `auto` device uses an available GPU or
falls back to NumPy on the CPU. Select a specific GPU with `--device cuda:0`.
PyTorch uses the same `cuda` device names for ROCm.

## Command line

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

## Python

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

## Memory and performance

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
