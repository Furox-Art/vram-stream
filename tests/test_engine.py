from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from vram_stream import DeviceUnavailableError, engine, matmul, plan_matmul


def test_skewed_plan_fits_and_is_maximal():
    plan = plan_matmul((2, 150), (150, 100), dtype="float64", budget_bytes=4096)
    assert (plan.tile_rows, plan.tile_inner, plan.tile_columns) == (2, 20, 20)
    assert plan.tile_bytes == 3840
    assert plan.tile_bytes <= 4096 < (2 * 21 + 21 * 21 + 2 * 21) * 8
    assert plan.full_matrix_bytes == (2 * 150 + 150 * 100 + 2 * 100) * 8


@pytest.mark.parametrize("budget", [0, -1, 11])
def test_invalid_budget(budget):
    with pytest.raises(ValueError, match="budget_bytes"):
        plan_matmul((2, 3), (3, 2), budget_bytes=budget)


@pytest.mark.parametrize("budget", [True, np.bool_(True), 3.5])
def test_budget_requires_an_integer(budget):
    with pytest.raises(TypeError, match="budget_bytes"):
        plan_matmul((2, 3), (3, 2), budget_bytes=budget)


def test_incompatible_shapes():
    with pytest.raises(ValueError, match="inner dimensions"):
        plan_matmul((2, 3), (4, 2))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cpu_blocks_accept_strided_read_only_inputs(dtype):
    rng = np.random.default_rng(1)
    left = rng.normal(size=(18, 13)).astype(dtype)[::2, ::-1]
    right = rng.normal(size=(13, 22)).astype(dtype)[:, ::2]
    left.flags.writeable = False
    right.flags.writeable = False
    output = np.empty((9, 11), dtype=dtype, order="F")
    result = matmul(left, right, out=output, device="cpu", budget_bytes=256)
    assert result.output is output
    assert result.device == "cpu"
    assert result.plan.tile_bytes <= 256
    assert result.plan.tile_inner < left.shape[1]
    np.testing.assert_allclose(result.output, left @ right, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "shapes", [((3, 0), (0, 4)), ((0, 2), (2, 4)), ((3, 2), (2, 0))]
)
def test_empty_matrices(shapes):
    left, right = (np.empty(shape, dtype=np.float32) for shape in shapes)
    result = matmul(left, right, device="cpu", budget_bytes=1)
    assert result.plan.tile_bytes == 0
    np.testing.assert_array_equal(result.output, left @ right)


def test_input_and_output_validation(tmp_path):
    left = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="float32 or float64"):
        matmul(left.astype(np.int64), left, device="cpu")
    with pytest.raises(ValueError, match="same dtype"):
        matmul(left, left.astype(np.float64), device="cpu")
    with pytest.raises(ValueError, match="share memory"):
        matmul(left, left, out=left, device="cpu")
    path = tmp_path / "array.npy"
    np.save(path, left)
    source = np.load(path, mmap_mode="r")
    output = np.load(path, mmap_mode="r+")
    with pytest.raises(ValueError, match="map a file"):
        matmul(source, left, out=output, device="cpu")


def test_auto_cpu_when_torch_is_missing(monkeypatch):
    def unavailable(_):
        raise ModuleNotFoundError("torch")

    monkeypatch.setattr(engine, "importlib", SimpleNamespace(import_module=unavailable))
    values = np.eye(2, dtype=np.float32)
    assert matmul(values, values).device == "cpu"
    with pytest.raises(DeviceUnavailableError, match="PyTorch"):
        matmul(values, values, device="cuda:0")


class FakeTensor:
    """NumPy-backed tensor operations used to test the GPU loop on CI."""

    def __init__(self, array):
        self.array = array
        self.dtype = array.dtype

    def __getitem__(self, key):
        return FakeTensor(self.array[key])

    def copy_(self, source):
        np.copyto(self.array, source.array)

    def zero_(self):
        self.array.fill(0)

    def addmm_(self, left, right):
        self.array += left.array @ right.array


def test_gpu_blocks_and_allocator_retry(monkeypatch):
    class OutOfMemoryError(RuntimeError):
        pass

    allocations = []
    cleared = []

    def empty(shape, *, device, dtype):
        assert device == ("cuda", 1)
        if len(allocations) == 1 and not cleared:
            raise OutOfMemoryError("simulated OOM")
        array = np.empty(shape, dtype=dtype)
        allocations.append(array.nbytes)
        return FakeTensor(array)

    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        device=lambda _: nullcontext(),
        mem_get_info=lambda _: (500, 1000),
        OutOfMemoryError=OutOfMemoryError,
        empty_cache=lambda: cleared.append(True),
    )
    fake = SimpleNamespace(
        cuda=cuda,
        from_numpy=FakeTensor,
        empty=empty,
        device=lambda *args: args,
        inference_mode=nullcontext,
    )
    monkeypatch.setattr(
        engine, "importlib", SimpleNamespace(import_module=lambda _: fake)
    )
    rng = np.random.default_rng(2)
    left = rng.normal(size=(7, 9))[:, ::-1]
    left.flags.writeable = False
    right = rng.normal(size=(9, 5))
    result = matmul(left, right, device="cuda:1", budget_bytes=1024)
    assert result.device == "cuda:1"
    assert result.oom_retries == 1
    assert sum(allocations[1:]) == result.plan.tile_bytes <= 200
    np.testing.assert_allclose(result.output, left @ right, rtol=1e-12, atol=1e-12)


def test_gpu_hardware():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA or ROCm device")
    rng = np.random.default_rng(3)
    left = rng.normal(size=(37, 43))
    right = rng.normal(size=(43, 29))
    result = matmul(left, right, device="cuda:0", budget_bytes=1024)
    assert result.device == "cuda:0"
    assert result.plan.tile_bytes <= 1024
    np.testing.assert_allclose(result.output, left @ right, rtol=1e-10, atol=1e-10)
