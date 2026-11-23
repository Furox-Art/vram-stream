import json
from pathlib import Path

import numpy as np
import pytest

from vram_stream import cli


@pytest.fixture
def input_files(tmp_path):
    left = np.arange(1035, dtype=np.float32).reshape(23, 45)
    right = np.arange(45 * 56, dtype=np.float32).reshape(45, 56)
    paths = [tmp_path / f"{name}.npy" for name in ("left", "right", "result")]
    np.save(paths[0], left)
    np.save(paths[1], right)
    return paths, left @ right


def test_plan_without_gpu_or_source_files(capsys, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("plan must work without querying the GPU or opening matrix files")

    monkeypatch.setattr(cli, "_cuda_device", unexpected)
    monkeypatch.setattr(cli, "open_memmap", unexpected)
    assert (
        cli.main(
            [
                "plan",
                "--left",
                "80000",
                "40000",
                "--right",
                "40000",
                "60000",
                "--budget-mib",
                "64",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["output_shape"] == [80000, 60000]
    assert report["tile_bytes"] <= 64 * 2**20 < report["full_matrix_bytes"]


def test_matmul_and_overwrite(input_files, capsys):
    paths, expected = input_files
    args = ["matmul", *map(str, paths), "--device", "cpu", "--budget-mib", "1"]
    assert cli.main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["output"] == str(paths[2])
    assert report["device"] == "cpu"
    np.testing.assert_allclose(np.load(paths[2]), expected)
    paths[2].write_bytes(b"unchanged")
    assert cli.main(args) == 1
    assert paths[2].read_bytes() == b"unchanged"
    assert cli.main([*args, "--overwrite"]) == 0
    np.testing.assert_allclose(np.load(paths[2]), expected)


def test_failed_compute_preserves_output(input_files, capsys, monkeypatch):
    paths, _ = input_files
    paths[2].write_bytes(b"old content")

    def fail(*args, out, **kwargs):
        out[0, 0] = 1
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(cli, "matmul", fail)
    args = ["matmul", *map(str, paths), "--device", "cpu", "--overwrite"]
    assert cli.main(args) == 1
    assert "simulated failure" in capsys.readouterr().err
    assert paths[2].read_bytes() == b"old content"
    assert list(paths[2].parent.glob(".vram-stream-*")) == []


def test_existing_input_cannot_be_overwritten(input_files, capsys):
    paths, _ = input_files
    paths[2].hardlink_to(paths[0])
    expected = paths[0].read_bytes()
    assert cli.main(["matmul", *map(str, paths), "--overwrite"]) == 1
    assert "different file" in capsys.readouterr().err
    assert paths[0].read_bytes() == expected


def test_concurrent_destination_is_preserved(input_files, capsys, monkeypatch):
    paths, _ = input_files
    real_link = cli.os.link

    def create_first(source, destination):
        Path(destination).write_bytes(b"another writer")
        real_link(source, destination)

    monkeypatch.setattr(cli.os, "link", create_first)
    assert cli.main(["matmul", *map(str, paths), "--device", "cpu"]) == 1
    assert capsys.readouterr().err.startswith("vram-stream:")
    assert paths[2].read_bytes() == b"another writer"
