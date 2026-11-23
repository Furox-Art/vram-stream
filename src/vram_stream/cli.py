"""Command line interface for planning and running blocked matrix products."""

import argparse
import json
import math
import mmap
import os
import sys
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from numpy.lib.format import open_memmap, write_array_header_2_0

from .engine import _cuda_device, matmul
from .planner import DEFAULT_BUDGET_BYTES, MIB, TilePlan, plan_matmul


def _plan_report(plan: TilePlan) -> dict:
    return asdict(plan) | {"output_shape": plan.output_shape}


def _path(value: str) -> Path:
    path = Path(value).expanduser().absolute()
    if path.suffix.lower() != ".npy":
        raise ValueError(f"expected a .npy file path, got {path}")
    return path


def _write_product(
    path: Path,
    left: np.ndarray,
    right: np.ndarray,
    plan: TilePlan,
    device: str,
) -> dict:
    with path.open("w+b") as stream:
        write_array_header_2_0(
            stream,
            {
                "descr": left.dtype.str,
                "fortran_order": False,
                "shape": plan.output_shape,
            },
        )
        offset = stream.tell()
        stream.truncate(offset + math.prod(plan.output_shape) * left.dtype.itemsize)
        stream.flush()
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_WRITE) as mapping:
            output = np.ndarray(
                plan.output_shape, dtype=left.dtype, buffer=mapping, offset=offset
            )
            result = matmul(
                left, right, out=output, device=device, budget_bytes=plan.budget_bytes
            )
            report = {
                "device": result.device,
                "elapsed_seconds": result.elapsed_seconds,
                "oom_retries": result.oom_retries,
                "plan": _plan_report(result.plan),
            }
            mapping.flush()
    return report


def _run_product(args: argparse.Namespace) -> dict:
    left_path, right_path, destination = map(
        _path, (args.left, args.right, args.output)
    )
    for path in (left_path, right_path):
        if destination.resolve() == path.resolve() or (
            destination.exists() and path.exists() and destination.samefile(path)
        ):
            raise ValueError("output must be a different file from both inputs")
    if os.path.lexists(destination) and not args.overwrite:
        raise FileExistsError(
            f"{destination} already exists. Use --overwrite to replace it."
        )

    left, right = (open_memmap(path, mode="r") for path in (left_path, right_path))
    if left.dtype != right.dtype:
        raise ValueError("both input files must have the same dtype")
    plan = plan_matmul(
        left.shape, right.shape, dtype=left.dtype, budget_bytes=args.budget_mib * MIB
    )
    with TemporaryDirectory(
        prefix=".vram-stream-", dir=destination.parent
    ) as directory:
        temporary = Path(directory) / "result.npy"
        report = _write_product(temporary, left, right, plan, args.device)
        if args.overwrite:
            os.replace(temporary, destination)
        else:
            # A second writer may have created the destination during computation.
            os.link(temporary, destination)
    return {"output": str(destination)} | report


def _device_report() -> dict:
    gpu = _cuda_device("auto")
    if gpu is None:
        return {"default_device": "cpu", "gpu_backend": None, "gpus": []}
    torch, index = gpu
    gpus = []
    for number in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(number)
        gpus.append(
            {
                "device": f"cuda:{number}",
                "name": torch.cuda.get_device_name(number),
                "free_bytes": free,
                "total_bytes": total,
            }
        )
    return {
        "default_device": f"cuda:{index}",
        "gpu_backend": "rocm" if torch.version.hip else "cuda",
        "gpus": gpus,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vram-stream",
        description="Multiply host-backed matrices in memory-budgeted CPU or GPU tiles.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("info", help="Show the available computation devices.")
    plan = commands.add_parser(
        "plan", help="Estimate device buffers without loading any data."
    )
    plan.add_argument(
        "--left", nargs=2, required=True, type=int, metavar=("ROWS", "COLS")
    )
    plan.add_argument(
        "--right", nargs=2, required=True, type=int, metavar=("ROWS", "COLS")
    )
    plan.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    run = commands.add_parser(
        "matmul", help="Multiply two .npy files and write another .npy file."
    )
    run.add_argument("left")
    run.add_argument("right")
    run.add_argument("output")
    run.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:<index>")
    run.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output file."
    )
    for command in (plan, run):
        command.add_argument(
            "--budget-mib",
            type=int,
            default=DEFAULT_BUDGET_BYTES // MIB,
            help="Maximum size of the three tile buffers in MiB (default %(default)s).",
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "info":
            report = _device_report()
        elif args.command == "plan":
            report = _plan_report(
                plan_matmul(
                    args.left,
                    args.right,
                    dtype=args.dtype,
                    budget_bytes=args.budget_mib * MIB,
                )
            )
        else:
            report = _run_product(args)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"vram-stream: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0
