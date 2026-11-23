"""Asynchronously prefetch nested CPU DataLoader batches to CUDA."""

from collections.abc import Iterator, Mapping
from contextlib import closing, contextmanager
from dataclasses import dataclass

from .engine import _cuda_device
from .planner import _integer


def _map_tensors(torch, value, apply):
    if isinstance(value, torch.Tensor):
        return apply(value)
    if isinstance(value, Mapping):
        return {key: _map_tensors(torch, item, apply) for key, item in value.items()}
    if isinstance(value, tuple):
        items = tuple(_map_tensors(torch, item, apply) for item in value)
        return type(value)(*items) if hasattr(value, "_fields") else items
    if isinstance(value, list):
        return [_map_tensors(torch, item, apply) for item in value]
    return value


@dataclass(slots=True)
class _Transfer:
    source: object
    batch: object
    ready: object


@contextmanager
def prefetch_dataloader(loader, *, device: int = 0) -> Iterator[Iterator]:
    """Prefetch CPU tensors in CPU DataLoader batches on a dedicated CUDA stream.

    Batches can contain mappings, sequences and metadata. Use pinned batches
    to overlap copies with compute. Keep source tensors stable while the copy
    is in flight. The context owns the iterator and drains pending copies on
    exit, including when a loop ends early.
    """
    device = _integer(device, "device", 0)
    torch, _ = _cuda_device(f"cuda:{device}")
    target = torch.device(f"cuda:{device}")
    copy_stream = torch.cuda.Stream(device=target)
    inputs = iter(loader)

    def to_device(tensor):
        if tensor.device.type != "cpu":
            raise ValueError("prefetch_dataloader expects CPU input tensors")
        return tensor.to(device=target, non_blocking=True)

    def stage():
        try:
            source = next(inputs)
        except StopIteration:
            return None
        try:
            with torch.cuda.stream(copy_stream):
                result = _map_tensors(torch, source, to_device)
                ready = torch.cuda.Event()
                ready.record(copy_stream)
        except BaseException:
            # Keep the source alive until any partially queued transfer ends.
            copy_stream.synchronize()
            raise
        return _Transfer(source, result, ready)

    def iterate():
        pending = stage()
        while pending is not None:
            current = pending
            try:
                consumer = torch.cuda.current_stream(target)
                consumer.wait_event(current.ready)
                _map_tensors(
                    torch,
                    current.batch,
                    lambda tensor, stream=consumer: tensor.record_stream(stream),
                )
                pending = stage()
                yield current.batch
            finally:
                current.ready.synchronize()

    with closing(iterate()) as batches:
        try:
            yield batches
        finally:
            copy_stream.synchronize()
