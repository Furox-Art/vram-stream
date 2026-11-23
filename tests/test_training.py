import json
import math
import os
import sys
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from time import perf_counter
from types import SimpleNamespace

import pytest

from vram_stream import (
    create_offload_trainer,
    offload_activations,
    offload_model,
    prefetch_dataloader,
    training,
    zero3_cpu_offload_config,
)
from vram_stream.dataloader import _map_tensors


def test_zero3_profile_offloads_model_and_optimizer():
    config = zero3_cpu_offload_config(pin_memory=False, bucket_elements=4096)
    zero = config["zero_optimization"]
    assert zero["stage"] == 3
    assert (
        zero["offload_param"]
        == zero["offload_optimizer"]
        == {"device": "cpu", "pin_memory": False}
    )
    assert zero["stage3_prefetch_bucket_size"] == 4096
    assert config["gradient_accumulation_steps"] == "auto"
    with pytest.raises(ValueError, match="bucket_elements"):
        zero3_cpu_offload_config(bucket_elements=0)


def test_fsdp_wraps_children_before_root(monkeypatch):
    class Module:
        def __init__(self, *children):
            self.children = children

        def modules(self):
            yield self
            for child in self.children:
                yield from child.modules()

    child = Module()
    group = Module(child)
    model = Module(group)
    calls = []

    def shard(module, **kwargs):
        calls.append((module, kwargs))
        return module

    torch = SimpleNamespace(
        nn=SimpleNamespace(Module=Module),
        distributed=SimpleNamespace(is_initialized=lambda: True),
    )
    fsdp = SimpleNamespace(
        CPUOffloadPolicy=lambda *, pin_memory: ("cpu", pin_memory),
        fully_shard=shard,
    )
    monkeypatch.setattr(training, "_pytorch", lambda: torch)
    monkeypatch.setattr(
        training, "importlib", SimpleNamespace(import_module=lambda _: fsdp)
    )
    assert offload_model(model, layers=(group, child), mesh="mesh") is model
    assert [module for module, _ in calls] == [child, group, model]
    assert all(
        options
        == {
            "mesh": "mesh",
            "offload_policy": ("cpu", True),
            "reshard_after_forward": True,
        }
        for _, options in calls
    )
    with pytest.raises(ValueError, match="submodules"):
        offload_model(model, layers=(Module(),))
    assert len(calls) == 3
    calls.clear()
    with pytest.raises(ValueError, match="keep_forward_layers"):
        offload_model(model, layers=(group,), keep_forward_layers=(child,))
    assert not calls
    offload_model(model, layers=(group, child), keep_forward_layers=(child,))
    assert [module for module, _ in calls] == [child, group, model]
    assert [options["reshard_after_forward"] for _, options in calls] == [
        False,
        True,
        True,
    ]
    calls.clear()
    with pytest.raises(ValueError, match="resident_layers"):
        offload_model(model, layers=(group,), resident_layers=(child,))
    assert not calls
    offload_model(
        model,
        layers=(group, child),
        resident_layers=(child,),
        keep_forward_layers=(child,),
    )
    assert calls[0] == (child, {"mesh": None, "reshard_after_forward": False})
    assert [module for module, _ in calls] == [child, group, model]
    assert all(options["offload_policy"] == ("cpu", True) for _, options in calls[1:])


def test_trainer_enters_activation_scope_only_in_training(monkeypatch):
    entered = []

    @contextmanager
    def cpu_save(*, pin_memory):
        entered.append(pin_memory)
        try:
            yield
        finally:
            entered.pop()

    class Trainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def compute_loss(self, model, inputs, **kwargs):
            return list(entered), inputs, kwargs

    torch = SimpleNamespace(
        is_grad_enabled=lambda: True,
        autograd=SimpleNamespace(graph=SimpleNamespace(save_on_cpu=cpu_save)),
    )
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(Trainer=Trainer))
    monkeypatch.setattr(training, "_pytorch", lambda: torch)
    trainer = create_offload_trainer(args="original_args")
    model = SimpleNamespace(training=True)
    assert trainer.compute_loss(model, "inputs", num_items_in_batch=4) == (
        [True],
        "inputs",
        {"num_items_in_batch": 4},
    )
    assert entered == []
    model.training = False
    assert trainer.compute_loss(model, "inputs")[0] == []


def test_nested_batch_mapping_keeps_metadata_and_tuple_types():
    class Tensor:
        pass

    class Entry(tuple):
        _fields = ("data", "label")

        def __new__(cls, data, label):
            return tuple.__new__(cls, (data, label))

    tensor = Tensor()
    source = {"pack": [Entry(tensor, "row"), (None, tensor)], "count": 2}
    result = _map_tensors(
        SimpleNamespace(Tensor=Tensor), source, lambda _: "device tensor"
    )
    assert result == {
        "pack": [("device tensor", "row"), (None, "device tensor")],
        "count": 2,
    }
    assert isinstance(result["pack"][0], Entry)


@pytest.mark.skipif(
    os.environ.get("VRAM_STREAM_TEST_CUDA") != "1",
    reason="set VRAM_STREAM_TEST_CUDA=1 for NVIDIA GPU tests",
)
def test_nested_dataloader_on_cuda_hardware():
    torch = pytest.importorskip("torch")
    items = [
        {
            "values": (torch.arange(3) + number, {"label": torch.tensor(number)}),
            "name": f"row-{number}",
        }
        for number in range(5)
    ]
    loader = torch.utils.data.DataLoader(items, batch_size=2, pin_memory=True)
    expected = list(loader)
    with prefetch_dataloader(loader) as batches:
        actual = list(batches)
    assert len(actual) == len(expected) == 3
    for received, reference in zip(actual, expected, strict=True):
        assert received["values"][0].is_cuda
        assert received["values"][1]["label"].is_cuda
        assert received["name"] == reference["name"]
        torch.testing.assert_close(received["values"][0].cpu(), reference["values"][0])
        torch.testing.assert_close(
            received["values"][1]["label"].cpu(), reference["values"][1]["label"]
        )
    with prefetch_dataloader(loader) as batches:
        next(batches)


@pytest.mark.skipif(
    os.environ.get("VRAM_STREAM_TEST_CUDA") != "1",
    reason="set VRAM_STREAM_TEST_CUDA=1 for NVIDIA GPU tests",
)
def test_training_offload_on_cuda_hardware(tmp_path):
    torch = pytest.importorskip("torch")
    from torch.distributed.device_mesh import init_device_mesh

    torch.cuda.set_device(0)
    torch.manual_seed(8)
    width = 768
    model = torch.nn.Sequential(
        *[
            torch.nn.Sequential(torch.nn.Linear(width, width), torch.nn.GELU())
            for _ in range(4)
        ]
    )
    inputs = torch.randn(16, width, device="cuda")
    targets = torch.randn_like(inputs)
    warmup_steps = 4
    steps = 24

    def train(network, *, activation_offload, check_forward=None):
        optimizer = torch.optim.AdamW(network.parameters(), lr=0.001, foreach=False)

        def step(check_layout=None):
            optimizer.zero_grad(set_to_none=True)
            with offload_activations() if activation_offload else nullcontext():
                loss = torch.nn.functional.mse_loss(network(inputs), targets)
            if check_layout is not None:
                check_layout()
            loss.backward()
            optimizer.step()
            return loss.detach()

        step(check_forward)
        for _ in range(warmup_steps - 1):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        losses = [step() for _ in range(steps)]
        torch.cuda.synchronize()
        stats = {
            "ms_per_step": round((perf_counter() - start) * 1000 / steps, 3),
            "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20, 3),
        }
        return torch.stack(losses).cpu(), stats, optimizer

    baseline = deepcopy(model).cuda()
    reference_loss, reference_stats, reference_optimizer = train(
        baseline, activation_offload=False
    )
    reference_weights = [value.detach().cpu() for value in baseline.parameters()]
    del baseline, reference_optimizer
    torch.cuda.empty_cache()

    def train_offloaded(mesh, *, keep_last, resident_last=0):
        network = deepcopy(model)
        layers = list(network.children())
        kept = layers[len(layers) - keep_last :]
        resident = layers[len(layers) - resident_last :]
        offload_model(
            network,
            layers=layers,
            keep_forward_layers=kept,
            resident_layers=resident,
            mesh=mesh,
        )

        def check_forward():
            for layer in layers:
                parameter = next(layer.parameters())
                if layer in kept or layer in resident:
                    assert parameter.device.type == "cuda"
                else:
                    assert parameter.to_local().device.type == "cpu"

        loss, stats, optimizer = train(
            network, activation_offload=True, check_forward=check_forward
        )
        torch.testing.assert_close(loss, reference_loss, rtol=1e-4, atol=1e-5)
        placed_parameters = (
            (parameter, "cuda" if layer in resident else "cpu")
            for layer in layers
            for parameter in layer.parameters()
        )
        for (actual, device), expected in zip(
            placed_parameters, reference_weights, strict=True
        ):
            local = actual.to_local()
            assert local.device.type == device
            assert actual.grad.to_local().device.type == device
            state = optimizer.state[actual]
            assert state["exp_avg"].to_local().device.type == device
            assert state["exp_avg_sq"].to_local().device.type == device
            torch.testing.assert_close(
                local.cpu(), expected, rtol=1e-4, atol=2e-6
            )
        return stats

    torch.distributed.init_process_group(
        backend="nccl",
        rank=0,
        world_size=1,
        init_method=(tmp_path / "torch-rendezvous").as_uri(),
    )
    try:
        mesh = init_device_mesh("cuda", (1,))
        offload_stats = train_offloaded(mesh, keep_last=0)
        torch.cuda.empty_cache()
        retained_stats = train_offloaded(mesh, keep_last=2)
        torch.cuda.empty_cache()
        mixed_stats = train_offloaded(mesh, keep_last=2, resident_last=2)
        print(
            json.dumps(
                {
                    "gpu": torch.cuda.get_device_name(0),
                    "steps": steps,
                    "warmup_steps": warmup_steps,
                    "baseline": reference_stats,
                    "cpu_offload": offload_stats,
                    "cpu_offload_keep_last_two": retained_stats,
                    "two_gpu_resident_groups": mixed_stats,
                    "matching_training": True,
                }
            )
        )
    finally:
        torch.distributed.destroy_process_group()


def _tiny_trainer_inputs(torch):
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 1)

        def forward(self, features, labels):
            output = self.linear(features)
            return {
                "loss": torch.nn.functional.mse_loss(output, labels),
                "logits": output,
            }

    dataset = [
        {"features": torch.ones(2) * number, "labels": torch.ones(1) * number}
        for number in range(8)
    ]
    return TinyModel(), dataset


@pytest.mark.skipif(
    os.environ.get("VRAM_STREAM_TEST_TRAINER") != "1",
    reason="set VRAM_STREAM_TEST_TRAINER=1 to exercise installed Trainer dependencies",
)
def test_real_hugging_face_trainer(tmp_path):
    import torch
    from transformers import Trainer, TrainingArguments

    def arguments(name):
        return TrainingArguments(
            output_dir=str(tmp_path / name),
            max_steps=2,
            report_to=[],
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            dataloader_pin_memory=torch.cuda.is_available(),
            save_strategy="no",
            disable_tqdm=True,
            seed=7,
        )

    torch.manual_seed(7)
    initial, dataset = _tiny_trainer_inputs(torch)
    baseline = Trainer(
        model=deepcopy(initial), args=arguments("baseline"), train_dataset=dataset
    )
    trainer = create_offload_trainer(
        model=deepcopy(initial),
        args=arguments("offload"),
        train_dataset=dataset,
        pin_activations=torch.cuda.is_available(),
    )
    expected = baseline.train()
    actual = trainer.train()
    assert trainer.state.global_step == 2
    assert math.isfinite(actual.training_loss)
    assert actual.training_loss == pytest.approx(expected.training_loss)
    for reference, updated in zip(
        baseline.model.parameters(), trainer.model.parameters(), strict=True
    ):
        torch.testing.assert_close(reference, updated)
    assert any(
        not torch.equal(first.detach().cpu(), updated.detach().cpu())
        for first, updated in zip(
            initial.parameters(), trainer.model.parameters(), strict=True
        )
    )
    print(
        json.dumps(
            {
                "device": str(trainer.args.device),
                "steps": actual.global_step,
                "trainer_loss": actual.training_loss,
                "matching_weights": True,
            }
        )
    )


@pytest.mark.skipif(
    os.environ.get("VRAM_STREAM_TEST_DEEPSPEED") != "1",
    reason="set VRAM_STREAM_TEST_DEEPSPEED=1 and launch with torchrun for a CUDA ZeRO-3 check",
)
def test_real_deepspeed_zero3_on_cuda(tmp_path):
    import torch
    from transformers import TrainingArguments

    assert torch.cuda.is_available(), "the ZeRO-3 check requires a CUDA device"
    initialized_before = torch.distributed.is_initialized()
    try:
        profile = zero3_cpu_offload_config(bucket_elements=4096)
        profile["zero_optimization"]["stage3_param_persistence_threshold"] = 0
        args = TrainingArguments(
            output_dir=str(tmp_path / "zero3"),
            deepspeed=profile,
            max_steps=2,
            report_to=[],
            per_device_train_batch_size=2,
            save_strategy="no",
            disable_tqdm=True,
            seed=7,
        )
        model, dataset = _tiny_trainer_inputs(torch)
        trainer = create_offload_trainer(model=model, args=args, train_dataset=dataset)
        result = trainer.train()
        engine = trainer.model_wrapped
        assert engine.zero_optimization_stage() == 3
        assert result.global_step == 2 and math.isfinite(result.training_loss)
        parameters = list(engine.module.parameters())
        assert parameters and all(
            parameter.ds_tensor.device.type == "cpu" for parameter in parameters
        )
        shards = engine.optimizer.fp32_partitioned_groups_flat
        assert shards and all(shard.device.type == "cpu" for shard in shards)
        print(
            json.dumps(
                {
                    "device": str(trainer.args.device),
                    "steps": result.global_step,
                    "zero_stage": 3,
                    "cpu_parameter_shards": len(parameters),
                    "cpu_optimizer_shards": len(shards),
                }
            )
        )
    finally:
        if not initialized_before and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
