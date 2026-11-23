"""CPU offload for training with PyTorch FSDP2 and Hugging Face Trainer."""

import importlib
from contextlib import nullcontext

from .engine import DeviceUnavailableError
from .planner import _integer


def _pytorch():
    try:
        return importlib.import_module("torch")
    except (ImportError, OSError) as error:
        raise DeviceUnavailableError("install a working build of PyTorch") from error


def offload_activations(*, pin_memory: bool = True):
    """Save tensors needed by backward in CPU memory during the forward pass."""
    return _pytorch().autograd.graph.save_on_cpu(pin_memory=pin_memory)


def offload_model(
    model, *, layers=(), keep_forward_layers=(), mesh=None, pin_memory: bool = True
):
    """Apply FSDP2 CPU offload to a model and its selected layer groups in place.

    Choose a subset of ``layers`` with ``keep_forward_layers`` to keep their
    gathered parameters in VRAM between forward and backward. This avoids
    gathering them again during backward and uses more VRAM when they are held.
    Parameter shards and optimizer state still live in CPU memory.

    Initialize torch.distributed before calling this function. Create the
    optimizer from the wrapped model's parameters after this call.
    """
    torch = _pytorch()
    try:
        fsdp = importlib.import_module("torch.distributed.fsdp")
        offload_policy = fsdp.CPUOffloadPolicy(pin_memory=pin_memory)
        fully_shard = fsdp.fully_shard
    except (ImportError, AttributeError) as error:
        raise DeviceUnavailableError(
            "install PyTorch with FSDP2 CPUOffloadPolicy"
        ) from error
    if not torch.distributed.is_initialized():
        raise RuntimeError("initialize torch.distributed before applying FSDP2 offload")
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a PyTorch module")

    children = list(model.modules())
    positions = {id(layer): index for index, layer in enumerate(children)}
    selected = list(layers)
    identities = {id(layer) for layer in selected}
    if (
        len(identities) != len(selected)
        or id(model) in identities
        or not identities <= positions.keys()
    ):
        raise ValueError("layers must contain distinct submodules of model")
    kept = {id(layer) for layer in keep_forward_layers}
    if not kept <= identities:
        raise ValueError("keep_forward_layers must be a subset of layers")

    # FSDP2 wraps child groups before their parents.
    for layer in sorted(selected, key=lambda value: positions[id(value)], reverse=True):
        fully_shard(
            layer,
            mesh=mesh,
            offload_policy=offload_policy,
            reshard_after_forward=id(layer) not in kept,
        )
    return fully_shard(
        model, mesh=mesh, offload_policy=offload_policy, reshard_after_forward=True
    )


def zero3_cpu_offload_config(
    *, pin_memory: bool = True, bucket_elements: int = 2**23
) -> dict:
    """Create a DeepSpeed ZeRO-3 CPU-offload configuration for TrainingArguments."""
    bucket_elements = _integer(bucket_elements, "bucket_elements", 1)
    return {
        "bf16": {"enabled": "auto"},
        "fp16": {"enabled": "auto"},
        "zero_optimization": {
            "stage": 3,
            "offload_param": {"device": "cpu", "pin_memory": pin_memory},
            "offload_optimizer": {"device": "cpu", "pin_memory": pin_memory},
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": bucket_elements,
            "stage3_prefetch_bucket_size": bucket_elements,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "zero_allow_untested_optimizer": True,
        "zero_force_ds_cpu_optimizer": False,
        "gradient_clipping": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
        "gradient_accumulation_steps": "auto",
    }


def create_offload_trainer(*args, pin_activations: bool = True, **kwargs):
    """Create a Hugging Face Trainer that offloads saved forward tensors."""
    torch = _pytorch()
    try:
        from transformers import Trainer
    except (ImportError, OSError) as error:
        raise DeviceUnavailableError(
            "install Transformers and Accelerate to use Trainer"
        ) from error

    class ActivationOffloadTrainer(Trainer):
        def compute_loss(self, model, inputs, *loss_args, **loss_kwargs):
            context = (
                offload_activations(pin_memory=pin_activations)
                if model.training and torch.is_grad_enabled()
                else nullcontext()
            )
            with context:
                return super().compute_loss(model, inputs, *loss_args, **loss_kwargs)

    return ActivationOffloadTrainer(*args, **kwargs)
