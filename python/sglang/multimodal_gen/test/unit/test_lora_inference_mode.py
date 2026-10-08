import pytest
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.multimodal_gen.runtime.layers.linear import ReplicatedLinear
from sglang.multimodal_gen.runtime.layers.lora.linear import (
    LinearWithLoRA,
    _compute_lora_delta,
    wrap_with_lora_layer,
)


def test_stacked_lora_delta_preserves_projection_order():
    x = torch.tensor([[2.0, 3.0]])
    lora_a = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    lora_b = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])

    actual = _compute_lora_delta(x, lora_a, lora_b)

    torch.testing.assert_close(actual, torch.tensor([[2.0, 4.0, 9.0, 12.0]]))


def test_lora_merge_unmerge_handles_inference_base_weight():
    with torch.inference_mode():
        base_layer = nn.Linear(4, 3, bias=False)

    layer = LinearWithLoRA(base_layer, lora_rank=2, lora_alpha=2)
    base_weight = layer.cpu_weight.clone()

    assert layer.base_layer.weight.is_inference()
    assert not base_weight.is_inference()

    lora_a = torch.ones(2, 4)
    lora_b = torch.full((3, 2), 0.5)
    expected_merged = base_weight + lora_b @ lora_a

    with torch.inference_mode(False):
        layer.set_lora_weights(
            lora_a,
            lora_b,
            clear_existing=True,
            merge_weights=True,
        )

    assert layer.merged
    assert not layer.base_layer.weight.is_inference()
    assert torch.allclose(layer.base_layer.weight, expected_merged)

    with torch.inference_mode(False):
        layer.unmerge_lora_weights()

    assert not layer.merged
    assert not layer.base_layer.weight.is_inference()
    assert torch.allclose(layer.base_layer.weight, base_weight)


class _AtenMulCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func.overloadpacket is torch.ops.aten.mul:
            self.count += 1
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("kind", ["linear", "replicated"])
@pytest.mark.parametrize("strength", [1.0, 0.5])
def test_unit_lora_scale_multiplies_delta_at_most_once(kind, strength):
    """An unmerged LoRA forward multiplies the delta by its scale at most once.

    Every LoRA linear runs per denoising step, so each extra elementwise kernel
    per call slows LoRA stages (LTX-2 refinement by ~6% per multiply).

    (#43028 adapted to this stack: no forward_batch.runtime_lora_scale yet —
    that input arrived with #30487, which predates neither pick's base.)
    """
    base = (
        ReplicatedLinear(4, 3, bias=False)
        if kind == "replicated"
        else nn.Linear(4, 3, bias=False)
    )
    layer = wrap_with_lora_layer(base, lora_rank=2, lora_alpha=2)
    layer.set_lora_weights(
        torch.ones(2, 4), torch.ones(3, 2), strength=strength, merge_weights=False
    )
    x = torch.ones(2, 4)
    counter = _AtenMulCounter()
    with counter:
        layer(x)
    assert counter.count == (0 if strength == 1.0 else 1)
