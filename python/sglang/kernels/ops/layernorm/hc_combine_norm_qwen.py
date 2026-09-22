"""Fused Qwen4-Exp hyper-connection combine + next-block grouped Gemma RMSNorm.

Derivation (S16), mirroring `GatedResidual.combine` (hyperconnection.py:280-327,
JIT path via kernels/ops/elementwise/hc_combine.py) immediately followed by the
CONSUMING block's `hc_norm` (``GroupedGemmaRMSNorm.forward``,
hyperconnection.py:45-74, JIT path via
kernels/ops/layernorm/grouped_gemma_rmsnorm.py). For ``hc_count = 4``,
``hidden_size = H`` and per-row slice ``R[m] = residual[m].view(4, H)``:

1. Gate (fp32 accumulate, hc_combine.cuh:64-108; the host-side formula is
   ``2 * sigmoid(F.linear(normed, inject_weight) / hc_count)``,
   hyperconnection.py:204-206 — computed INSIDE the kernel from the same
   ``inject_weight`` tensor the unfused kernel consumed; nothing is recomputed):

       a[m, c] = 2 * sigmoid(dot(normed_residual[m, :], inject_weight[c, :]) / hc_count)

2. Combine with a single bf16/fp16 rounding (hc_combine.cuh:112-130):

       updated[m, c*H + i] = round(fp32(residual) + a[m, c] * fp32(block_output))

3. Grouped Gemma RMSNorm (grouped_gemma_rmsnorm.cuh:56-116) of the ROUNDED
   ``updated`` values (the unfused path re-reads them from global memory; here
   the same rounded values are staged through shared memory), group = H, with
   the next block's ``weight`` (10240-wide, ``hc_per_branch_norm=True``) and eps:

       var[m, c]  = mean_i(fp32(updated[m, c*H + i])^2)          # fp32 tree, see .cuh
       inv[m, c]  = rsqrt(var[m, c] + eps)
       normalized[m, c*H + i] = round(fp32(updated) * inv * (1 + fp32(weight)))

Byte-parity: the fused kernel replicates both unfused reduction trees (gate
dot and per-group variance sum) node-for-node, so ``updated`` and
``normalized`` are bit-identical to the unfused composition. The bf16
round-trip between combine and norm is the numerics boundary; skipping it
would make ``spec_accept_length`` drift (plan S7 / S15 contract).

Packaged in the ``mhc_post_combine_norm_prefill.py`` idiom (JIT CUDA custom
op, preallocated outputs, ``mutates_args``). The DeepSeek-V4 Triton template
``hc_combine_norm.py`` was read for the launch-shape heuristic ONLY (its math
is a 4->1 collapse, not this 4-stream update); its arithmetic is cited here,
not copied.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_hc_combine_norm_qwen_module(
    hc_count: int, hidden_size: int, dtype: torch.dtype
) -> Module:
    """Compile and cache the fused combine+grouped-RMSNorm JIT module.

    Mirrors the validation style of ``hc_combine._jit_hc_combine_module`` and
    ``grouped_gemma_rmsnorm._jit_grouped_gemma_rmsnorm_module``: checks live
    under ``cache_once`` so they run once per (hc_count, hidden_size, dtype).
    """
    if dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(f"Unsupported dtype {dtype}. Supported: bfloat16, float16")
    if hidden_size <= 0 or hidden_size % 512 != 0 or hidden_size > 4096:
        # % 512: grouped_gemma_rmsnorm group constraint (cuh static_assert);
        # <= 4096: the norm tree's 160-thread shape must fit the 256-thread CTA.
        raise RuntimeError(
            f"Unsupported hidden_size {hidden_size}. Must be a multiple of 512 "
            "and at most 4096."
        )
    if hc_count <= 0 or hc_count % 4 != 0:
        raise RuntimeError(
            f"Unsupported hc_count {hc_count}. Must be a positive multiple of 4."
        )
    if (hc_count * hidden_size) % 2048 != 0:
        # hc_combine.cuh:140 — the row must map exactly onto 256 x 8-element vectors.
        raise RuntimeError(
            f"Unsupported hc_count * hidden_size {hc_count * hidden_size}. "
            "Must be a multiple of 2048."
        )
    args = make_cpp_args(hc_count, hidden_size, is_arch_support_pdl(), dtype)
    return load_jit(
        "hc_combine_norm_qwen",
        *args,
        cuda_files=["elementwise/hc_combine_norm_qwen.cuh"],
        cuda_wrappers=[
            ("hc_combine_norm_qwen", f"HcCombineNormQwenKernel<{args}>::run"),
        ],
    )


@register_custom_op(mutates_args=["updated", "normalized"])
def _hc_combine_norm_qwen_op(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    normed_residual: torch.Tensor,
    inject_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    updated: torch.Tensor,
    normalized: torch.Tensor,
    eps: float,
    hc_count: int,
    hidden_size: int,
) -> None:
    module = _jit_hc_combine_norm_qwen_module(hc_count, hidden_size, residual.dtype)
    module.hc_combine_norm_qwen(
        block_output,
        residual,
        normed_residual,
        inject_weight,
        norm_weight,
        updated,
        normalized,
        eps,
    )


def hc_combine_norm_qwen(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    normed_residual: torch.Tensor,
    inject_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    hc_count: int,
    hidden_size: int,
    eps: float = 1e-6,
    out: torch.Tensor | None = None,
    normed_out: torch.Tensor | None = None,
):
    """
    One-pass equivalent of ``hc_combine(...)`` followed by
    ``grouped_gemma_rmsnorm(updated, norm_weight, hidden_size, eps)``, emitting
    BOTH tensors (bit-identical to the unfused composition).

    Parameters
    ----------
    block_output    : CUDA tensor [..., hidden_size]
    residual        : CUDA tensor [..., hc_count * hidden_size]
    normed_residual : CUDA tensor, same shape/dtype as residual
    inject_weight   : CUDA tensor [hc_count, hc_count * hidden_size]
    norm_weight     : CUDA tensor [hc_count * hidden_size] (next block's hc_norm weight)
    hc_count        : number of hyper-connection branches
    hidden_size     : per-branch hidden size == norm group size
    eps             : RMSNorm epsilon
    out, normed_out : optional pre-allocated outputs (same shape/dtype as residual)

    Returns
    -------
    (updated, normalized), both same shape/dtype as residual.
    """
    row_size = hc_count * hidden_size
    y = block_output.reshape(-1, hidden_size)
    r = residual.reshape(-1, row_size)
    n = normed_residual.reshape(-1, row_size)
    if out is None:
        updated = torch.empty_like(r)
    else:
        updated = out.reshape(-1, row_size)
    if normed_out is None:
        normalized = torch.empty_like(r)
    else:
        normalized = normed_out.reshape(-1, row_size)

    if r.shape[0] == 0:
        return updated.reshape(residual.shape), normalized.reshape(residual.shape)

    _hc_combine_norm_qwen_op(
        y,
        r,
        n,
        inject_weight,
        norm_weight,
        updated,
        normalized,
        eps,
        hc_count,
        hidden_size,
    )
    return updated.reshape(residual.shape), normalized.reshape(residual.shape)
