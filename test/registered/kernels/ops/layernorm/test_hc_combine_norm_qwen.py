"""Tests for the fused Qwen4-Exp HC combine + grouped Gemma RMSNorm kernel (S16).

Primary oracle: the REAL unfused composition under the production dispatch
(hc_combine_split for <= 32 rows, hc_combine above; then
grouped_gemma_rmsnorm), compared BIT-FOR-BIT (torch.equal) on BOTH outputs.
Secondary oracle: a plain-PyTorch transcription of
GatedResidual._combine_compute and GroupedGemmaRMSNorm.forward
(hyperconnection.py) with the storage-dtype store/reload boundary between
combine and norm made explicit; compared by relative L2.
"""

import pytest
import torch
from sglang.kernels.ops.elementwise.hc_combine import hc_combine, hc_combine_split
from sglang.kernels.ops.layernorm.grouped_gemma_rmsnorm import grouped_gemma_rmsnorm
from sglang.kernels.ops.layernorm.hc_combine_norm_qwen import hc_combine_norm_qwen
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.cuda is None,
    reason="fused hc_combine_norm_qwen requires CUDA",
)

HC_COUNT = 4
HIDDEN_SIZE = 2560
ROW_SIZE = HC_COUNT * HIDDEN_SIZE
EPS = 1e-6

ROWS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 512, 4096]
DTYPES = [torch.bfloat16, torch.float16]


def _make_inputs(num_tokens, dtype, seed=0):
    torch.manual_seed(1000 * seed + num_tokens)
    block_output = torch.randn(num_tokens, HIDDEN_SIZE, dtype=dtype, device="cuda")
    residual = torch.randn(num_tokens, ROW_SIZE, dtype=dtype, device="cuda")
    normed_residual = torch.randn(num_tokens, ROW_SIZE, dtype=dtype, device="cuda")
    inject_weight = torch.randn(HC_COUNT, ROW_SIZE, dtype=dtype, device="cuda") * 0.02
    norm_weight = torch.randn(ROW_SIZE, dtype=dtype, device="cuda") * 0.02
    return block_output, residual, normed_residual, inject_weight, norm_weight


def _unfused_oracle(inp):
    """Production dispatch (hyperconnection.py:295-317) + the JIT norm path of
    GroupedGemmaRMSNorm.forward (hyperconnection.py:45-58) with group size =
    hidden_size (hc_per_branch_norm)."""
    block_output, residual, normed_residual, inject_weight, norm_weight = inp
    rows = residual.shape[0]
    if rows <= 32:
        updated = hc_combine_split(
            block_output,
            residual,
            normed_residual,
            inject_weight,
            HC_COUNT,
            HIDDEN_SIZE,
        )
    else:
        updated = hc_combine(
            block_output,
            residual,
            normed_residual,
            inject_weight,
            HC_COUNT,
            HIDDEN_SIZE,
        )
    normalized = grouped_gemma_rmsnorm(updated, norm_weight, HIDDEN_SIZE, EPS)
    return updated, normalized


def _torch_reference(inp, compute_dtype):
    """Plain-PyTorch transcription of GatedResidual._combine_compute
    (hyperconnection.py:200-217) then GroupedGemmaRMSNorm.forward's eager
    grouped path (hyperconnection.py:60-74). The store/reload rounding of
    `updated` through the storage dtype is explicit — the fused kernel must
    reproduce it because the unfused norm reads back rounded values.
    """
    block_output, residual, normed_residual, inject_weight, norm_weight = inp
    cd = compute_dtype
    R = residual.to(cd).unflatten(-1, (HC_COUNT, HIDDEN_SIZE))
    gate = normed_residual.to(cd) @ inject_weight.to(cd).transpose(0, 1)
    a = 2 * torch.sigmoid(gate / HC_COUNT)
    raw_new = (R + block_output.to(cd).unsqueeze(-2) * a.unsqueeze(-1)).flatten(-2)
    raw_for_norm = raw_new.to(residual.dtype).to(cd)
    xg = raw_for_norm.unflatten(-1, (HC_COUNT, HIDDEN_SIZE))
    variance = xg.pow(2).mean(dim=-1, keepdim=True)
    normed = (xg * torch.rsqrt(variance + EPS)).flatten(-2)
    normed = (normed * (1.0 + norm_weight.to(cd))).to(residual.dtype)
    return raw_new.to(residual.dtype), normed


def _rel_l2(a, b):
    return ((a.float() - b.float()).norm() / (a.float().norm() + 1e-12)).item()


# Measured worst-case rel-L2 vs the torch reference over rows x dtypes x
# {fp32, fp64} compute on SM120: updated 2.0e-5, normalized 1.1e-4 (the
# reference models the storage-dtype rounding boundary, so only the fp32
# accumulation tree differs). The 2e-2 assertion below is a loose guard.


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_tokens", ROWS)
def test_fused_matches_unfused_bit_for_bit(num_tokens, dtype):
    """Byte-parity contract (S16): BOTH outputs bit-identical to the unfused
    production composition, not merely close."""
    inp = _make_inputs(num_tokens, dtype)
    up_u, no_u = _unfused_oracle(inp)
    up_f, no_f = hc_combine_norm_qwen(*inp, HC_COUNT, HIDDEN_SIZE, EPS)
    assert up_f.shape == up_u.shape and up_f.dtype == dtype
    assert no_f.shape == no_u.shape and no_f.dtype == dtype
    assert torch.equal(up_f, up_u), (
        f"updated diverged: {(up_f != up_u).sum().item()}/{up_f.numel()} elems"
    )
    assert torch.equal(no_f, no_u), (
        f"normalized diverged: {(no_f != no_u).sum().item()}/{no_f.numel()} elems"
    )


@pytest.mark.parametrize("compute_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_tokens", [1, 8, 32, 48, 96, 512, 4096])
def test_fused_vs_torch_reference(num_tokens, dtype, compute_dtype):
    inp = _make_inputs(num_tokens, dtype)
    up_r, no_r = _torch_reference(inp, compute_dtype)
    up_f, no_f = hc_combine_norm_qwen(*inp, HC_COUNT, HIDDEN_SIZE, EPS)
    assert _rel_l2(up_f, up_r) <= 2e-2
    assert _rel_l2(no_f, no_r) <= 2e-2


@pytest.mark.parametrize("dtype", DTYPES)
def test_updated_independent_of_norm_weight(dtype):
    inp = _make_inputs(64, dtype)
    bo, res, nres, iw, _ = inp
    up1, no1 = hc_combine_norm_qwen(
        bo, res, nres, iw, torch.zeros_like(iw[0]), HC_COUNT, HIDDEN_SIZE, EPS
    )
    up2, no2 = hc_combine_norm_qwen(
        bo, res, nres, iw, torch.ones_like(iw[0]), HC_COUNT, HIDDEN_SIZE, EPS
    )
    assert torch.equal(up1, up2)
    assert not torch.equal(no1, no2)
    # and the w=1 run still equals the unfused oracle with the same weight
    up_u, no_u = _unfused_oracle((bo, res, nres, iw, torch.ones_like(iw[0])))
    assert torch.equal(up2, up_u) and torch.equal(no2, no_u)


def test_zero_rows():
    bo, res, nres, iw, nw = _make_inputs(8, torch.bfloat16)
    up, no = hc_combine_norm_qwen(
        bo[:0], res[:0], nres[:0], iw, nw, HC_COUNT, HIDDEN_SIZE, EPS
    )
    assert up.shape == (0, ROW_SIZE) and no.shape == (0, ROW_SIZE)
    assert up.dtype == torch.bfloat16 and no.dtype == torch.bfloat16


def test_rejects_unsupported_dtype():
    bo, res, nres, iw, nw = _make_inputs(4, torch.bfloat16)
    with pytest.raises(RuntimeError):
        hc_combine_norm_qwen(
            bo.float(),
            res.float(),
            nres.float(),
            iw.float(),
            nw.float(),
            HC_COUNT,
            HIDDEN_SIZE,
            EPS,
        )


@pytest.mark.parametrize("num_tokens", [1, 33, 100])
def test_preallocated_outputs_are_written(num_tokens):
    inp = _make_inputs(num_tokens, torch.bfloat16)
    up0, no0 = hc_combine_norm_qwen(*inp, HC_COUNT, HIDDEN_SIZE, EPS)
    out = torch.full_like(up0, float("nan"))
    normed_out = torch.full_like(no0, float("nan"))
    up1, no1 = hc_combine_norm_qwen(
        *inp, HC_COUNT, HIDDEN_SIZE, EPS, out=out, normed_out=normed_out
    )
    assert up1.data_ptr() == out.data_ptr()
    assert no1.data_ptr() == normed_out.data_ptr()
    assert torch.equal(up1, up0) and torch.equal(no1, no0)
    assert not out.isnan().any() and not normed_out.isnan().any()
