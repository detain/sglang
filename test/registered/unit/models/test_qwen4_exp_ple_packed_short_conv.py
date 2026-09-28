"""The packed PLE short conv must match the padded one.

The padded extend path materializes rows x longest-row x channels; a
mixed-chunk batch (one long prefill plus many one-token decode tails) made
that dozens of times larger than the tokens and OOMed. The packed path runs
one conv1d over flat per-row [state | tokens] segments; outputs, the
committed conv state and the radix-track snapshot must be unchanged.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.models import qwen4_exp  # noqa: E402

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

CHANNELS = 16
KERNEL = 4
DILATION = 3
STATE_LEN = (KERNEL - 1) * DILATION
NUM_SLOTS = 64


def _layer(device, dtype):
    conv1d = torch.nn.Conv1d(
        CHANNELS,
        CHANNELS,
        KERNEL,
        groups=CHANNELS,
        bias=False,
        dilation=DILATION,
    ).to(device=device, dtype=dtype)
    layer = SimpleNamespace(
        layer_id=0,
        conv1d=conv1d,
        short_conv_state_len=STATE_LEN,
        short_conv_dilation=DILATION,
        conv_channels=CHANNELS,
    )
    layer._short_conv_packed = (
        lambda *a: qwen4_exp.Qwen4ExpPLELayer._short_conv_packed(layer, *a)
    )
    return layer


def _batch(lengths, device, pad_tokens=0):
    lengths_t = torch.tensor(lengths, device=device)
    processed = sum(lengths) + pad_tokens
    positions = torch.arange(processed, device=device)
    starts = torch.cat([lengths_t.new_zeros(1), torch.cumsum(lengths_t, 0)])
    req = (torch.searchsorted(starts, positions, right=True) - 1).clamp(
        0, len(lengths) - 1
    )
    offsets = positions - starts.index_select(0, req)
    return qwen4_exp._PLEBatch(
        mode=ForwardMode.MIXED,
        use_decode_fast_path=False,
        physical_tokens=processed,
        processed_tokens=processed,
        lengths=lengths_t,
        row_width=max(lengths),
        req_indices=req,
        token_offsets=offsets,
        valid_tokens=offsets < lengths_t.index_select(0, req),
        state_indices=torch.arange(1, len(lengths) + 1, device=device),
        ngram_context=None,
        ngram_eos_token_id=None,
    )


def _forward_batch(lengths, device):
    # Row 0 (the prefill) snapshots mid-chunk; the tails are untracked.
    rows = len(lengths)
    mask = torch.zeros(rows, dtype=torch.bool, device=device)
    mask[0] = True
    aligned = torch.zeros(rows, dtype=torch.long, device=device)
    aligned[0] = lengths[0] // 2
    return SimpleNamespace(
        mamba_track_indices=torch.arange(40, 40 + rows, device=device),
        mamba_track_mask=mask,
        mamba_track_aligned_lens=lambda: aligned,
    )


def _run(packed, lengths, device, dtype, pad_tokens=0):
    torch.manual_seed(0)
    layer = _layer(device, dtype)
    conv_state = torch.randn(NUM_SLOTS, CHANNELS, STATE_LEN, device=device, dtype=dtype)
    batch = _batch(lengths, device, pad_tokens)
    x = torch.randn(batch.processed_tokens, CHANNELS, device=device, dtype=dtype)
    pool = SimpleNamespace(short_conv_layer_cache=lambda layer_id: conv_state)
    with (
        patch.object(qwen4_exp, "get_req_to_token_pool", lambda: pool),
        patch.object(qwen4_exp, "_ple_padding_is_wasteful", lambda b: packed),
    ):
        out = qwen4_exp.Qwen4ExpPLELayer._short_conv(
            layer, x, _forward_batch(lengths, device), batch
        )
    return out[batch.valid_tokens], conv_state


class TestPackedShortConv(CustomTestCase):
    def _check(self, device, dtype, atol):
        # One long prefill chunk plus a dozen one-token decode tails, with a
        # couple of padding tokens past the last row.
        lengths = [300] + [1] * 12 + [7]
        ref_out, ref_state = _run(False, lengths, device, dtype, pad_tokens=2)
        out, state = _run(True, lengths, device, dtype, pad_tokens=2)
        torch.testing.assert_close(out, ref_out, atol=atol, rtol=0)
        # Slot 0 is the inert dump every untracked row snapshots into; its
        # duplicate-index scatter order is unspecified on CUDA.
        torch.testing.assert_close(state[1:], ref_state[1:], atol=atol, rtol=0)

    def test_matches_padded_cpu(self):
        self._check("cpu", torch.float32, atol=1e-6)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_matches_padded_cuda_bf16(self):
        self._check("cuda", torch.bfloat16, atol=1e-2)

    def test_wasteful_padding_gate(self):
        mixed = _batch([4000] + [1] * 40, "cpu")
        prefill = _batch([4000, 3000, 900], "cpu")
        self.assertTrue(qwen4_exp._ple_padding_is_wasteful(mixed))
        self.assertFalse(qwen4_exp._ple_padding_is_wasteful(prefill))


if __name__ == "__main__":
    unittest.main()
