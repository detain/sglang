"""The ReplaySSM GDN verify commit must still commit the Qwen4 PLE side states.

Target verify writes the PLE short-conv and N-gram states only to per-step
intermediate buffers; `commit_mamba_states_after_verify` returns early on the
ReplaySSM branch, before `update_mamba_state_after_mtp_verify` (the regular PLE
commit site), so that branch has to commit them itself.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    commit_ple_state_after_mtp_verify,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

NUM_SLOTS = 6
NUM_LAYERS = 2
CONV_DIM = 3
CONV_WIDTH = 2
CONTEXT_LEN = 2
DRAFT_TOKENS = 4


def _make_ple_pools(bs: int):
    short_conv_pool = SimpleNamespace(
        conv_state=torch.zeros(NUM_LAYERS, NUM_SLOTS, CONV_DIM, CONV_WIDTH),
        intermediate_conv_state=torch.arange(
            NUM_LAYERS * bs * DRAFT_TOKENS * CONV_DIM * CONV_WIDTH, dtype=torch.float32
        ).reshape(NUM_LAYERS, bs, DRAFT_TOKENS, CONV_DIM, CONV_WIDTH),
    )
    ngram_pool = SimpleNamespace(
        context=torch.zeros(NUM_SLOTS, CONTEXT_LEN, dtype=torch.int64),
        intermediate_context=torch.arange(
            bs * DRAFT_TOKENS * CONTEXT_LEN, dtype=torch.int64
        ).reshape(bs, DRAFT_TOKENS, CONTEXT_LEN)
        + 100,
    )
    return short_conv_pool, ngram_pool


class TestCommitPleStateAfterMtpVerify(CustomTestCase):
    def test_commits_last_accepted_step_and_track_step(self):
        bs = 2
        short_conv_pool, ngram_pool = _make_ple_pools(bs)
        pool = SimpleNamespace(short_conv_pool=short_conv_pool, ngram_pool=ngram_pool)
        slots = torch.tensor([3, 1])
        steps = torch.tensor([2, 0])
        track_slots = torch.tensor([4, 5])
        track_steps = torch.tensor([1, -1])

        commit_ple_state_after_mtp_verify(pool, slots, steps, track_slots, track_steps)

        for row, (slot, step) in enumerate(zip(slots.tolist(), steps.tolist())):
            torch.testing.assert_close(
                short_conv_pool.conv_state[:, slot],
                short_conv_pool.intermediate_conv_state[:, row, step],
            )
            torch.testing.assert_close(
                ngram_pool.context[slot], ngram_pool.intermediate_context[row, step]
            )
        torch.testing.assert_close(
            ngram_pool.context[4], ngram_pool.intermediate_context[0, 1]
        )
        # step -1 means no interval crossing: the track slot stays untouched.
        self.assertTrue(torch.all(ngram_pool.context[5] == 0))
        self.assertTrue(torch.all(short_conv_pool.conv_state[:, 5] == 0))

    def test_noop_without_ple_pools(self):
        commit_ple_state_after_mtp_verify(
            SimpleNamespace(), torch.tensor([0]), torch.tensor([0]), None, None
        )
        empty = SimpleNamespace(
            short_conv_pool=SimpleNamespace(
                conv_state=None, intermediate_conv_state=None
            ),
            ngram_pool=SimpleNamespace(context=None, intermediate_context=None),
        )
        commit_ple_state_after_mtp_verify(
            empty, torch.tensor([0]), torch.tensor([0]), None, None
        )


class TestReplaySSMVerifyCommitsPle(CustomTestCase):
    def test_replayssm_gdn_branch_commits_ple_states(self):
        from sglang.srt.speculative.spec_utils import commit_mamba_states_after_verify

        bs = 2
        short_conv_pool, ngram_pool = _make_ple_pools(bs)
        slots = torch.tensor([3, 1])

        mamba_pool = SimpleNamespace(
            replayssm_spec_fold=False,
            replayssm_is_kda=False,
            replayssm_cache_base=torch.zeros(NUM_SLOTS, dtype=torch.int32),
            replayssm_spec_write_pos=None,
            replayssm_is_flush=None,
        )
        spec_state = SimpleNamespace(
            replayssm_d=torch.zeros(1, 1, 8, 1),
            temporal=torch.zeros(1, dtype=torch.bfloat16),
            replayssm_k=None,
            replayssm_g=None,
            replayssm_rawv=None,
            replayssm_rawk=None,
            conv=[None],
            intermediate_conv_window=[None],
        )
        req_pool = SimpleNamespace(
            mamba_pool=mamba_pool,
            short_conv_pool=short_conv_pool,
            ngram_pool=ngram_pool,
            get_speculative_mamba2_params_all_layers=lambda: spec_state,
            get_mamba_indices=lambda req_pool_indices: slots,
        )
        attn_backend = MagicMock()
        target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                model_config=None, req_to_token_pool=req_pool, attn_backend=attn_backend
            )
        )
        batch = MagicMock()
        batch.forward_mode.is_idle.return_value = False
        batch.mamba_track_indices = None
        batch.req_pool_indices = torch.tensor([0, 1])
        batch.seq_lens = torch.tensor([10, 20], dtype=torch.int32)
        accept_lens = torch.tensor([3, 1], dtype=torch.int32)
        accept_index = torch.tensor([[0, 1, 2, -1], [4, -1, -1, -1]], dtype=torch.int32)

        with (
            patch(
                "sglang.srt.speculative.spec_utils.mambaish_config",
                return_value={"some": "config"},
            ),
            patch(
                "sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode."
                "commit_gdn_replayssm_spec"
            ),
            patch(
                "sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode."
                "commit_gdn_replayssm_circular"
            ),
            patch(
                "sglang.kernels.ops.mamba.mamba_state_scatter_triton."
                "fused_conv_window_scatter_with_mask"
            ),
        ):
            commit_mamba_states_after_verify(
                target_worker,
                batch,
                accept_lens,
                accept_index,
                draft_token_num=DRAFT_TOKENS,
            )

        # The ReplaySSM branch returns early -- the backend hook is not reached.
        attn_backend.update_mamba_state_after_mtp_verify.assert_not_called()
        for row, (slot, step) in enumerate(zip(slots.tolist(), [2, 0])):
            torch.testing.assert_close(
                ngram_pool.context[slot], ngram_pool.intermediate_context[row, step]
            )
            torch.testing.assert_close(
                short_conv_pool.conv_state[:, slot],
                short_conv_pool.intermediate_conv_state[:, row, step],
            )


if __name__ == "__main__":
    unittest.main()
