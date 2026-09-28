"""Circular GDN ReplaySSM history must reach `temporal` before a plain extend.

Between verifies the circular spec-verify protocol keeps `temporal` at the
ring's `cache_base` with the committed tail in the ring. Only the verify
kernel replays that ring; the extend kernels read `temporal` directly, so a
running request extended outside verify (a mixed-chunk decode tail) must be
folded first, and only rows with pending history may be touched.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative.spec_utils import (
    fold_gdn_replayssm_history_for_extend,
    gdn_replayssm_circular_active,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _req_pool(write_pos, is_flush):
    spec_state = SimpleNamespace(
        temporal="temporal",
        replayssm_d="d",
        replayssm_k="k",
        replayssm_g="g",
        replayssm_rawv="rawv",
        replayssm_rawk="rawk",
    )
    mamba_pool = SimpleNamespace(
        replayssm_spec_write_pos=torch.tensor(write_pos, dtype=torch.int32),
        replayssm_cache_base=torch.zeros(len(write_pos), dtype=torch.int32),
        replayssm_is_flush=torch.tensor(is_flush, dtype=torch.int8),
        replayssm_is_kda=False,
    )
    return SimpleNamespace(
        mamba_pool=mamba_pool,
        get_speculative_mamba2_params_all_layers=lambda: spec_state,
        get_mamba_indices=lambda idx: idx + 100,
    )


class TestReplaySSMExtendFold(CustomTestCase):
    def test_flags_only_rows_with_pending_history(self):
        # req-pool rows 1..4: row 1 has history, row 2 is empty, row 3 has
        # history and is already flagged, row 4 is empty but flagged.
        pool = _req_pool([0, 3, 0, 5, 0], [0, 0, 0, 1, 1])
        rows = torch.tensor([1, 2, 3, 4])
        with patch(
            "sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode."
            "commit_gdn_replayssm_circular"
        ) as commit:
            fold_gdn_replayssm_history_for_extend(pool, rows)
        self.assertEqual(
            pool.mamba_pool.replayssm_is_flush.tolist(), [0, 1, 0, 1, 1]
        )
        kwargs = commit.call_args.kwargs
        self.assertEqual(kwargs["replay_indices"].tolist(), [1, 2, 3, 4])
        self.assertEqual(kwargs["state_batch_indices"].tolist(), [101, 102, 103, 104])
        # Zero accepted tokens: write_pos already holds every committed step.
        self.assertEqual(kwargs["accept_lens"].tolist(), [0, 0, 0, 0])
        self.assertIsNone(kwargs.get("mamba_track_indices"))
        self.assertEqual(kwargs["checkpoint_state"], "temporal")
        self.assertEqual(kwargs["d_residual_cache"], "rawv")

    def test_circular_protocol_detection(self):
        self.assertTrue(gdn_replayssm_circular_active(_req_pool([0], [0])))
        kda = _req_pool([0], [0])
        kda.mamba_pool.replayssm_is_kda = True
        self.assertFalse(gdn_replayssm_circular_active(kda))
        no_cursor = _req_pool([0], [0])
        no_cursor.mamba_pool.replayssm_cache_base = None
        self.assertFalse(gdn_replayssm_circular_active(no_cursor))
        self.assertFalse(gdn_replayssm_circular_active(SimpleNamespace()))


if __name__ == "__main__":
    unittest.main()
