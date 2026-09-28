"""A prefill co-batched with running decodes must still write its checkpoint.

prepare_for_extend claims a ping-pong slot and stamps mamba_last_track_seqlen
for every tracked extend row before the scheduler decides to mix. merge_batch
drops the batch's track tensors, so a MIXED forward used to skip the
checkpoint write while the radix cache still donated the claimed slot at the
stamped depth; every later prefix hit then restored stale state (#39342).
mix_with_running now carries the extend rows' tracking across the merge and
pads the decode tails as untracked rows.
"""

import types
import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _req(next_track_idx):
    return types.SimpleNamespace(
        kv=types.SimpleNamespace(mamba_next_track_idx=next_track_idx)
    )


def _extend_side():
    # Two extend rows: row 0 tracked (slot 40 at depth 2040), row 1 untracked.
    return types.SimpleNamespace(
        mamba_track_indices=torch.tensor([40, 41]),
        mamba_track_mask=torch.tensor([True, False]),
        mamba_track_seqlens=torch.tensor([2041, -1]),
        mamba_prefill_track_mask_cpu=[True, False],
        mamba_track_seqlens_cpu=[2041, -1],
        # ping-pong slots per req-pool row: [slot at idx 0, slot at idx 1]
        req_to_token_pool=types.SimpleNamespace(
            req_index_to_mamba_ping_pong_track_buffer_mapping=torch.tensor(
                [[0, 0], [10, 11], [12, 13], [14, 15]]
            )
        ),
    )


def _running(track_indices=None, next_track_idx=(0, 1)):
    return types.SimpleNamespace(
        batch_size=lambda: 2,
        mamba_track_indices=track_indices,
        req_pool_indices=torch.tensor([2, 3]),
        reqs=[_req(i) for i in next_track_idx],
    )


class TestMambaTrackForMixed(unittest.TestCase):
    def test_extend_rows_keep_tracking_and_tails_are_masked_off(self):
        # A decode tail sitting on a track boundary carries its own slot but
        # must not become a writer in the mixed forward.
        indices, mask, seqlens, mask_cpu, seqlens_cpu = (
            ScheduleBatch._mamba_track_for_mixed(
                _extend_side(), _running(track_indices=torch.tensor([12, 15]))
            )
        )
        self.assertEqual(indices.tolist(), [40, 41, 12, 15])
        self.assertEqual(mask.tolist(), [True, False, False, False])
        self.assertEqual(seqlens.tolist(), [2041, -1, -1, -1])
        self.assertEqual(mask_cpu, [True, False, False, False])
        self.assertEqual(seqlens_cpu, [2041, -1, -1, -1])

    def test_spec_running_batch_gets_its_slots_from_reqs(self):
        # Spec decode prepares its track indices inside the forward, so the
        # running batch has none; every tail still needs a valid slot id.
        indices, mask, _, _, _ = ScheduleBatch._mamba_track_for_mixed(
            _extend_side(), _running(track_indices=None, next_track_idx=(1, 0))
        )
        self.assertEqual(indices.tolist(), [40, 41, 13, 14])
        self.assertEqual(mask.tolist(), [True, False, False, False])

    def test_no_extra_buffer_carries_nothing(self):
        extend = _extend_side()
        extend.mamba_track_mask = None
        self.assertIsNone(ScheduleBatch._mamba_track_for_mixed(extend, _running()))

    def test_missing_cpu_mirrors_stay_missing(self):
        extend = _extend_side()
        extend.mamba_prefill_track_mask_cpu = None
        extend.mamba_track_seqlens_cpu = None
        _, _, _, mask_cpu, seqlens_cpu = ScheduleBatch._mamba_track_for_mixed(
            extend, _running(track_indices=torch.tensor([12, 15]))
        )
        self.assertIsNone(mask_cpu)
        self.assertIsNone(seqlens_cpu)


if __name__ == "__main__":
    unittest.main()
