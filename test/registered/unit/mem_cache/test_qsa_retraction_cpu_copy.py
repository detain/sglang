"""QSA retraction backup must carry the per-token compressed-K rows.

`Req.offload_kv_cache` snapshots the KV through the allocator's
`get_cpu_copy`, frees the token slots, and `load_kv_cache` restores into
freshly allocated slots. QSA addresses compressed keys by
``compressed_row = full_slot // qsa_compress_ratio`` and their lifecycle rides
the full-KV allocator, so the rows for the new slots may already hold another
request's compressed keys when the resume reads block selections -- the same
"valid KV read through stale sidecar" class as sgl-project/sglang#39830, on
the retraction path.

The device KV/Mamba pools are CUDA-only to construct, so the two leaf pools of
the real `HybridLinearKVPool.get_cpu_copy`/`load_cpu_copy` composition are
stubbed and the QSA legs plus the real wrapper are exercised on CPU tensors
(same approach as test_hicache_mamba_slot_side_states.py).
"""

import unittest

import torch

from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

RATIO = 4
HEADS = 2
DIM = 3
CAPACITY = 32
NUM_LOCAL_LAYERS = 2


class _StubFullKVPool:
    """Stands in for the MHA full-KV pool: a CPU copy is an opaque marker
    tensor recording the slot ids it was taken from."""

    def get_cpu_copy(self, indices, req_pool_index=None):
        return {"slots": indices.clone(), "req_pool_index": req_pool_index}

    def load_cpu_copy(self, kv_cache_cpu, indices, req_pool_index=None):
        self.loaded = (kv_cache_cpu, indices, req_pool_index)


class _StubMambaPool:
    def get_cpu_copy(self, mamba_indices):
        return {"mamba": mamba_indices}

    def load_cpu_copy(self, mamba_cpu, mamba_indices):
        self.loaded = (mamba_cpu, mamba_indices)


def _make_pool():
    pool = object.__new__(QSATokenToKVPool)
    pool.qsa_compress_ratio = RATIO
    pool.qsa_compressed_k_buffer_pool = [
        torch.zeros(CAPACITY, HEADS, DIM, dtype=torch.float32)
        for _ in range(NUM_LOCAL_LAYERS)
    ]
    pool.full_kv_pool = _StubFullKVPool()
    pool.mamba_pool = _StubMambaPool()
    pool._mamba_translate = lambda ids: ids
    return pool


class TestQSARetractionCpuCopy(CustomTestCase):
    # Slot ids must respect the allocation invariant: an offloaded prefix is
    # position-contiguous and its slots come from allocator pages, so every
    # group of RATIO consecutive positions shares one compressed row on both
    # the old and the new side (runs align at multiples of RATIO in position).

    def test_get_cpu_copy_carries_qsa_rows(self):
        pool = _make_pool()
        # 2 full compression groups: positions -> slots [8..11], [20..23].
        old_indices = torch.tensor([8, 9, 10, 11, 20, 21, 22, 23])
        for buf in pool.qsa_compressed_k_buffer_pool:
            buf[2] = 2.0
            buf[5] = 5.0
        payload = pool.get_cpu_copy(old_indices, mamba_indices=None)
        self.assertIsInstance(payload, dict)
        # KV/mamba legs ride unchanged through HybridLinearKVPool.
        self.assertEqual(payload["kv"][0]["slots"].tolist(), old_indices.tolist())
        self.assertIsNone(payload["kv"][1])
        # One [tokens, heads, dim] CPU tensor per local layer, in slot order.
        self.assertEqual(len(payload["qsa_compressed_k"]), NUM_LOCAL_LAYERS)
        expected_rows = [2, 2, 2, 2, 5, 5, 5, 5]
        for leg in payload["qsa_compressed_k"]:
            self.assertEqual(leg.device.type, "cpu")
            self.assertEqual(leg.shape, (len(old_indices), HEADS, DIM))
            for out, row in zip(leg, expected_rows):
                self.assertTrue(torch.all(out == float(row)))

    def test_load_cpu_copy_restores_into_new_slots(self):
        pool = _make_pool()
        old_indices = torch.tensor([8, 9, 10, 11, 20, 21, 22, 23])
        for buf in pool.qsa_compressed_k_buffer_pool:
            buf[2] = 2.0
            buf[5] = 5.0
        payload = pool.get_cpu_copy(old_indices, mamba_indices=torch.tensor([7]))

        # Slots reclaimed and reused by another request: corrupt the old rows,
        # then stamp foreign values onto the new slots' rows.
        for buf in pool.qsa_compressed_k_buffer_pool:
            buf[[2, 5]] = -1.0
        new_indices = torch.tensor([100, 101, 102, 103, 108, 109, 110, 111])
        for buf in pool.qsa_compressed_k_buffer_pool:
            buf[25] = 99.0
            buf[27] = 98.0

        pool.load_cpu_copy(payload, new_indices, mamba_indices=torch.tensor([7]))

        # The KV leg reached the real base implementation (which unpacked the
        # 2-tuple and forwarded the marker to the stub leaf pools).
        loaded_kv, loaded_indices, _ = pool.full_kv_pool.loaded
        self.assertEqual(loaded_kv["slots"].tolist(), old_indices.tolist())
        self.assertEqual(loaded_indices.tolist(), new_indices.tolist())
        self.assertEqual(pool.mamba_pool.loaded[1].tolist(), [7])

        # QSA rows for the NEW slots hold the saved values; rows for other
        # slots are untouched.
        for buf in pool.qsa_compressed_k_buffer_pool:
            self.assertTrue(torch.all(buf[25] == 2.0))
            self.assertTrue(torch.all(buf[27] == 5.0))
            self.assertTrue(torch.all(buf[2] == -1.0))
            self.assertTrue(torch.all(buf[5] == -1.0))
            self.assertTrue(torch.all(buf[0] == 0.0))
            self.assertTrue(torch.all(buf[31] == 0.0))

    def test_load_cpu_copy_tolerates_legacy_tuple(self):
        # A (kv, mamba) 2-tuple from a producer without the QSA leg must still
        # restore KV instead of exploding on resume.
        pool = _make_pool()
        legacy = ({"slots": torch.tensor([8])}, None)
        pool.load_cpu_copy(legacy, torch.tensor([100]))
        loaded_kv, loaded_indices, _ = pool.full_kv_pool.loaded
        self.assertEqual(loaded_kv["slots"].tolist(), [8])
        self.assertEqual(loaded_indices.tolist(), [100])
        # No QSA scatter happened.
        for buf in pool.qsa_compressed_k_buffer_pool:
            self.assertTrue(torch.all(buf == 0.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
