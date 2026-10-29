"""QSA retraction backup must carry the compressed-K rows and the pending ring.

`Req.offload_kv_cache` snapshots the KV through the allocator's
`get_cpu_copy`, frees the token slots, and `load_kv_cache` restores into
freshly allocated slots AND a different request-pool row. QSA addresses
compressed keys by ``compressed_row = full_slot // qsa_compress_ratio`` and
their lifecycle rides the full-KV allocator, so the rows for the new slots may
already hold another request's compressed keys when the resume reads block
selections -- the same "valid KV read through stale sidecar" class as
sgl-project/sglang#39830, on the retraction path. The per-request pending
index-key ring (keys + RoPE) is addressed by request row, and the resume lands
on a NEW row, so it must travel with the backup too (PR #45865); a straddling
group's leading members would otherwise be compressed from the row's previous
owner.

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
NUM_GROUPS = 2  # the QSA_RING_GROUPS floor; the ring legs track the attribute
REQ_ROWS = 6
RING_ROWS = RATIO * NUM_GROUPS


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


def _make_pool(qsa_num_groups: int = NUM_GROUPS):
    pool = object.__new__(QSATokenToKVPool)
    pool.qsa_compress_ratio = RATIO
    pool.qsa_compressed_k_buffer_pool = [
        torch.zeros(CAPACITY, HEADS, DIM, dtype=torch.float32)
        for _ in range(NUM_LOCAL_LAYERS)
    ]
    pool.qsa_num_groups = qsa_num_groups
    pool.qsa_key_state_buffer_pool = [
        torch.zeros(REQ_ROWS * RATIO * qsa_num_groups, HEADS, DIM, dtype=torch.float32)
        for _ in range(NUM_LOCAL_LAYERS)
    ]
    pool.qsa_rope_position_buffer = torch.zeros(
        (REQ_ROWS * RATIO * qsa_num_groups, 3), dtype=torch.int64
    )
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

    # --- PR #45865: the pending index-key ring travels with the backup -------

    def test_resumed_request_keeps_its_pending_ring(self):
        """A request retracted from a row that another request then reuses must
        resume with its own pending ring -- the WHOLE ring, all
        ``ratio * qsa_num_groups`` rows: with the QSA_RING_GROUPS floor the
        members of the group still pending after a verify step can sit in
        either ring group, and the resume reads them before re-storing."""
        pool = _make_pool()
        old_ring = torch.arange(1 * RING_ROWS, 2 * RING_ROWS)
        for layer, buf in enumerate(pool.qsa_key_state_buffer_pool):
            buf[old_ring] = (
                (torch.arange(RING_ROWS) + 100 * layer).view(-1, 1, 1).float()
            )
        pool.qsa_rope_position_buffer[old_ring] = torch.arange(RING_ROWS * 3).view(
            RING_ROWS, 3
        )

        backup = pool.get_cpu_copy(torch.tensor([8, 9, 10, 11]), req_pool_index=1)
        self.assertIsNotNone(backup["qsa_pending_ring"])

        # Row 3 (the resume's new row) now holds a foreign request's ring.
        new_ring = torch.arange(3 * RING_ROWS, 4 * RING_ROWS)
        for buf in pool.qsa_key_state_buffer_pool:
            buf[new_ring] = -7.0
        pool.qsa_rope_position_buffer[new_ring] = -7

        pool.load_cpu_copy(backup, torch.tensor([100, 101, 102, 103]), req_pool_index=3)
        for layer, buf in enumerate(pool.qsa_key_state_buffer_pool):
            want = (torch.arange(RING_ROWS) + 100 * layer).view(-1, 1, 1).float()
            self.assertTrue(torch.equal(buf[new_ring], want.expand_as(buf[new_ring])))
        self.assertTrue(
            torch.equal(
                pool.qsa_rope_position_buffer[new_ring],
                torch.arange(RING_ROWS * 3).view(RING_ROWS, 3),
            )
        )

    def test_partial_last_group_and_distinct_ring_values(self):
        """Ported from PR #45865: a request whose length is not a multiple of
        the ratio, moved to a different row and reordered pages; every group,
        ring member and RoPE axis has its own value."""
        page = 16  # stub-pool page geometry; slots -> compressed rows // RATIO

        def pages(*ids):
            return torch.cat([torch.arange(p * page, (p + 1) * page) for p in ids])

        pool = _make_pool()
        n = page + RATIO - 1  # one full page plus a partial group
        old_slots = pages(1, 5)[:n]
        new_slots = pages(4, 2)[:n]
        old_groups = old_slots[::RATIO] // RATIO
        new_groups = new_slots[::RATIO] // RATIO
        for layer, buf in enumerate(pool.qsa_compressed_k_buffer_pool):
            for g, row in enumerate(old_groups):
                buf[row] = 100 * layer + g + 1
        ring_old = torch.arange(1 * RING_ROWS, 2 * RING_ROWS)
        for layer, buf in enumerate(pool.qsa_key_state_buffer_pool):
            buf[ring_old] = (
                (torch.arange(RING_ROWS) + 50 * layer + 1).view(-1, 1, 1).float()
            )
        pool.qsa_rope_position_buffer[ring_old] = torch.arange(RING_ROWS * 3).view(
            RING_ROWS, 3
        )

        backup = pool.get_cpu_copy(old_slots, req_pool_index=1)
        pool.load_cpu_copy(backup, new_slots, req_pool_index=4)

        for layer, buf in enumerate(pool.qsa_compressed_k_buffer_pool):
            for g, row in enumerate(new_groups):
                self.assertTrue(torch.all(buf[row] == 100 * layer + g + 1))
        ring_new = torch.arange(4 * RING_ROWS, 5 * RING_ROWS)
        for layer, buf in enumerate(pool.qsa_key_state_buffer_pool):
            want = (torch.arange(RING_ROWS) + 50 * layer + 1).view(-1, 1, 1).float()
            self.assertTrue(torch.equal(buf[ring_new], want.expand_as(buf[ring_new])))
        self.assertTrue(
            torch.equal(
                pool.qsa_rope_position_buffer[ring_new],
                torch.arange(RING_ROWS * 3).view(RING_ROWS, 3),
            )
        )

    def test_pp_stage_without_qsa_layers(self):
        """Ported from PR #45865: with no local QSA layer there are no QSA
        buffers and no ring to carry -- the shared RoPE ring must not be
        overwritten from a stage that never wrote it."""
        pool = _make_pool()
        pool.qsa_compressed_k_buffer_pool = []
        pool.qsa_key_state_buffer_pool = []
        ring = torch.arange(2 * RING_ROWS, 3 * RING_ROWS)
        pool.qsa_rope_position_buffer[ring] = 7

        backup = pool.get_cpu_copy(torch.arange(0, RATIO), req_pool_index=1)
        pool.load_cpu_copy(backup, torch.arange(16, 16 + RATIO), req_pool_index=2)

        self.assertIsNone(backup["qsa_pending_ring"])
        self.assertTrue(torch.all(pool.qsa_rope_position_buffer[ring] == 7))

    def test_empty_indices(self):
        """Ported from PR #45865: an empty retraction snapshot still round-trips
        (the ring rides the request row, not the token indices)."""
        pool = _make_pool()
        empty = torch.empty(0, dtype=torch.int64)
        pool.load_cpu_copy(
            pool.get_cpu_copy(empty, req_pool_index=1), empty, req_pool_index=2
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
