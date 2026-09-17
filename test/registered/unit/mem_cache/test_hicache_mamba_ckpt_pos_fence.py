"""HiCache: a restored Mamba checkpoint must match its node's token position.

Issue #39830: KV bytes could be restored byte-perfect while the mamba slot
paired with them held a checkpoint taken at a different token boundary (the
slot had been re-recorded under another owner), so the model read valid KV
through a stale state -- fluent wrong output, silently. The fence records the
checkpoint's token position on the host slots at backup/prefetch commit time
and, at load-back pairing, refuses the restore (loud degrade to a cache miss,
plus poisoning of the stale host copy) when the recorded position disagrees
with the depth of the node being restored.

The copy kernels are CUDA-only, so this test drives the CPU-side record /
check / poison logic directly: `__new__`-built pools (as
test_hicache_mamba_slot_side_states.py does) over real UnifiedTreeNode chains
and a stub tree core that mirrors the host-eviction wrapper.
"""

import threading
import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components.base import (
    CacheTransferPhase,
    ComponentType,
    PrepareLoadBackResult,
)
from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeNode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

MAMBA = ComponentType.MAMBA
NUM_HOST_SLOTS = 4


def _make_host_pool(with_pos_buffer: bool):
    """CPU MambaPoolHost with only the fields the position record touches."""
    host = MambaPoolHost.__new__(MambaPoolHost)
    host.device = "cpu"
    host.size = NUM_HOST_SLOTS
    host.lock = threading.RLock()
    host.ckpt_pos_buffer = (
        torch.full((NUM_HOST_SLOTS,), -1, dtype=torch.int64)
        if with_pos_buffer
        else None
    )
    return host


def _node(tokens, parent=None):
    node = UnifiedTreeNode((MAMBA,))
    node.key = RadixKey(array("q", tokens))
    node.parent = parent
    return node


class _StubLRU:
    def __init__(self):
        self.nodes = []

    def in_list(self, node):
        return node in self.nodes

    def remove_node(self, node):
        if node in self.nodes:
            self.nodes.remove(node)

    def insert_mru(self, node):
        self.nodes.append(node)


def _make_component(pool, nodes, rust_node_by_id=False):
    """MambaComponent over a stub tree core mirroring the HOST detach wrapper."""
    comp = MambaComponent.__new__(MambaComponent)
    comp._mamba_pool_host = pool
    comp._ckpt_pos_rejects = 0
    lru = _StubLRU()
    updated_leaf_sets = []
    freed_values = []

    def node_by_id(node_id):
        if rust_node_by_id:
            raise NotImplementedError("node_by_id: not yet ported to the Rust core")
        return nodes[node_id]

    def evict_and_detach(node, c, device_frees, host_frees, target, tracker):
        # Mirrors UnifiedTreeCore._evict_component_and_detach_lru for HOST.
        _, host_freed = c.evict_component(
            node,
            target=target,
            device_frees=device_frees,
            host_frees=host_frees,
        )
        tracker[c.component_type] += host_freed
        if lru.in_list(node):
            lru.remove_node(node)

    comp.tree_core = SimpleNamespace(
        node_by_id=node_by_id,
        host_lru_lists={MAMBA: lru},
        _evict_component_and_detach_lru=evict_and_detach,
        _update_evictable_leaf_sets=updated_leaf_sets.append,
    )
    comp.cache = SimpleNamespace(
        _free_values=lambda d, h: freed_values.append((d, h))
    )
    return comp, lru, updated_leaf_sets, freed_values


class TestCkptPosRecord(CustomTestCase):
    def test_record_and_read_round_trip(self):
        pool = _make_host_pool(True)
        pool.record_ckpt_pos(torch.tensor([2]), 15)
        self.assertEqual(pool.get_ckpt_pos(torch.tensor([2])), 15)

    def test_unknown_slot_reads_as_none(self):
        pool = _make_host_pool(True)
        self.assertIsNone(pool.get_ckpt_pos(torch.tensor([0])))

    def test_inconsistent_multi_slot_reads_as_none(self):
        pool = _make_host_pool(True)
        pool.record_ckpt_pos(torch.tensor([0]), 10)
        pool.record_ckpt_pos(torch.tensor([1]), 20)
        self.assertIsNone(pool.get_ckpt_pos(torch.tensor([0, 1])))
        self.assertEqual(pool.get_ckpt_pos(torch.tensor([1, 1])), 20)

    def test_record_on_bufferless_pool_is_a_noop(self):
        pool = _make_host_pool(False)
        pool.record_ckpt_pos(torch.tensor([0]), 5)
        self.assertIsNone(pool.get_ckpt_pos(torch.tensor([0])))

    def test_clear_resets_records(self):
        pool = _make_host_pool(True)
        pool.record_ckpt_pos(torch.tensor([1]), 12)
        pool.mem_state = torch.zeros((pool.size,), dtype=torch.uint8)
        pool.clear()
        self.assertIsNone(pool.get_ckpt_pos(torch.tensor([1])))


class TestRestoreFence(CustomTestCase):
    def _matched_setup(self):
        # root(0) -> parent(10 tokens) -> child(5 tokens): child depth is 15.
        root = _node([])
        parent = _node(list(range(10)), parent=root)
        child = _node(list(range(10, 15)), parent=parent)
        pool = _make_host_pool(True)
        host_value = torch.tensor([3])
        child.component_data[MAMBA].host_value = host_value
        nodes = {child.id: child}
        return pool, child, host_value, nodes

    def test_matched_position_restores_silently(self):
        pool, child, host_value, nodes = self._matched_setup()
        pool.record_ckpt_pos(host_value, 15)
        comp, _, updated, freed = _make_component(pool, nodes)
        prep = comp.prepare_load_back(child.id, req=None)
        self.assertFalse(prep.rejected)
        self.assertIsNone(prep.allocated_mamba_slot)
        self.assertIsNotNone(child.component_data[MAMBA].host_value)
        self.assertEqual(updated, [])
        self.assertEqual(freed, [])

    def test_mismatched_position_degrades_to_miss_and_poisons(self):
        pool, child, host_value, nodes = self._matched_setup()
        # The slot was re-recorded under a shallower owner: stale pairing.
        pool.record_ckpt_pos(host_value, 7)
        comp, lru, updated, freed = _make_component(pool, nodes)
        lru.insert_mru(child)
        child.component_data[MAMBA].host_lock_ref = 1  # under host lock (load-back)
        with self.assertLogs(
            "sglang.srt.mem_cache.unified_cache.components.mamba", "ERROR"
        ):
            prep = comp.prepare_load_back(child.id, req=None)
        self.assertTrue(prep.rejected)
        self.assertIsNone(prep.allocated_mamba_slot)
        cd = child.component_data[MAMBA]
        self.assertIsNone(cd.host_value, "stale host copy must be detached")
        self.assertEqual(len(freed), 1)
        device_frees, host_frees = freed[0]
        self.assertEqual(device_frees, {})
        self.assertEqual([t.tolist() for t in host_frees[MAMBA]], [[3]])
        self.assertFalse(lru.in_list(child))
        self.assertEqual(updated, [child])

    def test_unrecorded_slot_is_never_falsely_accused(self):
        pool, child, _, nodes = self._matched_setup()  # buffer exists, slot -1
        comp, _, updated, freed = _make_component(pool, nodes)
        prep = comp.prepare_load_back(child.id, req=None)
        self.assertFalse(prep.rejected)
        self.assertEqual(freed, [])

    def test_bufferless_pool_skips_the_check(self):
        _, child, host_value, nodes = self._matched_setup()
        pool = _make_host_pool(False)
        comp, _, updated, freed = _make_component(pool, nodes)
        prep = comp.prepare_load_back(child.id, req=None)
        self.assertFalse(prep.rejected)
        self.assertEqual(freed, [])

    def test_rust_tree_core_skips_the_check(self):
        pool, child, host_value, _ = self._matched_setup()
        pool.record_ckpt_pos(host_value, 7)  # would mismatch if inspectable
        comp, _, updated, freed = _make_component(pool, {}, rust_node_by_id=True)
        prep = comp.prepare_load_back(child.id, req=None)
        self.assertFalse(prep.rejected)
        self.assertEqual(freed, [])

    def test_node_without_host_mamba_is_unaffected(self):
        root = _node([])
        pool = _make_host_pool(True)
        comp, _, updated, freed = _make_component(pool, {root.id: root})
        prep = comp.prepare_load_back(root.id, req=None)
        self.assertFalse(prep.rejected)
        self.assertEqual(freed, [])


class TestCommitRecordsPosition(CustomTestCase):
    def test_backup_host_commit_records_node_depth(self):
        root = _node([])
        parent = _node(list(range(10)), parent=root)
        child = _node(list(range(10, 15)), parent=parent)
        pool = _make_host_pool(True)
        comp, _, _, _ = _make_component(pool, {child.id: child})
        host_indices = torch.tensor([1])
        comp.commit_hicache_transfer(
            child,
            CacheTransferPhase.BACKUP_HOST,
            [PoolTransfer(name=PoolName.MAMBA, host_indices=host_indices)],
            cache_actions=[],
        )
        cd = child.component_data[MAMBA]
        self.assertIsNotNone(cd.host_value)
        self.assertEqual(pool.get_ckpt_pos(host_indices), 15)

    def test_existing_host_value_is_not_re_recorded(self):
        root = _node([])
        child = _node(list(range(10, 15)), parent=root)
        pool = _make_host_pool(True)
        pool.record_ckpt_pos(torch.tensor([1]), 15)
        comp, _, _, _ = _make_component(pool, {child.id: child})
        cd = child.component_data[MAMBA]
        cd.host_value = torch.tensor([1])
        comp.commit_hicache_transfer(
            child,
            CacheTransferPhase.BACKUP_HOST,
            [PoolTransfer(name=PoolName.MAMBA, host_indices=torch.tensor([2]))],
            cache_actions=[],
        )
        self.assertIsNone(pool.get_ckpt_pos(torch.tensor([2])))


class TestDefaultsUnaffected(CustomTestCase):
    def test_default_prepare_result_is_not_a_refusal(self):
        # Components without mamba side state keep the old (untouched) flow.
        self.assertFalse(PrepareLoadBackResult().rejected)

    def test_component_without_host_pool_skips_every_leg(self):
        root = _node([])
        child = _node(list(range(5)), parent=root)
        child.component_data[MAMBA].host_value = torch.tensor([0])
        comp, _, _, freed = _make_component(None, {child.id: child})
        # HiCache host tier off: nothing to fence, the flow is unchanged.
        self.assertTrue(comp._host_ckpt_position_ok(child.id))
        prep = comp.prepare_load_back(child.id, req=None)
        self.assertFalse(prep.rejected)
        self.assertEqual(freed, [])


if __name__ == "__main__":
    unittest.main()
