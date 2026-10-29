"""Tripwire for the release/publish ownership split PR #38355 claimed to break.

Upstream PR #38355 ("chunked-prefill radix-insert race corrupting QSA KV
pages", issue #38319) was triaged as MISDIAGNOSED. Its premise -- that a
mid-prefill tree publish followed by a retract/abort leaves the radix tree
holding DANGLING KV pages, with the QSA compressed rows (full_slot // ratio)
amplifying the corruption -- does not hold on this tree: the publish and the
release are disjoint by contract. The tree owns ``[0, cache_protected_len)``;
the request's release frees only ``[cache_protected_len, owned_kv_len)`` plus
the overallocated tail. The original symptom family was already explained and
fixed by the landed entries 196/220 classes, not by this race.

This file converts that refutation research into a permanent regression pin.
It exists so the #38355 class STAYS DEAD: any future change that lets a
release path touch tree-owned pages (a free starting below
``cache_protected_len``, or a publish inserting without advancing
``cache_protected_len``) must fail HERE, LOUDLY, by test name -- before it
can be re-diagnosed as a "new" corruption race. T3 additionally trips when
the PR's own edit targets reappear (a MambaRadixCache-shaped hook or the
QSA variant auto-detect), which is the signal to re-litigate the diagnosis,
not to auto-port the PR.

    python -m pytest test/registered/unit/mem_cache/test_chunked_publish_release_disjointness.py -v
"""

import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import checkpoint_kv_cache, release_kv_cache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# Prod page_size for the QSA arm; pages are the unit of the claimed race.
PAGE = 64


# --------------------------------------------------------------------------- #
# T1: contract-level pin. Fakes drive the REAL release_kv_cache (common.py)
#     and record the physical slots of every free window it issues.
# --------------------------------------------------------------------------- #
class _Harness:
    """Real CPU paged allocator; records every free window AND the device
    slots it covers, so the dangling check runs on physical slots."""

    def __init__(self, num_tokens: int):
        self.alloc = PagedTokenToKVPoolAllocator(
            size=num_tokens * 2,
            page_size=PAGE,
            dtype=torch.float16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        self.row = None
        self.freed_slots: set[int] = set()
        self.freed_windows: list[tuple[int, int]] = []
        self.adopted_slots: set[int] = set()  # slots the tree took mid-prefill

    def alloc_row(self, n: int):
        self.row = self.alloc.alloc(n)
        return self.row

    def publish_chunk(self, req, up_to: int):
        # Mirrors checkpoint_into_tree's tail: the tree adopts the page-
        # floored prefix and cache_protected_len advances in the SAME call.
        new_cpl = (up_to // PAGE) * PAGE
        assert new_cpl > req.kv.cache_protected_len, "publish must advance cpl"
        self.adopted_slots |= set(self.row[:new_cpl].tolist())
        req.kv.cache_protected_len = new_cpl

    def free_kv_row(self, kv, ranges):
        for start, end in ranges:
            if start == end:  # an empty window frees nothing
                continue
            self.freed_windows.append((start, end))
            self.freed_slots |= set(self.row[start:end].tolist())


def _make_tree_cache(harness: _Harness):
    tc = SimpleNamespace()
    tc.token_to_kv_pool_allocator = harness.alloc
    tc.claim_kv_row = lambda req: False
    tc.unlock = lambda lock: None
    tc.on_release = lambda req, checkpointed: None
    tc.supports_mamba = lambda: True
    tc.req_to_token_pool = SimpleNamespace(free=lambda req: None)
    tc.checkpoint = lambda req, *, up_to, **kw: harness.publish_chunk(req, up_to)
    tc.free_kv_row = harness.free_kv_row
    return tc


def _make_req(committed: int, allocated: int, finished: bool):
    kv = ReqKvInfo(req_pool_idx=0)
    kv.kv_committed_len = committed
    kv.kv_allocated_len = allocated
    req = SimpleNamespace()
    req.rid = "race-tripwire"
    req.kv = kv
    req.finished = lambda: finished
    req.finished_len = committed if finished else None
    req.origin_input_ids = array("q", range(committed))
    req.skip_radix_cache_insert = False
    req.lock = None
    req.refresh_fill_ids = lambda: None
    # The real Req.owned_kv_len() answers kv_committed_len for an unfinished
    # request; stub the finished case to the same for this contract test.
    req.owned_kv_len = lambda: kv.kv_committed_len
    return req


class TestReleaseWindowContract(CustomTestCase):
    def setUp(self):
        # release_kv_cache reads the spec/serving config off the published bags.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="scheduler")

    def test_retract_after_midprefill_publish_frees_only_unadopted_tail(self):
        """PR #38355 premise: the retract frees chunk-1 pages the tree
        references. Contract: release_kv_cache(checkpoint=False) -- the
        retract/abort path (schedule_batch.py:2290, scheduler.py:3739) --
        frees [cache_protected_len, owned_kv_len) only. Chunk 1 published,
        chunk 2 written but unpublished: the freed window must not intersect
        what the tree adopted."""
        h = _Harness(2 * PAGE)
        h.alloc_row(2 * PAGE)
        tc = _make_tree_cache(h)
        req = _make_req(committed=2 * PAGE, allocated=2 * PAGE, finished=False)
        h.publish_chunk(req, PAGE)  # the chunk-boundary checkpoint

        release_kv_cache(req, tc, checkpoint=False)  # the retract

        self.assertEqual(req.kv.cache_protected_len, PAGE)
        self.assertTrue(h.adopted_slots, "publish adopted nothing")
        self.assertFalse(
            h.freed_slots & h.adopted_slots,
            "DANGLING: the retract freed tree-adopted slots "
            f"{sorted(h.freed_slots & h.adopted_slots)[:5]} -- "
            "the #38355 corruption class IS live on this tree",
        )
        # The freed window is exactly the unpublished tail [cpl, owned):
        self.assertEqual(h.freed_windows, [(PAGE, 2 * PAGE)])

    def test_release_window_never_overlaps_tree_owned_pages(self):
        """Finish path: release_kv_cache(checkpoint=True) publishes
        up_to=owned FIRST, so [cpl_new, owned) is empty and only the
        overallocated tail [owned, allocated) may go. Nothing below the
        tree-owned prefix [0, cache_protected_len) is ever released."""
        h = _Harness(4 * PAGE)
        h.alloc_row(4 * PAGE)
        tc = _make_tree_cache(h)
        req = _make_req(committed=2 * PAGE, allocated=4 * PAGE, finished=True)
        h.publish_chunk(req, PAGE)  # a chunk was already adopted mid-prefill
        # A stop trimmed the committed prefix: the client-visible output
        # (finished_len) ends before kv_committed_len, so the release is
        # entitled to hand back [committed, allocated) (see #22373 semantics
        # asserted by _release_overallocated_kv_indices).
        req.origin_input_ids = array("q", [7])
        req.finished_len = PAGE

        release_kv_cache(req, tc, checkpoint=True)

        self.assertEqual(req.kv.cache_protected_len, 2 * PAGE)
        self.assertFalse(
            h.freed_slots & h.adopted_slots,
            "DANGLING: the release freed tree-adopted slots "
            f"{sorted(h.freed_slots & h.adopted_slots)[:5]} -- "
            "the #38355 corruption class IS live on this tree",
        )
        # The finish-checkpoint adopted [PAGE, 2*PAGE); only the
        # overallocated tail [2*PAGE, 4*PAGE) went back to the allocator.
        self.assertEqual(h.freed_windows, [(2 * PAGE, 4 * PAGE)])


# --------------------------------------------------------------------------- #
# T2: real UnifiedRadixCache on CPU -- publish a chunk, retract, and prove the
#     tree's match never answers with a slot the release freed. Both TreeCore
#     backends are pinned (rust falls back to python where it is unavailable).
# --------------------------------------------------------------------------- #
class TestRealCacheChunkedPublishRetract(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy", page_size=PAGE), role="scheduler")

    def _drive_publish_then_retract(self, backend: str):
        req_to_token_pool = ReqToTokenPool(
            size=2, max_context_len=4 * PAGE, device="cpu", enable_memory_saver=False
        )
        allocator = PagedTokenToKVPoolAllocator(
            size=8 * PAGE,
            page_size=PAGE,
            dtype=torch.float16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        cache = UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=allocator,
                page_size=PAGE,
                tree_components=(ComponentType.FULL,),
                tree_core_backend=backend,
            )
        )
        prompt = array("q", range(4 * PAGE))
        req = Req(
            rid="race-tripwire",
            origin_input_text="",
            origin_input_ids=prompt,
            sampling_params=SamplingParams(max_new_tokens=8),
        )
        (req.kv.req_pool_idx,) = req_to_token_pool.alloc([req])
        req.lock = cache.lock(cache.root_node_handle())
        indices = allocator.alloc(4 * PAGE)
        self.assertIsNotNone(indices, "fixture pool is too small")
        req_to_token_pool.write((req.kv.req_pool_idx, slice(0, 4 * PAGE)), indices)
        req.output_ids = array("q")
        req.refresh_fill_ids()

        # Chunk 1 of a chunked prefill lands; the scheduler stashes the
        # request and publishes its KV to the tree at the chunk boundary
        # (scheduler.stash_chunked_request -> checkpoint_kv_cache).
        req.extend_end = 2 * PAGE
        req.kv.kv_committed_len = 2 * PAGE
        req.kv.kv_allocated_len = 4 * PAGE
        checkpoint_kv_cache(req, cache)
        self.assertEqual(req.kv.cache_protected_len, 2 * PAGE)
        adopted = set(
            req_to_token_pool.req_to_token[req.kv.req_pool_idx, : 2 * PAGE]
            .clone()
            .tolist()
        )

        # Chunk 2 wrote its KV; the request is retracted (or aborted) before
        # finishing. The tail is everything the release is entitled to free.
        req.kv.kv_committed_len = 4 * PAGE
        tail = (
            req_to_token_pool.req_to_token[req.kv.req_pool_idx, 2 * PAGE : 4 * PAGE]
            .clone()
            .tolist()
        )
        available_before = allocator.available_size()

        release_kv_cache(req, cache, checkpoint=False)  # the retract

        self.assertEqual(allocator.available_size(), available_before + 2 * PAGE)

        match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", prompt[: 2 * PAGE])))
        )
        tree_slots = cache.path_device_indices(match.last_device_node)
        return req, match, adopted, tail, set(tree_slots.tolist())

    def test_match_after_chunked_publish_retract_never_returns_freed_pages(self):
        """The exact #38355 sequence on the real unified cache: mid-prefill
        checkpoint, then release(checkpoint=False). The tree must keep the
        published prefix matchable, answering ONLY with the slots it adopted
        -- never with a slot the release handed back. A non-empty
        intersection here means a live dangling node: that WOULD revive the
        #38355 class on the unified path. The QSA compressed rows ride the
        full-KV slots (qsa_kv_pool.py: "lifecycle rides the full-KV allocator
        and radix tree"), so pinning the full-slot disjointness pins QSA too."""
        for backend in ("python", "rust"):
            with self.subTest(tree_core_backend=backend):
                req, match, adopted, tail, tree_slots = (
                    self._drive_publish_then_retract(backend)
                )
                # The retract left the protection cursor where the publish
                # put it -- the release does not walk it.
                self.assertEqual(req.kv.cache_protected_len, 2 * PAGE)
                # The published prefix survived the retract, page for page.
                self.assertEqual(
                    match.device_prefix_len,
                    2 * PAGE,
                    "#38355 tripwire: the retract dropped the tree's "
                    "published prefix below the chunk boundary",
                )
                # The tree answers with exactly the adopted pages...
                self.assertEqual(
                    tree_slots,
                    adopted,
                    "#38355 tripwire: match_prefix of the published prefix "
                    "answers with slots other than the ones the tree adopted",
                )
                # ...and none of them went back to the allocator.
                self.assertFalse(
                    tree_slots & set(tail),
                    "DANGLING: the retract freed tree-owned pages "
                    f"{sorted(tree_slots & set(tail))[:5]} -- "
                    "the #38355 corruption class IS live on the unified path",
                )


# --------------------------------------------------------------------------- #
# T3: staleness tripwire -- #38355's edit targets and auto-detect API are gone.
#     RED does NOT mean the bug is back; it means the PR's premises are back
#     in play, so RE-READ #38355 before dismissing (or porting) anything.
# --------------------------------------------------------------------------- #
class TestPr38355PremisesGone(CustomTestCase):
    def test_mamba_radix_cache_target_and_variant_api_deleted(self):
        import importlib
        import importlib.util

        import sglang.srt.mem_cache as mc

        self.assertIsNone(
            importlib.util.find_spec(f"{mc.__name__}.mamba_radix_cache"),
            "#38355's edit target mamba_radix_cache.py is back; re-read the "
            "PR before dismissing it -- the MISDIAGNOSED verdict was made "
            "against the unified path",
        )
        qsa_cfg = importlib.import_module("sglang.srt.layers.attention.qsa.config")
        self.assertNotIn(
            "variant",
            getattr(qsa_cfg.QSAProfile, "__annotations__", {}),
            "QSAProfile.variant is back; the PR's auto-detect could apply again",
        )
        self.assertFalse(
            hasattr(qsa_cfg, "QSA_VARIANT_COMPRESSED"),
            "the PR's QSA_VARIANT_COMPRESSED marker is back; re-litigate the "
            "MISDIAGNOSED verdict before porting #38355",
        )
        self.assertNotIn(
            "disable_chunked_radix_insert",
            {
                f
                for c in CacheInitParams.__mro__
                for f in getattr(c, "__dataclass_fields__", {})
            },
            "#38355's kill switch was re-added -- the tree no longer "
            "guarantees publish/release disjointness? Investigate T1/T2 first",
        )
        self.assertNotIn(
            "disable_chunked_radix_insert",
            getattr(ServerArgs, "__annotations__", {}),
            "#38355's CLI flag was re-added; the refutation above predates it",
        )
        self.assertFalse(
            hasattr(UnifiedRadixCache, "cache_unfinished_req"),
            "the PR's chunked=True insertion hook is back on the unified "
            "cache; re-validate the diagnosis before trusting this pin",
        )
        self.assertTrue(hasattr(UnifiedRadixCache, "checkpoint"))


if __name__ == "__main__":
    unittest.main()
