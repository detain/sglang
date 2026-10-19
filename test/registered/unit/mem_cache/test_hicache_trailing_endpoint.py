"""Trailing sidecars may only be read at an endpoint the hit query verified.

A hybrid (attention + Mamba/GDN) prefix is restorable from L3 only at a page
whose ``<hash>.mamba`` sidecar exists: the mamba component writes exactly one
sidecar per node boundary, so most pages inside an otherwise stored prefix have
none. The hit query folds that into a single deepest boundary, but a span that
host capacity later shortens used to end on an arbitrary interior page instead.
The sidecar read at that endpoint always missed, and the all-or-nothing result
check then discarded a KV prefix that had transferred in full
(``completed == requested`` in the discard log).

These tests drive the real ``HiCacheFile`` backend, the real rank sync and the
real allocation clamp on CPU; the rank-agreement case runs two gloo ranks.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import shutil
import tempfile
import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache import hybrid_cache_controller as hcc
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
    PrefetchOperation,
)
from sglang.test.test_utils import CustomTestCase

_PAGE = 4
_HASHES = [f"p{i}" for i in range(10)]
# Mamba states land on node boundaries only, so pages 3 and 8 (prefix lengths)
# carry a sidecar and the other seven do not.
_SIDECAR_PAGES = (3, 8)
# The query folds the span to its deepest restorable endpoint, i.e. 8 pages.
_HIT_PAGES = max(_SIDECAR_PAGES)
_TOKENS = _PAGE * len(_HASHES)


def _t(n_bytes: int) -> torch.Tensor:
    return torch.full((n_bytes,), 1, dtype=torch.uint8)


def _mamba_transfer() -> PoolTransfer:
    # What the mamba component builds for a prefetch: one placeholder key, so
    # the trailing window is the span's last page.
    return PoolTransfer(
        name=PoolName.MAMBA,
        keys=["__placeholder__"],
        hit_policy=PoolHitPolicy.TRAILING_PAGES,
    )


def _make_backend(file_path: str) -> HiCacheFile:
    backend = HiCacheFile(
        HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=True,
            model_name="testmodel",
            extra_config={"enable_metadata_cache": False},
        ),
        file_path=file_path,
    )
    for key in _HASHES:
        assert backend.set(key, _t(8))
    for pages in _SIDECAR_PAGES:
        assert backend.set(f"{_HASHES[pages - 1]}.mamba", _t(8))
    return backend


def _make_operation(
    pool_transfers, restorable, hit_pages: int = _HIT_PAGES
) -> PrefetchOperation:
    operation = PrefetchOperation(
        CacheRequestHandle("req-1", 0),
        token_ids=list(range(_TOKENS)),
        last_hash=None,
        prefix_keys=None,
        pool_transfers=pool_transfers,
    )
    # The post-query, post-sync view the scheduler allocates against.
    operation.all_hash_values = list(_HASHES)
    operation.hash_value = list(_HASHES[:hit_pages])
    operation.storage_hit_count = hit_pages * _PAGE
    operation.query_pool_hit_pages = {
        PoolName.KV: len(_HASHES),
        PoolName.MAMBA: max(restorable) if restorable else 0,
    }
    operation.restorable_prefix_pages = list(restorable)
    return operation


def _make_controller(capacity_tokens: int):
    """Controller whose host arena hands out at most ``capacity_tokens``."""
    cc = HybridCacheController.__new__(HybridCacheController)
    cc.page_size = _PAGE
    cc.mem_pool_host = SimpleNamespace(
        anchor_entry=SimpleNamespace(host_pool=object(), name=PoolName.KV),
        available_size=lambda: capacity_tokens,
        entry_map={},
    )
    attempts: list[int] = []

    def _alloc(operation, need_size):
        attempts.append(need_size)
        if need_size > capacity_tokens:
            return None
        return torch.arange(need_size)

    cc.alloc_prefetch_host_buffers = _alloc
    return cc, attempts


class TestFileBackendEndpoints(CustomTestCase):
    """The query has the whole existence set in hand; report all of it."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="hicache_endpoint_backend_")
        self.backend = _make_backend(self.dir)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_trailing_pool_reports_every_restorable_endpoint(self):
        result = self.backend.batch_exists_v2(_HASHES, [_mamba_transfer()])
        # KV is stored for all ten pages, but the span folds to page 8.
        self.assertEqual(result.kv_hit_pages, _HIT_PAGES)
        self.assertEqual(
            result.extra_pool_hit_pages,
            {PoolName.KV: len(_HASHES), PoolName.MAMBA: _HIT_PAGES},
        )
        # 3 is usable too, and it is the only option once capacity cuts in.
        self.assertEqual(result.restorable_prefix_pages, list(_SIDECAR_PAGES))

    def test_stop_points_intersect_across_pools(self):
        indexer = PoolTransfer(
            name=PoolName.INDEXER,
            keys=list(_HASHES),
            hit_policy=PoolHitPolicy.ALL_PAGES,
        )
        for key in _HASHES[:6]:
            assert self.backend.set(f"{key}.{PoolName.INDEXER}", _t(8))
        result = self.backend.batch_exists_v2(_HASHES, [_mamba_transfer(), indexer])
        self.assertEqual(
            result.extra_pool_hit_pages,
            {PoolName.KV: len(_HASHES), PoolName.INDEXER: 6, PoolName.MAMBA: 8},
        )
        # Page 6 would fold the min to 6, but no mamba state is stored there:
        # only the intersection of the two pools' stop points is restorable.
        self.assertEqual(result.restorable_prefix_pages, [3])
        self.assertEqual(result.kv_hit_pages, 3)

    def test_kv_only_span_is_unconstrained(self):
        result = self.backend.batch_exists_v2(_HASHES, [])
        self.assertEqual(result.kv_hit_pages, len(_HASHES))
        self.assertEqual(result.restorable_prefix_pages, list(range(1, 11)))


class TestPartialSpanEndpointClamp(CustomTestCase):
    """A capacity-shortened span stops on a verified endpoint or not at all."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="hicache_endpoint_clamp_")
        self.backend = _make_backend(self.dir)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _allocate(self, cc, operation, hit_tokens, min_tokens, evict_host):
        with mock.patch.object(hcc, "uses_shared_host_layout", lambda pool: False):
            return cc.allocate_storage_hit(
                operation,
                hit_tokens,
                allow_partial=True,
                min_tokens=min_tokens,
                evict_host=evict_host,
            )

    def _endpoint_keys(self, cc, operation, alloc_len):
        """What the IO thread reads the sidecar at for the allocated span."""
        pages = alloc_len // _PAGE
        cc._sync_trailing_keys(operation.pool_transfers, operation.hash_value, pages)
        return list(operation.pool_transfers[0].keys)

    def test_clamp_lands_on_a_verified_endpoint(self):
        # Host space covers 6 of the 8 pages the query vouched for.
        cc, attempts = _make_controller(capacity_tokens=6 * _PAGE)
        operation = _make_operation([_mamba_transfer()], list(_SIDECAR_PAGES))
        host_indices, alloc_len = self._allocate(
            cc, operation, 8 * _PAGE, _PAGE, mock.Mock()
        )
        self.assertIsNotNone(host_indices)
        self.assertEqual(alloc_len, 3 * _PAGE)
        operation.hash_value = operation.hash_value[: alloc_len // _PAGE]
        keys = self._endpoint_keys(cc, operation, alloc_len)
        self.assertEqual(keys, [_HASHES[2]])
        for key in keys:
            # The whole point: this read resolves instead of forfeiting the KV
            # prefix that was fetched alongside it.
            self.assertTrue(
                self.backend.exists(f"{key}.mamba"), f"no sidecar stored at {key}"
            )
        self.assertNotIn(6 * _PAGE, attempts)

    def test_span_without_a_reachable_endpoint_is_revoked(self):
        # Only the deepest page is restorable and capacity cannot reach it.
        cc, attempts = _make_controller(capacity_tokens=6 * _PAGE)
        operation = _make_operation([_mamba_transfer()], [8])
        host_indices, alloc_len = self._allocate(
            cc, operation, 8 * _PAGE, _PAGE, mock.Mock()
        )
        self.assertIsNone(host_indices)
        self.assertEqual(alloc_len, 0)
        # No half-span launch that is doomed to miss the sidecar.
        self.assertEqual(attempts, [8 * _PAGE, 8 * _PAGE])

    def test_full_span_keeps_the_query_endpoint(self):
        cc, attempts = _make_controller(capacity_tokens=_TOKENS)
        operation = _make_operation([_mamba_transfer()], list(_SIDECAR_PAGES))
        host_indices, alloc_len = self._allocate(
            cc, operation, 8 * _PAGE, _PAGE, mock.Mock()
        )
        self.assertIsNotNone(host_indices)
        self.assertEqual(alloc_len, 8 * _PAGE)
        self.assertEqual(attempts, [8 * _PAGE])
        operation.hash_value = operation.hash_value[: alloc_len // _PAGE]
        keys = self._endpoint_keys(cc, operation, alloc_len)
        self.assertEqual(keys, [_HASHES[7]])
        self.assertTrue(self.backend.exists(f"{keys[0]}.mamba"))

    def test_no_trailing_pool_leaves_the_clamp_alone(self):
        cc, _ = _make_controller(capacity_tokens=6 * _PAGE)
        operation = _make_operation([], list(_SIDECAR_PAGES))
        host_indices, alloc_len = self._allocate(
            cc, operation, 8 * _PAGE, _PAGE, mock.Mock()
        )
        self.assertIsNotNone(host_indices)
        self.assertEqual(alloc_len, 6 * _PAGE)

    def test_unknown_candidates_do_not_constrain(self):
        # A backend that reports no stop points: only the folded boundary is
        # vouched for, so a shorter span must not be launched either.
        cc, _ = _make_controller(capacity_tokens=6 * _PAGE)
        operation = _make_operation([_mamba_transfer()], list(_SIDECAR_PAGES))
        operation.restorable_prefix_pages = None
        host_indices, alloc_len = self._allocate(
            cc, operation, 8 * _PAGE, _PAGE, mock.Mock()
        )
        self.assertIsNotNone(host_indices)
        self.assertEqual(alloc_len, 6 * _PAGE)

    def test_head_trimmed_span_keeps_the_pinned_endpoint(self):
        cc, _ = _make_controller(capacity_tokens=6 * _PAGE)
        operation = _make_operation([_mamba_transfer()], list(_SIDECAR_PAGES))
        # Buffer mode trims device-resident pages and pins the sidecar endpoint
        # at the untrimmed query boundary, which the query did verify.
        operation.sidecar_hash_values = list(_HASHES)
        operation.sidecar_hit_pages = _HIT_PAGES
        host_indices, alloc_len = self._allocate(
            cc, operation, 8 * _PAGE, _PAGE, mock.Mock()
        )
        self.assertIsNotNone(host_indices)
        self.assertEqual(alloc_len, 6 * _PAGE)


def _sync_endpoints_on_two_ranks(rank, rendezvous):
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        cc = HybridCacheController.__new__(HybridCacheController)
        cc.page_size = _PAGE
        cc.prefetch_hits_sync_groups = [None]
        # Per-rank shards: each rank holds a different interior endpoint, and
        # only page 8 is stored on both. Reducing per-rank maxima (the old
        # behaviour) would pick an endpoint the peer cannot read.
        local = [3, 8] if rank == 0 else [5, 8]
        operation = _make_operation([_mamba_transfer()], local)
        agreed = HybridCacheController._sync_prefetch_hit_query(
            cc, operation, min(local) * _PAGE
        )
        assert operation.restorable_prefix_pages == [8], (rank, operation)
        # The folded hit is the MIN of the local boundaries, unchanged by the
        # extra mask entries.
        assert agreed == 3 * _PAGE, (rank, agreed)
        endpoint = HybridCacheController._trailing_endpoint_tokens(cc, operation, 24)
        # Capacity reaches 6 pages on both ranks; neither can restore there, and
        # both have to agree to launch nothing rather than diverge.
        assert endpoint == 0, (rank, endpoint)
        gathered = [None, None]
        torch.distributed.all_gather_object(gathered, operation.restorable_prefix_pages)
        assert gathered[0] == gathered[1] == [8], gathered
    finally:
        torch.distributed.destroy_process_group()


class TestRankAgreedEndpoints(CustomTestCase):
    def test_two_ranks_intersect_their_stop_points(self):
        with tempfile.TemporaryDirectory(prefix="hicache-endpoint-ranks-") as tmp:
            torch.multiprocessing.spawn(
                _sync_endpoints_on_two_ranks,
                args=(f"{tmp}/store",),
                nprocs=2,
                join=True,
            )

    def test_mask_length_follows_the_span_not_the_hit(self):
        # A rank whose query was skipped (terminated operation) or failed must
        # still contribute the same collective shape.
        terminated = _make_operation([_mamba_transfer()], [])
        terminated.restorable_prefix_pages = None
        full = _make_operation([_mamba_transfer()], list(_SIDECAR_PAGES))
        self.assertEqual(
            len(hcc.local_restorable_mask(terminated, _PAGE, 0)),
            len(hcc.local_restorable_mask(full, _PAGE, _HIT_PAGES * _PAGE)),
        )
        self.assertEqual(
            hcc.local_restorable_mask(full, _PAGE, _HIT_PAGES * _PAGE),
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        )
        # No query evidence at all: only the folded boundary is offered.
        self.assertEqual(
            hcc.local_restorable_mask(terminated, _PAGE, 2 * _PAGE),
            [0, 0, 1] + [0] * (len(_HASHES) - 2),
        )


if __name__ == "__main__":
    unittest.main()
