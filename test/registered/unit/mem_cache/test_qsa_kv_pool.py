import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.kv_cache_configurator import (
    _qsa_cache_sharding_graph_reservation_gb,
)
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_qsa_cache_sharding_budget_covers_graph_and_ple_prefill_output():
    reserve_gb = _qsa_cache_sharding_graph_reservation_gb(
        sharding_size=4,
        is_qsa=True,
        cuda_graph_enabled=True,
        prefill_tokens=8192,
        hidden_size=2560,
        hyper_connection_count=4,
        activation_element_size=2,
    )

    graph_private_gb = 4 * 0.5
    ple_tensor_gb = 8192 * 2560 * 4 * 2 / (1 << 30)
    assert reserve_gb == pytest.approx(graph_private_gb + 3 * ple_tensor_gb)


@pytest.mark.parametrize(
    ("sharding_size", "is_qsa", "cuda_graph_enabled"),
    [(1, True, True), (4, False, True), (4, True, False)],
)
def test_qsa_cache_sharding_budget_preserves_unsharded_or_eager_behavior(
    sharding_size, is_qsa, cuda_graph_enabled
):
    assert (
        _qsa_cache_sharding_graph_reservation_gb(
            sharding_size=sharding_size,
            is_qsa=is_qsa,
            cuda_graph_enabled=cuda_graph_enabled,
            prefill_tokens=8192,
            hidden_size=2560,
            hyper_connection_count=4,
            activation_element_size=2,
        )
        == 0.0
    )


def test_qsa_allocations_follow_parent_mooncake_scope(monkeypatch):
    active_scopes = set()
    allocations = 0
    original_zeros = torch.zeros

    @contextmanager
    def scope(name):
        active_scopes.add(name)
        try:
            yield
        finally:
            active_scopes.remove(name)

    def init_parent(pool, **_):
        pool.full_kv_pool = SimpleNamespace(
            memory_saver_adapter=SimpleNamespace(
                region=lambda _: scope("memory_saver")
            ),
            enable_custom_mem_pool=True,
            custom_mem_pool=object(),
        )

    def allocate(*args, **kwargs):
        nonlocal allocations
        assert active_scopes == {"memory_saver", "custom_pool"}
        allocations += 1
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(HybridLinearKVPool, "__init__", init_parent)
    monkeypatch.setattr(QSATokenToKVPool, "get_kv_size_bytes", lambda _: (0, 0))
    monkeypatch.setattr(torch.cuda, "use_mem_pool", lambda _: scope("custom_pool"))
    monkeypatch.setattr("sglang.srt.mem_cache.qsa_kv_pool.torch.zeros", allocate)

    QSATokenToKVPool(
        size=8,
        dtype=torch.bfloat16,
        page_size=4,
        head_num=1,
        head_dim=8,
        full_attention_layer_ids=[1, 3],
        device="cpu",
        mamba_pool=object(),
        qsa_index_kv_heads=1,
        qsa_index_head_dim=8,
        qsa_compress_ratio=2,
        qsa_token_topk=4,
        num_request_slots=3,
    )

    assert allocations == 4


def test_pending_ring_num_groups_covers_unaligned_windows():
    """Every group a verify window can touch, and never below the floor.

    A window that does not start on a group boundary still needs its first
    group's earlier members, so it spans ceil((draft + ratio - 1) / ratio)
    groups: 4 drafts at ratio 4 need 2 (one group let a draft row overwrite a
    member four positions earlier in the same forward). The floor is
    ``QSA_RING_GROUPS`` = 2 even for an absent draft-token count or a
    single-token decode window (PR #47051): the two-group ring is the layout
    that makes a rejected verify row's key self-restoring.
    """
    cases = [
        (0, 2),
        (1, 2),
        (2, 2),
        (4, 2),
        (5, 2),
        (6, 3),
        (8, 3),
        (9, 3),
        (16, 5),
    ]
    for max_num_draft_tokens, expected in cases:
        got = QSATokenToKVPool.pending_ring_num_groups(
            max_num_draft_tokens=max_num_draft_tokens, compress_ratio=4
        )
        assert got == expected, (max_num_draft_tokens, got, expected)


def _build_pool(monkeypatch, *, qsa_num_groups):
    """A QSA pool with the parent allocator stubbed, plus the widths it asked for."""
    widths = []
    original_zeros = torch.zeros

    @contextmanager
    def scope(_name):
        yield

    def init_parent(pool, **_):
        pool.full_kv_pool = SimpleNamespace(
            memory_saver_adapter=SimpleNamespace(region=scope),
            enable_custom_mem_pool=True,
            custom_mem_pool=object(),
        )

    def allocate(*args, **kwargs):
        widths.append(tuple(args[0]) if args else None)
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(HybridLinearKVPool, "__init__", init_parent)
    monkeypatch.setattr(QSATokenToKVPool, "get_kv_size_bytes", lambda _: (0, 0))
    monkeypatch.setattr(torch.cuda, "use_mem_pool", lambda _: scope("custom_pool"))
    monkeypatch.setattr("sglang.srt.mem_cache.qsa_kv_pool.torch.zeros", allocate)

    pool = QSATokenToKVPool(
        size=8,
        dtype=torch.bfloat16,
        page_size=4,
        head_num=1,
        head_dim=8,
        full_attention_layer_ids=[1, 3],
        device="cpu",
        mamba_pool=object(),
        qsa_index_kv_heads=1,
        qsa_index_head_dim=8,
        qsa_compress_ratio=2,
        qsa_token_topk=4,
        num_request_slots=3,
        qsa_num_groups=qsa_num_groups,
    )
    return pool, widths


def test_pending_ring_capacity_scales_with_num_groups(monkeypatch):
    """The ring holds ``num_request_slots * ratio * num_groups`` slots.

    Guards the capacity term a wider verify window depends on: dropping
    ``* qsa_num_groups`` still allocates a valid-looking buffer, so nothing
    else in the suite notices until a long window overwrites itself.
    """
    _, one = _build_pool(monkeypatch, qsa_num_groups=1)
    _, three = _build_pool(monkeypatch, qsa_num_groups=3)
    assert one[0][0] == 3 * 2 * 1
    assert three[0][0] == 3 * 2 * 3


def test_pending_state_transfer_item_covers_the_whole_ring(monkeypatch):
    """One transfer item is one request's whole ring, indexed by ``req_pool_idx``.

    The peer computes ``src + req_pool_idx * item_len``, so an item that covers
    only one group reads a *different request's* rows as soon as the ring holds
    more than one group. Both sides derive the length the same way, so nothing
    in the transfer raises -- the receiving side just gets the wrong keys.
    """
    pool, _ = _build_pool(monkeypatch, qsa_num_groups=3)
    pool.full_attention_layer_id_mapping = {1: 0, 3: 1}

    _, _, item_lens = pool.get_qsa_pending_state_buf_infos()

    # One request's ring is ``compress_ratio * num_groups`` rows; the pool holds
    # ``num_request_slots`` of them, and the item must be the former, not the latter
    # nor a single group.
    rows_per_request_ring = 2 * 3
    row_bytes = 1 * 8 * torch.empty(0, dtype=torch.bfloat16).element_size()
    assert item_lens[0] == rows_per_request_ring * row_bytes


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

# TODO(merge5): entry 125/200 + 285-287 ring tests KEPT past upstream #45672
# prune (mirror cases only upstream; ring/sharding cases are fork-owned);
# re-derive register_cpu_ci est + prune audit (batch B2).
