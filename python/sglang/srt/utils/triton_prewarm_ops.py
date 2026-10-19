"""Dummy-launch the Triton specializations that prod serving hits after init.

Companion to :func:`sglang.srt.utils.triton_load_watch.prewarm_device_loads`.
That sweep can only device-load kernels whose ``CompiledKernel`` object already
exists; Triton builds that object lazily, at a specialization's first launch, so
a specialization that engine init never exercised still loads its cubin
mid-serving. The 2026-10-18/19 next9 logs are unambiguous about this: 92 late
device-load warnings and *zero* slow-compile warnings, i.e. every late load was
a Triton disk-cache hit whose object was constructed by the serving launch
itself.

Each recipe below launches a real operator with tiny throwaway tensors so the
specialization is built and loaded now, while engine init still owns the device
memory. Rules the recipes follow:

* Every argument list is copied from a **real call site or a real registered
  test**, named in the recipe's docstring. Triton keys a specialization on the
  constexpr values plus the ``{==1, divisible-by-16, other}`` classes of its int
  arguments, so an invented value does not warm the kernel serving will use — it
  loads a *different* cubin and wastes the memory this whole exercise is about.
  Where a constexpr is not resolvable from live state, the recipe sweeps the
  handful of values the call sites can produce.
* Nothing here touches request state, the KV cache, or the mamba pools: only
  small local tensors, freed when the recipe returns. Live objects (page size,
  pool geometry, the ``req_to_token`` table) are read, never written — the QSA
  graph-metadata kernels do write, but only into buffers this module allocated.
* Every recipe is individually guarded. A missing attribute, a changed operator
  signature, or a kernel that refuses to launch on this GPU downgrades to a
  debug log; boot never fails because of a prewarm.

Deliberately NOT warmed here, so the next reader does not mistake ``_RECIPES``
for an exhaustive list:

* ``_small_copy_kernel`` (kernels/ops/memory/small_copy.py) — its ``BIT_WIDTHS``
  constexpr tuple comes from the live CudaGraphBufferRegistry slot groups, so a
  dummy launch would have to reproduce that layout. Graph capture exercises it
  for the shapes it captures; the rest is left to the lazy path.
* ``fused_moe_kernel`` — one specialization per (block, expert, group, dtype)
  tuning config; the Max-Q bucket precompile is a separate owed lever.
* ``_qsa_mqa_prefill_kernel`` — needs the live indexer head count / head_dim and
  the row-count class; a guessed geometry loads a different cubin.
* ``_layer_norm_fwd_1pass_kernel``, ``_causal_conv1d_fwd_kernel`` — model-forward
  kernels whose config space (hidden width, num_warps, chunk/width constexprs) is
  only known to a real batch.
* ``_qsa_graph_row_metadata_kernel`` under QSA compressed sharding
  (``--qsa-cache-sharding-size > 1``): the live page table belongs to one shard,
  so the recipe declines instead of guessing its geometry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# {==1, %16, other} — those three classes are the entire int-argument
# specialization space, so this covers every batch size serving can present.
BS_BUCKETS: Tuple[int, ...] = (1, 2, 16)


@dataclass
class _Ctx:
    """Live geometry the recipes need, resolved once, all-optional."""

    device: torch.device
    page_size: int = 1
    req_to_token: Optional[torch.Tensor] = None
    req_to_token_width: Optional[int] = None
    max_cache_len: Optional[int] = None
    max_spec_len: Optional[int] = None
    vocab_size: Optional[int] = None
    logits_dtype: torch.dtype = torch.bfloat16
    qsa_pool: Optional[Any] = None
    qsa_max_pages: Optional[int] = None
    qsa_num_steps: Optional[int] = None


def prewarm_known_kernels(device: torch.device, worker: Any = None) -> List[str]:
    """Run every known-kernel recipe and return the names that launched.

    Called from ``triton_load_watch.prewarm_device_loads``; ``worker`` is the
    scheduler's model worker, used only to read pool/config geometry.
    """
    if not torch.cuda.is_available():
        return []
    if getattr(torch.version, "hip", None) is not None:
        return []

    ctx = _collect_ctx(device, worker)
    ran: List[str] = []
    for name, recipe in _RECIPES:
        try:
            recipe(ctx)
        except Exception as e:
            # Expected on any deployment that does not use this kernel: the
            # geometry is missing, or the operator's signature moved.
            logger.debug(
                "Triton prewarm: recipe '%s' did not run (%s: %s)",
                name,
                type(e).__name__,
                e,
            )
            continue
        ran.append(name)
    logger.info(
        "Triton prewarm: dummy-launched %d/%d known serving recipes (%s)",
        len(ran),
        len(_RECIPES),
        ", ".join(ran) if ran else "none",
    )
    return ran


def _collect_ctx(device: torch.device, worker: Any) -> _Ctx:
    ctx = _Ctx(device=device)
    runner = _resolve_model_runner(worker)
    if runner is None:
        return ctx

    server_args = getattr(runner, "server_args", None)
    if server_args is not None:
        ctx.page_size = max(1, int(getattr(server_args, "page_size", 1) or 1))
        ctx.max_spec_len = getattr(server_args, "speculative_num_draft_tokens", None)

    req_to_token = getattr(runner, "req_to_token", None)
    if isinstance(req_to_token, torch.Tensor) and req_to_token.ndim == 2:
        ctx.req_to_token = req_to_token
        ctx.req_to_token_width = req_to_token.shape[1]

    allocator = getattr(runner, "token_to_kv_pool_allocator", None)
    page_size = getattr(allocator, "page_size", None)
    if isinstance(page_size, int) and page_size > 0:
        ctx.page_size = page_size

    req_pool = getattr(runner, "req_to_token_pool", None)
    mamba_pool = getattr(req_pool, "mamba_pool", None)
    replayssm_d = getattr(mamba_pool, "replayssm_d", None)
    if isinstance(replayssm_d, torch.Tensor) and replayssm_d.ndim >= 2:
        # Same expression as the prod call site (speculative/spec_utils.py).
        ctx.max_cache_len = int(replayssm_d.shape[-2])

    model_config = getattr(runner, "model_config", None)
    ctx.vocab_size = getattr(model_config, "vocab_size", None)
    dtype = getattr(model_config, "dtype", None)
    if isinstance(dtype, torch.dtype):
        ctx.logits_dtype = dtype

    backend = getattr(runner, "attn_backend", None)
    pool = getattr(backend, "token_to_kv_pool", None)
    page_table = getattr(backend, "_graph_compressed_page_table", None)
    if pool is not None and isinstance(page_table, torch.Tensor):
        ctx.qsa_pool = pool
        ctx.qsa_max_pages = int(page_table.shape[1])
        # The multi-step draft wrapper holds one backend per draft step; the
        # target backend does not, so fall back to the configured step count.
        step_backends = getattr(backend, "attn_backends", None)
        if step_backends is not None:
            ctx.qsa_num_steps = len(step_backends)
        elif server_args is not None:
            num_steps = getattr(server_args, "speculative_num_steps", None)
            ctx.qsa_num_steps = int(num_steps) if num_steps else None
    return ctx


def _resolve_model_runner(worker: Any) -> Optional[Any]:
    """Unwrap ``TpModelWorker`` / the spec-worker wrapper to its ModelRunner."""
    if worker is None:
        return None
    runner = getattr(worker, "model_runner", None)
    if runner is not None:
        return runner
    inner = getattr(worker, "target_worker", None)
    return getattr(inner, "model_runner", None)


def _zeros(n: int, dtype: Any, device: torch.device) -> torch.Tensor:
    return torch.zeros(max(int(n), 1), dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# mem_cache: the two allocation-path kernels that loaded late in the 10-18 crash
# ---------------------------------------------------------------------------
def _prewarm_alloc_extend(ctx: _Ctx) -> None:
    """Call site: ``PagedTokenToKVPoolAllocator.alloc_extend``
    (srt/mem_cache/allocator/paged.py:246).

    constexprs are ``bs_upper=next_power_of_2(bs)`` and ``page_size`` (live).
    With ``prefix_lens == seq_lens == 0`` the kernel computes ``num_part1 = 0``,
    stores nothing, and returns at the part-1 boundary check, so the dummy launch
    neither reads ``free_pages`` nor writes ``out_indices``.
    """
    from sglang.kernels.ops.memory.allocator import alloc_extend_kernel
    from sglang.srt.utils.common import next_power_of_2

    page = int(ctx.page_size)
    for bs in BS_BUCKETS:
        prefix_lens = _zeros(bs, torch.int32, ctx.device)
        seq_lens = _zeros(bs, torch.int32, ctx.device)
        last_loc = _zeros(bs, torch.int32, ctx.device)
        # Sized like the allocator's own buffers rather than the (empty) result,
        # so the int64 index pointers match a real call.
        free_pages = _zeros(bs * page + 16, torch.int64, ctx.device)
        out_indices = _zeros(bs * page + 16, torch.int64, ctx.device)
        alloc_extend_kernel[(bs,)](
            prefix_lens,
            seq_lens,
            last_loc,
            free_pages,
            out_indices,
            next_power_of_2(bs),
            page,
        )


def _prewarm_assign_req_to_token_pool(ctx: _Ctx) -> None:
    """Call site: ``assign_req_to_token_pool_func``
    (srt/mem_cache/allocation.py:737).

    ``pool_len`` is a constexpr, taken from the live ``req_to_token`` width. The
    dummy rows have ``start_offset == end_offset == 0``, so ``num_loop`` is 0 and
    the kernel neither loads from ``out_cache_loc`` nor stores into the table;
    ``req_pool_indices`` are 0, so its address arithmetic stays inside row 0.
    """
    from sglang.srt.mem_cache.allocation import assign_req_to_token_pool_func

    width = int(ctx.req_to_token_width or 0)
    if width <= 0:
        raise RuntimeError("no req_to_token width available")
    for bs in BS_BUCKETS:
        req_pool_indices = _zeros(bs, torch.int64, ctx.device)
        start_offset = _zeros(bs, torch.int32, ctx.device)
        end_offset = _zeros(bs, torch.int32, ctx.device)
        out_cache_loc = _zeros(bs, torch.int64, ctx.device)
        # A 1-row table of the live width: only row 0 can be addressed here.
        table = torch.zeros((1, width), dtype=torch.int32, device=ctx.device)
        assign_req_to_token_pool_func(
            req_pool_indices,
            table,
            start_offset,
            end_offset,
            out_cache_loc,
            bs,
        )


# ---------------------------------------------------------------------------
# mamba / GDN cuda-graph replay metadata
# ---------------------------------------------------------------------------
def _prewarm_fused_replay_state_indices(ctx: _Ctx) -> None:
    """Operator: ``fused_replay_state_indices``
    (kernels/ops/mamba/mamba_state_indices_triton.py), called from
    ``MambaAttnBackendBase._replay_metadata``.

    constexprs are ``BS_UPPER=next_power_of_2(total_bs)`` and ``HAS_V2P``; both
    v2p forms are swept because which one a deployment uses depends on the
    memory-pool flavour. All tensors are local, and row 0 of the (4-element)
    mapping is a valid index. The padded-row variant (``valid_bs < total_bs``)
    shares the same key, so it is only launched where it costs nothing extra.
    """
    from sglang.kernels.ops.mamba.mamba_state_indices_triton import (
        fused_replay_state_indices,
    )

    for bs in BS_BUCKETS:
        mamba_index_mapping = _zeros(4, torch.int32, ctx.device)
        for v2p_len in (None, 2):
            v2p = None if v2p_len is None else _zeros(v2p_len, torch.int64, ctx.device)
            fused_replay_state_indices(
                req_pool_indices=_zeros(bs, torch.int64, ctx.device),
                mamba_index_mapping=mamba_index_mapping,
                out_state_indices=_zeros(bs, torch.int32, ctx.device),
                valid_bs=bs,
                total_bs=bs,
                v2p=v2p,
            )


def _gdn_cursors(ctx: _Ctx) -> Tuple[torch.Tensor, ...]:
    """Local stand-ins for ``MambaPool.replayssm_{spec_write_pos,cache_base,
    is_flush}`` (int32/int32/int8 per memory_pool.py:1179-1191)."""
    device = ctx.device
    return (
        torch.zeros(8, dtype=torch.int32, device=device),
        torch.zeros(8, dtype=torch.int32, device=device),
        torch.zeros(8, dtype=torch.int8, device=device),
    )


def _prewarm_gdn_spec_cursors(ctx: _Ctx) -> None:
    """Call site: ``commit_gdn_replayssm_spec`` (srt/speculative/spec_utils.py),
    which wraps ``_advance_gdn_spec_cursors_kernel``.

    ``MAX_CACHE_LEN`` comes from the live ring length and ``MAX_SPEC_LEN`` from
    ``speculative_num_draft_tokens``. ``FOLD_EVERY_COMMIT`` follows the checkpoint
    dtype at the call site, which is not readable without the spec state, so both
    values are swept (one is the real one; the other costs a few KB of cubin).
    With ``num_accepted == 0`` and ``is_flush == 0`` every store writes a zero
    over a zero in the local cursor tensors.
    """
    from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode import (
        commit_gdn_replayssm_spec,
    )

    max_cache_len = int(ctx.max_cache_len or 0)
    if max_cache_len <= 0 or not ctx.max_spec_len:
        raise RuntimeError("no replayssm spec geometry available")
    for bs in BS_BUCKETS:
        num_accepted = _zeros(bs, torch.int32, ctx.device)
        replay_indices = _zeros(bs, torch.int64, ctx.device)
        for fold in (False, True):
            write_pos, cache_base, is_flush = _gdn_cursors(ctx)
            commit_gdn_replayssm_spec(
                write_pos,
                cache_base,
                is_flush,
                num_accepted,
                replay_indices,
                max_cache_len,
                int(ctx.max_spec_len),
                fold_every_commit=fold,
                null_block_id=-1,
            )


def _prewarm_gdn_circular_fold(ctx: _Ctx) -> None:
    """Call site: the tail of ``commit_gdn_replayssm_circular``
    (gdn_replayssm_spec_decode.py:1531) launching
    ``_finish_gdn_replayssm_circular_fold_kernel``.

    Same constexpr family as the spec-cursor recipe (``MAX_CACHE_LEN``,
    ``BLOCK=next_power_of_2(B)``, ``NULL_BLOCK_ID=-1``). With ``is_flush == 0``
    the fold mask is empty, so nothing is stored. The sibling materialise kernel
    needs the whole d/k/g ring plus the layer/head dims, so only this cheap
    cursor-fold half is warmed.
    """
    import triton

    from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode import (
        _finish_gdn_replayssm_circular_fold_kernel,
    )

    max_cache_len = int(ctx.max_cache_len or 0)
    if max_cache_len <= 0:
        raise RuntimeError("no replayssm ring length available")
    for bs in BS_BUCKETS:
        write_pos, cache_base, is_flush = _gdn_cursors(ctx)
        state_batch_indices = _zeros(bs, torch.int32, ctx.device)
        replay_indices = _zeros(bs, torch.int64, ctx.device)
        _finish_gdn_replayssm_circular_fold_kernel[(1,)](
            write_pos,
            cache_base,
            is_flush,
            state_batch_indices,
            replay_indices,
            bs,
            stride_indices=state_batch_indices.stride(0),
            stride_replay_indices=replay_indices.stride(0),
            MAX_CACHE_LEN=max_cache_len,
            BLOCK=triton.next_power_of_2(max(1, bs)),
            NULL_BLOCK_ID=-1,
        )


# ---------------------------------------------------------------------------
# QSA cuda-graph metadata (the largest share of the prod late loads)
# ---------------------------------------------------------------------------
def _qsa_geometry(pool: Any) -> Tuple[int, int, int, int, int, int]:
    ratio = int(pool.qsa_compress_ratio)
    compressed_page = int(pool.qsa_compressed_page_size)
    num_groups = int(getattr(pool, "qsa_num_groups", 1) or 1)
    sharding = getattr(pool, "qsa_compressed_sharding", None)
    world_size = int(getattr(sharding, "world_size", 1) or 1)
    rank = int(getattr(sharding, "rank", 0) or 0)
    return ratio, compressed_page * ratio, num_groups, compressed_page, world_size, rank


def _qsa_metadata_factory(ctx: _Ctx, pool: Any, max_pages: int):
    """Build a self-contained metadata pair over locally allocated buffers.

    Field list copied from the validated recipe
    ``test/registered/kernels/ops/attention/qsa/test_qsa.py::
    test_qsa_graph_metadata_kernels_match_legacy_host_path`` (and mirrors
    ``QwenSparseAttnBackend._capture_cuda_graph_metadata``). Zeroed row inputs
    keep ``visible_pages`` at 0, so the page-table loop runs no iterations and the
    live ``req_to_token`` table is touched by exactly one read (row 0, token 0,
    the ``last_loc`` the kernel always loads) and never written; every buffer the
    kernels store into belongs to this prewarm.
    """
    from sglang.srt.layers.attention.qsa.metadata import QSAIndexerMetadata
    from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
        QwenSparseAttnMetadata,
    )

    ratio = int(pool.qsa_compress_ratio)
    compressed_page = int(pool.qsa_compressed_page_size)

    def make(rows: int):
        indexer = QSAIndexerMetadata(
            sequence_lengths=_zeros(rows, torch.int32, ctx.device),
            token_to_batch_idx=_zeros(rows, torch.int32, ctx.device),
            token_slot_table=torch.zeros(
                (rows, 1), dtype=torch.int32, device=ctx.device
            ),
            out_cache_loc=_zeros(rows, torch.int64, ctx.device),
            token_to_kv_pool=pool,
            compress_ratio=ratio,
            block_topk=pool.qsa_block_topk,
            req_pool_indices=_zeros(rows, torch.int32, ctx.device),
            is_cuda_graph=True,
            graph_write_locs=_zeros(rows, torch.int32, ctx.device),
            write_owner_mask=torch.zeros(rows, dtype=torch.bool, device=ctx.device),
            graph_compressed_page_table=torch.zeros(
                (rows, max_pages), dtype=torch.int32, device=ctx.device
            ),
            graph_compressed_lengths=_zeros(rows, torch.int32, ctx.device),
            graph_prefix_lengths=_zeros(rows, torch.int32, ctx.device),
            decode_block_positions=torch.full(
                (rows, max_pages * compressed_page),
                -1,
                dtype=torch.int32,
                device=ctx.device,
            ),
            decode_logical_positions=_zeros(rows, torch.int32, ctx.device),
            pending_ring_slots=_zeros(rows, torch.int64, ctx.device),
            graph_ring_group_locs=torch.zeros(
                (rows, ratio), dtype=torch.int32, device=ctx.device
            ),
        )
        return QwenSparseAttnMetadata(
            sequence_lengths=indexer.sequence_lengths,
            token_to_batch_idx=indexer.token_to_batch_idx,
            token_slot_table=indexer.token_slot_table,
            indexer_metadata=indexer,
            row_req_pool_indices=_zeros(rows, torch.int32, ctx.device),
            is_cuda_graph=True,
        )

    return make


def _prewarm_qsa_graph_metadata(ctx: _Ctx) -> None:
    """Operator: ``launch_graph_metadata``
    (srt/layers/attention/qsa/graph_metadata.py:209), the sync-free replay
    refresh that every QSA cuda-graph replay runs.

    constexprs RATIO / NUM_GROUPS / FULL_PAGE / COMPRESSED_PAGE / WORLD_SIZE /
    RANK come from the live QSA pool, and MODE from the three serving modes
    (0 decode, 1 target-verify, 2 draft-extend — see
    ``_replay_cuda_graph_metadata_gpu``). ``MAX_PAGES`` is a runtime int, so the
    live value is used directly. Recipe provenance: test_qsa.py::
    test_qsa_graph_metadata_kernels_match_legacy_host_path.
    """
    from sglang.srt.layers.attention.qsa.graph_metadata import launch_graph_metadata

    pool, max_pages = ctx.qsa_pool, ctx.qsa_max_pages
    if pool is None or not max_pages:
        raise RuntimeError("no QSA graph metadata state")
    if ctx.req_to_token is None:
        raise RuntimeError("no req_to_token table")
    _, _, _, _, world, rank = _qsa_geometry(pool)
    if world != 1 or rank != 0:
        raise RuntimeError("QSA compressed sharding not warmed")

    make = _qsa_metadata_factory(ctx, pool, int(max_pages))
    for bs in BS_BUCKETS:
        metadata = make(bs)
        seq_lens = _zeros(bs, torch.int32, ctx.device)
        req_pool_indices = _zeros(bs, torch.int32, ctx.device)
        extend_lens = _zeros(bs, torch.int32, ctx.device)
        for mode in (0, 1, 2):
            if mode == 1:
                # Mode 1 divides by extend_len; without a speculative draft
                # length the serving path cannot reach this mode either.
                if not ctx.max_spec_len:
                    continue
                extend_len = int(ctx.max_spec_len)
            else:
                extend_len = 0
            for num_padding in {0, 1, bs}:
                for seq_offset in (0, 1):
                    launch_graph_metadata(
                        mode=mode,
                        bs=bs,
                        num_rows=bs,
                        seq_lens=seq_lens,
                        req_pool_indices=req_pool_indices,
                        extend_lens=extend_lens if mode == 2 else None,
                        extend_len=extend_len,
                        num_padding=num_padding,
                        metadata=metadata,
                        req_to_token=ctx.req_to_token,
                        pool=pool,
                        seq_offset=seq_offset,
                    )


def _prewarm_qsa_draft_graph_metadata(ctx: _Ctx) -> None:
    """Operators: ``prepare_draft_graph_metadata`` +
    ``launch_draft_graph_metadata``, the NEXTN draft multi-step refresh
    (``init_forward_metadata_out_graph``, qwen_sparse_attn_backend.py:3486-3515).

    The step count is baked into the kernel (``tl.static_range(len(buffers))``),
    so it is read live from the draft wrapper where possible, else from
    ``speculative_num_steps``. Recipe provenance: test_qsa.py::
    test_qsa_draft_metadata_multi_step_graph.
    """
    from sglang.srt.layers.attention.qsa.graph_metadata import (
        launch_draft_graph_metadata,
        prepare_draft_graph_metadata,
    )

    pool, max_pages = ctx.qsa_pool, ctx.qsa_max_pages
    steps = int(ctx.qsa_num_steps or 0)
    if pool is None or not max_pages or steps <= 0 or ctx.req_to_token is None:
        raise RuntimeError("no QSA draft graph state")
    _, _, _, _, world, rank = _qsa_geometry(pool)
    if world != 1 or rank != 0:
        raise RuntimeError("QSA compressed sharding not warmed")
    make = _qsa_metadata_factory(ctx, pool, int(max_pages))

    for bs in BS_BUCKETS:
        metadata = tuple(make(bs) for _ in range(steps))
        args = prepare_draft_graph_metadata(metadata, ctx.req_to_token, pool)
        seq_lens = _zeros(bs, torch.int32, ctx.device)
        req_pool_indices = _zeros(bs, torch.int32, ctx.device)
        for num_padding in {0, 1, bs}:
            launch_draft_graph_metadata(
                args, seq_lens, req_pool_indices, bs, num_padding
            )


# ---------------------------------------------------------------------------
# grammar bitmask
# ---------------------------------------------------------------------------
def _prewarm_token_bitmask(ctx: _Ctx) -> None:
    """Call site: ``XGrammarGrammar.apply_vocab_mask``
    (srt/constrained/xgrammar_backend.py:134) — always with ``indices=None`` on
    CUDA, and the bitmask width is xgrammar's ``ceil(vocab / 32)``
    (``_allocate_token_bitmask``).

    The constexprs are ``NUM_SMS`` (device SM count — the same GPU here and at
    serving) and ``BLOCK_SIZE=4096``; ``num_warps`` comes from the logits element
    size, so the live model dtype is used. The varying ints are ``num_rows``,
    ``vocab_size`` and the two row strides, all derived from the live vocab size
    so their divisibility classes match. A -1 bitmask allows every token, so the
    kernel leaves the dummy logits untouched.
    """
    from sglang.kernels.ops.grammar.bitmask_ops import (
        apply_token_bitmask_inplace_triton,
    )

    vocab = int(ctx.vocab_size or 0)
    if vocab <= 0:
        raise RuntimeError("no vocab size available")
    width = (vocab + 31) // 32
    for bs in BS_BUCKETS:
        logits = torch.zeros((bs, vocab), dtype=ctx.logits_dtype, device=ctx.device)
        bitmask = torch.full((bs, width), -1, dtype=torch.int32, device=ctx.device)
        apply_token_bitmask_inplace_triton(logits, bitmask)


_RECIPES: List[Tuple[str, Callable[[_Ctx], None]]] = [
    ("alloc_extend_kernel", _prewarm_alloc_extend),
    ("assign_req_to_token_pool", _prewarm_assign_req_to_token_pool),
    ("_fused_replay_state_indices_kernel", _prewarm_fused_replay_state_indices),
    ("_advance_gdn_spec_cursors_kernel", _prewarm_gdn_spec_cursors),
    ("_finish_gdn_replayssm_circular_fold_kernel", _prewarm_gdn_circular_fold),
    ("_qsa_graph_metadata_kernels", _prewarm_qsa_graph_metadata),
    ("_qsa_draft_graph_metadata_kernel", _prewarm_qsa_draft_graph_metadata),
    ("apply_token_bitmask_inplace_kernel", _prewarm_token_bitmask),
]
