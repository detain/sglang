"""Fused QSA indexer-prep kernels must match the eager indexer path bit-for-bit,
up to rare last-ulp RMSNorm flips (see assert_bit_comparable). With an fp8
indexer cache the same rows are cast to e4m3 at the store, so a bf16 last-ulp
flip may move the e4m3 code by one step (see assert_fp8_within_one_ulp)."""

from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from sglang.srt.layers.attention.qsa.kernel import (
    average_pool_qsa_keys,
    expand_qsa_block_indices,
    torch_expand_qsa_block_indices,
)
from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs

HEAD_DIM = 128
NUM_Q_HEADS = 4
RATIO = 4
HIDDEN = 2560
EPS = 1e-6


@pytest.fixture(autouse=True)
def runtime_context():
    publish(ServerArgs(model_path="dummy"), role="test")
    try:
        yield
    finally:
        reset_context()


def _make_config():
    return SimpleNamespace(
        indexer_n_heads=NUM_Q_HEADS,
        indexer_kv_heads=1,
        indexer_head_dim=HEAD_DIM,
        indexer_budget=2048,
        indexer_compress_ratio=RATIO,
        hidden_size=HIDDEN,
        rms_norm_eps=EPS,
    )


def _make_rotary(mrope_section, mrope_interleaved, device, dtype=torch.bfloat16):
    return MRotaryEmbedding(
        head_size=HEAD_DIM,
        rotary_dim=HEAD_DIM,
        max_position_embeddings=32768,
        base=1000000,
        is_neox_style=True,
        dtype=dtype,
        mrope_section=mrope_section,
        mrope_interleaved=mrope_interleaved,
    )


def _make_indexer(rotary, device, dtype=torch.bfloat16):
    # Build under the model dtype like ModelRunner does; device-only .to()
    # afterwards so the fp32 cos_sin_cache buffer keeps its dtype.
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        indexer = QSAIndexer(
            _make_config(), layer_id=0, quant_config=None, rotary_emb=rotary
        )
        indexer.to(device=device)
    finally:
        torch.set_default_dtype(prev_dtype)
    with torch.no_grad():
        out_features = (NUM_Q_HEADS + 1) * HEAD_DIM
        indexer.index_qk_proj.weight.data.copy_(
            torch.randn(out_features, HIDDEN, device=device, dtype=dtype) * 0.02
        )
        for norm in (indexer.q_layernorm, indexer.k_layernorm):
            w = torch.randn(HEAD_DIM, device=device, dtype=dtype) * 0.1
            norm._weight_loader(norm.weight, w)
    return indexer


class FakePool:
    """Minimal stand-in for the QSA KV pool buffers used by the indexer."""

    index_state_dtype = torch.bfloat16

    def __init__(
        self,
        num_slots,
        num_compressed,
        device,
        dtype=torch.bfloat16,
        compressed_dtype=None,
    ):
        # Ring in the compute dtype, compressed cache and index Q in the storage dtype.
        self.qsa_compressed_dtype = compressed_dtype or dtype
        self.key_state = torch.zeros(num_slots, 1, HEAD_DIM, dtype=dtype, device=device)
        self.qsa_rope_position_buffer = torch.zeros(
            num_slots, 3, dtype=torch.int64, device=device
        )
        self.compressed = torch.zeros(
            num_compressed, 1, HEAD_DIM, dtype=self.qsa_compressed_dtype, device=device
        )

    def get_qsa_key_state_buffer(self, layer_id):
        return self.key_state

    def set_qsa_key_state_buffer(self, layer_id, loc, token_k):
        self.key_state[loc.long()] = token_k.to(self.key_state.dtype)

    def set_qsa_rope_position_buffer(self, loc, positions):
        positions = positions.long()
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(3, -1)
        self.qsa_rope_position_buffer[loc.long()] = positions.transpose(0, 1)

    def get_qsa_rope_position_buffer(self, loc):
        return self.qsa_rope_position_buffer[loc.long()]

    def get_qsa_compressed_k_buffer(self, layer_id):
        return self.compressed

    def set_qsa_compressed_k_buffer(self, layer_id, loc, compressed_k):
        self.compressed[loc.long()] = compressed_k.to(self.compressed.dtype)


def assert_bit_comparable(actual, expected, max_frac=1e-5, max_abs=0.02):
    """Eager RMSNorm (flashinfer CuTe DSL) reduces in an unreproducible order,
    so ~1 row in 30k flips by 1-2 bf16 ulp; max_frac and max_abs bound that."""
    diff = (actual.float() - expected.float()).abs()
    mismatches = int((diff > 0).sum())
    allowed = max(16, int(max_frac * actual.numel()))
    assert mismatches <= allowed, f"{mismatches} mismatched elements"
    if mismatches:
        peak = diff.max().item()
        assert peak <= max_abs, f"largest deviation {peak} exceeds {max_abs}"


def assert_fp8_within_one_ulp(actual, expected):
    """One e4m3 code step, or 2^-8 absolute for subnormals and sign crossings."""
    assert actual.dtype == expected.dtype == torch.float8_e4m3fn
    code_diff = (
        actual.view(torch.uint8).int() - expected.view(torch.uint8).int()
    ).abs()
    abs_diff = (actual.float() - expected.float()).abs()
    bad = int((~((code_diff <= 1) | (abs_diff <= 2**-8))).sum())
    assert bad == 0, f"{bad} fp8 elements differ by more than one ulp"


def _eager_compress_reference(indexer, pool, group_locs, write_locs):
    """The pre-fusion compression chain, via the indexer's own helpers."""
    key_groups = pool.get_qsa_key_state_buffer(0)[group_locs.long()]
    pooled = average_pool_qsa_keys(key_groups)
    rope_positions = indexer._rope_from_matrix(
        pool.get_qsa_rope_position_buffer(group_locs[:, 0])
    )
    normalized = indexer.normalize_compressed_keys(pooled, rope_positions)
    pool.set_qsa_compressed_k_buffer(0, write_locs, normalized)


@pytest.mark.parametrize("num_groups", [1, 5, 2000])
@pytest.mark.parametrize(
    "mrope_section, mrope_interleaved",
    [([24, 20, 20], True), ([24, 20, 20], False), (None, False)],
)
@pytest.mark.parametrize("compressed_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_fused_compress_matches_eager(
    num_groups, mrope_section, mrope_interleaved, compressed_dtype
):
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(num_groups)
    rotary = _make_rotary(mrope_section, mrope_interleaved, device, dtype)
    indexer = _make_indexer(rotary, device, dtype)

    pool_ref = FakePool(8192, 4096, device, dtype, compressed_dtype)
    pool_new = FakePool(8192, 4096, device, dtype, compressed_dtype)
    pool_new.key_state.copy_(
        pool_ref.key_state.copy_(
            torch.randn(8192, 1, HEAD_DIM, device=device, dtype=dtype)
        )
    )
    positions = torch.randint(0, 30000, (8192, 3), device=device)
    pool_new.qsa_rope_position_buffer.copy_(positions)
    pool_ref.qsa_rope_position_buffer.copy_(positions)

    # Random groups; slot 0 doubles as the CUDA-graph dummy write target, so
    # allow repeats there too.
    group_locs = torch.randint(0, 8192, (num_groups, RATIO), device=device).to(
        torch.int32
    )
    write_locs = torch.randperm(4096, device=device)[:num_groups].to(torch.int32)

    _eager_compress_reference(indexer, pool_ref, group_locs, write_locs)
    indexer._fused_compress_store(pool_new, group_locs, write_locs)

    if compressed_dtype == torch.float8_e4m3fn:
        assert_fp8_within_one_ulp(pool_new.compressed, pool_ref.compressed)
    else:
        assert_bit_comparable(pool_new.compressed, pool_ref.compressed)


def test_fused_q_prep_fp8_matches_eager_cast():
    """fp8 Q from the fused prep must match the eager Q cast at the end (one ulp)."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(11)
    rotary = _make_rotary([24, 20, 20], True, device, dtype)
    indexer = _make_indexer(rotary, device, dtype)
    tokens = 300
    hidden = torch.randn(tokens, HIDDEN, device=device, dtype=dtype)
    positions = (
        torch.arange(5000, 5000 + tokens, device=device)
        .unsqueeze(0)
        .expand(3, -1)
        .contiguous()
    )
    qk, _ = indexer.index_qk_proj(hidden)
    q_ref = indexer.q_layernorm(qk[:, : NUM_Q_HEADS * HEAD_DIM].reshape(-1, HEAD_DIM))
    q_ref = indexer.apply_rope(positions, q_ref.reshape(tokens, NUM_Q_HEADS, HEAD_DIM))
    pool = FakePool(4096, 4096, device, dtype, torch.float8_e4m3fn)
    cache_loc = torch.arange(1, tokens + 1, device=device)
    q_new, _, stored = indexer.project_qk(
        hidden, positions, pool=pool, cache_loc=cache_loc
    )
    assert stored
    assert q_new.dtype == torch.float8_e4m3fn
    assert_fp8_within_one_ulp(q_new, q_ref.to(torch.float8_e4m3fn))


def _selected_score_multisets(logits, block_indices):
    """Sorted logits of the selected blocks per row, so tied blocks compare equal."""
    out = []
    for row in range(block_indices.shape[0]):
        picked = block_indices[row][block_indices[row] >= 0].long()
        out.append(logits[row, picked].sort().values)
    return out


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_tilelang_mqa_matches_torch_reference(dtype):
    """TileLang scoring vs the fp32 torch reference on the same operands. The
    selected blocks are compared as score multisets because fp8 ties at the
    top-k boundary."""
    from sglang.srt.layers.attention.qsa.kernel import qsa_fast_topk
    from sglang.srt.layers.attention.qsa.mqa import (
        HAS_TILELANG,
        tilelang_qsa_mqa_decode,
        tilelang_qsa_mqa_prefill,
        torch_qsa_mqa_decode,
        torch_qsa_mqa_prefill,
    )

    if not HAS_TILELANG:
        pytest.skip("tilelang unavailable")
    device = torch.device("cuda")
    torch.manual_seed(3)
    # Decode: paged compressed cache. qsa_fast_topk only supports the production
    # block budget (512).
    batch, max_pages, page_size, block_topk = 6, 256, 16, 512
    q = torch.randn(batch, NUM_Q_HEADS, HEAD_DIM, device=device).to(dtype)
    cache = torch.randn(512, page_size, 1, HEAD_DIM, device=device).to(dtype)
    page_table = torch.stack(
        [torch.randperm(512, device=device)[:max_pages] for _ in range(batch)]
    ).to(torch.int32)
    context_lens = torch.randint(
        600, max_pages * page_size + 1, (batch,), device=device
    ).to(torch.int32)
    max_model_len = max_pages * page_size
    ref = torch_qsa_mqa_decode(q, cache, page_table, context_lens, max_model_len)
    out = tilelang_qsa_mqa_decode(q, cache, page_table, context_lens, max_model_len)
    finite = torch.isfinite(ref)
    assert torch.equal(finite, torch.isfinite(out))
    torch.testing.assert_close(out[finite], ref[finite], rtol=2e-3, atol=2e-3)
    row_starts = torch.zeros_like(context_lens)
    sel_ref = _selected_score_multisets(
        ref, qsa_fast_topk(ref, row_starts, context_lens, topk=block_topk)
    )
    sel_out = _selected_score_multisets(
        ref, qsa_fast_topk(out, row_starts, context_lens, topk=block_topk)
    )
    for a, b in zip(sel_ref, sel_out):
        torch.testing.assert_close(a, b, rtol=2e-3, atol=2e-3)
    # Prefill: packed keys with ragged causal ranges.
    rows, keys = 70, 512
    qp = torch.randn(rows, NUM_Q_HEADS, HEAD_DIM, device=device).to(dtype)
    kp = torch.randn(keys, 1, HEAD_DIM, device=device).to(dtype)
    row_starts = torch.zeros(rows, dtype=torch.int32, device=device)
    row_ends = torch.randint(1, keys + 1, (rows,), device=device).to(torch.int32)
    ref_p = torch_qsa_mqa_prefill(qp, kp, row_starts, row_ends)
    out_p = tilelang_qsa_mqa_prefill(qp, kp, row_starts, row_ends)
    finite = torch.isfinite(ref_p)
    assert torch.equal(finite, torch.isfinite(out_p))
    torch.testing.assert_close(out_p[finite], ref_p[finite], rtol=2e-3, atol=2e-3)


def test_tilelang_mqa_rejects_mixed_operand_dtypes():
    """Mixed fp8/bf16 operands must fail instead of dequantizing the cache."""
    from sglang.srt.layers.attention.qsa.mqa import (
        HAS_TILELANG,
        tilelang_qsa_mqa_decode,
    )

    if not HAS_TILELANG:
        pytest.skip("tilelang unavailable")
    device = torch.device("cuda")
    q = torch.randn(2, NUM_Q_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    cache = torch.randn(8, 16, 1, HEAD_DIM, device=device).to(torch.float8_e4m3fn)
    page_table = torch.zeros(2, 4, dtype=torch.int32, device=device)
    context_lens = torch.full((2,), 32, dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="same dtype"):
        tilelang_qsa_mqa_decode(q, cache, page_table, context_lens, 64)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_expand_block_indices_int_inputs(dtype):
    device = torch.device("cuda")
    torch.manual_seed(0)
    rows, block_topk, token_topk, ratio = 37, 512, 2048, 4
    query_positions = torch.randint(0, 8000, (rows,), dtype=dtype, device=device)
    sequence_lengths = (
        query_positions + torch.randint(1, 9, (rows,), dtype=dtype, device=device)
    ).to(dtype)
    # Production contract: top-k only selects blocks inside [0, seq_len//4),
    # so no selected block ever masks out against sequence_lengths.
    counts = torch.randint(0, block_topk + 1, (rows,))
    block_indices = torch.full((rows, block_topk), -1, dtype=torch.int32)
    seq_lens_host = sequence_lengths.cpu()
    for r in range(rows):
        limit = max(int(seq_lens_host[r]) // ratio, 1)
        count = min(int(counts[r]), limit)
        if count:
            block_indices[r, :count] = torch.randperm(limit)[:count].to(torch.int32)
    block_indices = block_indices.to(device)
    out = expand_qsa_block_indices(
        block_indices, query_positions, sequence_lengths, ratio, token_topk
    )
    ref = torch_expand_qsa_block_indices(
        block_indices.cpu(),
        query_positions.cpu(),
        sequence_lengths.cpu(),
        ratio,
        token_topk,
    )
    assert torch.equal(out.cpu(), ref)


def test_decode_selection_equivalent():
    """Last-ulp norm flips must not change the selected blocks:
    scores are fp32 sums of 128-dim dots, so a 1-ulp flip only matters on exact ties."""
    from sglang.srt.layers.attention.qsa.kernel import qsa_fast_topk
    from sglang.srt.layers.attention.qsa.mqa import torch_qsa_mqa_decode

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(7)
    rotary = _make_rotary([24, 20, 20], True, device, dtype)
    indexer = _make_indexer(rotary, device, dtype)

    batch, max_pages, page_size = 4, 32, 64
    max_model_len = max_pages * page_size
    hidden = torch.randn(batch, HIDDEN, device=device, dtype=dtype)
    positions = (
        torch.arange(8000, 8000 + batch, device=device)
        .unsqueeze(0)
        .expand(3, -1)
        .contiguous()
    )
    qk, _ = indexer.index_qk_proj(hidden)

    # Eager index q.
    q_ref = indexer.q_layernorm(qk[:, : NUM_Q_HEADS * HEAD_DIM].reshape(-1, HEAD_DIM))
    q_ref = q_ref.reshape(batch, NUM_Q_HEADS, HEAD_DIM)
    q_ref = indexer.apply_rope(positions, q_ref)

    # Fused index q.
    pool = FakePool(64, 4096, device, dtype)
    cache_loc = torch.arange(1, batch + 1, device=device)
    q_new, _, stored = indexer.project_qk(
        hidden, positions, pool=pool, cache_loc=cache_loc
    )
    assert stored

    compressed_cache = torch.randn(
        64, page_size, 1, HEAD_DIM, device=device, dtype=dtype
    )
    page_table = torch.arange(max_pages, dtype=torch.int32, device=device).repeat(
        batch, 1
    )
    context_lens = torch.full((batch,), 1500, dtype=torch.int32, device=device)

    def select(q):
        logits = torch_qsa_mqa_decode(
            q, compressed_cache, page_table, context_lens, max_model_len
        )
        row_starts = torch.zeros_like(context_lens)
        return qsa_fast_topk(logits, row_starts, context_lens, topk=512)

    idx_ref = select(q_ref)
    idx_new = select(q_new[:, :NUM_Q_HEADS].contiguous())
    for row in range(batch):
        ref_set = set(idx_ref[row][idx_ref[row] >= 0].tolist())
        new_set = set(idx_new[row][idx_new[row] >= 0].tolist())
        assert ref_set == new_set, f"row {row}: selection mismatch"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))


@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("prefix_members", [1, 2, 3])
def test_cross_prefix_group_matches_in_chunk_compress(fused, prefix_members, monkeypatch):
    """A group crossing an unaligned extend prefix (a mixed-chunk decode tail,
    a private chunk-cache tail) recompresses its prefix-side members from the
    pending ring; the result must equal the same group compressed with every
    member in one chunk. A second, aligned row in the batch is unaffected."""
    from sglang.srt.layers.attention.qsa.metadata import (
        build_group_ring_slots,
        build_pending_ring_slots,
        build_rope_position_matrix,
    )

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(prefix_members)
    rotary = _make_rotary([24, 20, 20], True, device, dtype)
    indexer = _make_indexer(rotary, device, dtype)
    if not fused:
        monkeypatch.setattr(indexer, "_use_fused_compress", lambda pool: False)

    # Row 0: request slot 1, group [36, 40) crosses its prefix; the current
    # chunk is 4 - prefix_members tokens ending at the group end.
    # Row 1: request slot 2, an aligned 8-token prefill (two groups).
    group_start = 36
    prefix = group_start + prefix_members
    tail_len = RATIO - prefix_members
    extend_lens = [tail_len, 8]
    prefix_lens = [prefix, 0]
    req_pool_indices = torch.tensor([1, 2], device=device)
    keys = torch.randn(RATIO + 8, 1, HEAD_DIM, device=device, dtype=dtype)
    group_keys, prefill_keys = keys[:RATIO], keys[RATIO:]
    rope = torch.stack(
        [
            torch.arange(group_start, group_start + RATIO, device=device),
            torch.arange(group_start, group_start + RATIO, device=device) + 7,
            torch.arange(group_start, group_start + RATIO, device=device) + 11,
        ]
    )

    pool = FakePool(64, 64, device, dtype)
    # The earlier forwards left the prefix members pending in the ring.
    ring = build_group_ring_slots(
        req_pool_indices=req_pool_indices,
        group_end_positions=torch.tensor([group_start + RATIO - 1], device=device),
        sequence_ids=torch.tensor([0], device=device),
        compress_ratio=RATIO,
    )
    pool.set_qsa_key_state_buffer(0, ring[0, :prefix_members], group_keys[:prefix_members])
    pool.set_qsa_rope_position_buffer(ring[0, :prefix_members], rope[:, :prefix_members])

    token_k = torch.cat([group_keys[prefix_members:], prefill_keys])
    row_rope = torch.cat(
        [rope[:, prefix_members:], torch.arange(8, device=device).expand(3, -1)], dim=1
    )
    logical = torch.cat(
        [
            torch.arange(prefix, prefix + tail_len, device=device),
            torch.arange(8, device=device),
        ]
    )
    token_to_batch_idx = torch.repeat_interleave(
        torch.arange(2, device=device), torch.tensor(extend_lens, device=device)
    )
    seq_lens = torch.tensor(
        [p + e for p, e in zip(prefix_lens, extend_lens)], device=device
    )
    write_locs = torch.tensor([9, 10, 11], dtype=torch.int32, device=device)
    group_ends = torch.tensor([group_start + RATIO - 1, 3, 7], device=device)
    seq_ids = torch.tensor([0, 1, 1], device=device)
    metadata = SimpleNamespace(
        token_to_kv_pool=pool,
        token_to_batch_idx=token_to_batch_idx,
        req_pool_indices=req_pool_indices,
        sequence_lengths=seq_lens,
        is_cuda_graph=False,
        write_locs=write_locs,
        compress_group_positions=group_ends,
        compress_sequence_ids=seq_ids,
        compress_member_rows=torch.tensor(
            [-prefix_members, tail_len, tail_len + 4], device=device
        ),
        compress_prefix_members=torch.tensor([prefix_members, 0, 0], device=device),
        has_cross_prefix_group=True,
        compress_group_ring_locs=build_group_ring_slots(
            req_pool_indices=req_pool_indices,
            group_end_positions=group_ends,
            sequence_ids=seq_ids,
            compress_ratio=RATIO,
        ),
        extend_rope_matrix=build_rope_position_matrix(row_rope, token_k.shape[0]),
    )
    state_slots = build_pending_ring_slots(
        token_to_batch_idx=token_to_batch_idx,
        req_pool_indices=req_pool_indices,
        sequence_lengths=seq_lens,
        logical_positions=logical,
        compress_ratio=RATIO,
        is_extend=True,
    )
    indexer.update_key_state_and_compress(
        token_k, logical, row_rope, metadata, state_slots=state_slots
    )

    # Reference: every member in one chunk, compressed by the same path.
    ref_pool = FakePool(64, 64, device, dtype)
    ref_pool.key_state[: RATIO + 8] = keys
    ref_pool.qsa_rope_position_buffer[:RATIO] = rope.transpose(0, 1)
    ref_pool.qsa_rope_position_buffer[RATIO : RATIO + 8] = torch.arange(
        8, device=device
    )[:, None].expand(-1, 3)
    ref_locs = torch.arange(RATIO + 8, device=device, dtype=torch.int32).view(3, RATIO)
    if fused:
        indexer._fused_compress_store(ref_pool, ref_locs, write_locs)
    else:
        _eager_compress_reference(indexer, ref_pool, ref_locs, write_locs)

    assert_bit_comparable(pool.compressed[9:12], ref_pool.compressed[9:12])
    # The crossing group must not be the clamped main-pass result.
    assert pool.compressed[9].abs().sum() > 0


@pytest.mark.parametrize("num_groups", [1, 2])
def test_verify_windows_compress_committed_keys(num_groups):
    """MTP verify writes every draft row to the pending ring before it
    compresses the groups those rows complete. With a one-group ring, a
    window starting at base % 4 != 0 overwrote the slot of a member four
    positions earlier in the same forward, so the committed group was
    compressed from a draft's key. A ring spanning every group the window
    touches (2 for 4 drafts at ratio 4) must reproduce the committed keys."""
    from sglang.srt.layers.attention.qsa.metadata import (
        build_group_ring_slots,
        build_pending_ring_slots,
    )
    from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

    device = torch.device("cuda")
    dtype = torch.bfloat16
    draft = 4
    assert (
        QSATokenToKVPool.pending_ring_num_groups(
            max_num_draft_tokens=draft, compress_ratio=RATIO
        )
        == 2
    )
    torch.manual_seed(0)
    rng = torch.Generator().manual_seed(1)
    rotary = _make_rotary([24, 20, 20], True, device, dtype)
    indexer = _make_indexer(rotary, device, dtype)
    total = 400
    true_keys = torch.randn(total + draft, 1, HEAD_DIM, device=device, dtype=dtype)
    req = torch.tensor([1], device=device)
    pool = FakePool(64, 256, device, dtype)

    def ring(positions):
        return build_pending_ring_slots(
            token_to_batch_idx=torch.zeros_like(positions),
            req_pool_indices=req,
            sequence_lengths=positions + 1,
            logical_positions=positions,
            compress_ratio=RATIO,
            is_extend=False,
            num_groups=num_groups,
        )

    # A prefill left the first pending tail in the ring.
    base = 37
    tail = torch.arange((base // RATIO) * RATIO, base, device=device)
    pool.set_qsa_key_state_buffer(0, ring(tail), true_keys[tail])
    pool.set_qsa_rope_position_buffer(ring(tail), tail)

    while base < total:
        positions = torch.arange(base, base + draft, device=device)
        accept = int(torch.randint(1, draft + 1, (1,), generator=rng))
        keys = true_keys[positions].clone()
        keys[accept:] = torch.randn_like(keys[accept:])  # rejected drafts
        ends = positions[(positions + 1) % RATIO == 0]
        metadata = SimpleNamespace(
            token_to_kv_pool=pool,
            token_to_batch_idx=torch.zeros(draft, dtype=torch.long, device=device),
            req_pool_indices=req,
            sequence_lengths=positions + 1,
            is_cuda_graph=False,
            write_locs=(ends // RATIO + 1).to(torch.int32),
            compress_group_positions=ends,
            compress_sequence_ids=torch.zeros_like(ends),
            compress_member_rows=None,
            compress_group_ring_locs=build_group_ring_slots(
                req_pool_indices=req,
                group_end_positions=ends,
                sequence_ids=torch.zeros_like(ends),
                compress_ratio=RATIO,
                num_groups=num_groups,
            ),
        )
        indexer.update_key_state_and_compress(
            keys, positions, positions, metadata, state_slots=ring(positions)
        )
        base += accept

    ref = FakePool(total + draft, 256, device, dtype)
    ref.key_state[: total + draft] = true_keys
    ref.qsa_rope_position_buffer[: total + draft] = torch.arange(
        total + draft, device=device
    )[:, None].expand(-1, 3)
    groups = torch.arange(40 // RATIO, base // RATIO - 1, device=device)
    member_locs = (groups[:, None] * RATIO + torch.arange(RATIO, device=device)).to(
        torch.int32
    )
    indexer._fused_compress_store(ref, member_locs, (groups + 1).to(torch.int32))

    got, want = pool.compressed[groups + 1], ref.compressed[groups + 1]
    bad = int(((got.float() - want.float()).abs().amax(dim=(1, 2)) > 0.05).sum())
    if num_groups == 1:
        print(f"one-group ring: {bad}/{groups.numel()} committed groups corrupted")
        assert bad > 0, "one-group ring should reproduce the verify collision"
    else:
        assert bad == 0, f"{bad} committed groups compressed from overwritten keys"
