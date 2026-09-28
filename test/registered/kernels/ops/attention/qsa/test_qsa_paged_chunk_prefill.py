"""Paged chunk-prefill kernel tests: pool addressing vs the packed oracle.

The oracle is the existing packed path: densify the pool rows the same way
``QwenSparseAttnBackend._build_chunk_prefill_shared`` does and run
``sparse_gqa_fwd_interface_triton_ck`` on them.  Both paths then feed the
identical bytes through the identical arithmetic, so results must match with
``torch.equal`` -- exact, zero tolerance.  Any difference is an indexing bug.
"""

import sys

import pytest
import torch
from sglang.srt.layers.attention.qsa.sparse_attn import (
    sparse_gqa_fwd_interface_triton_ck,
    sparse_gqa_fwd_interface_triton_paged,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="paged chunk-prefill requires CUDA"
)

HEAD_DIM = 256
TOPK = 64
SCALE = HEAD_DIM**-0.5
# Smallest fp8 pool (1 KV head x 256 columns) whose LAST element index passes
# 2**31: 8_388_609 rows suffice; round up for margin and keep 8.4M.
LARGE_POOL_SLOTS = 8_400_000


def _slot_plan(kv_lens, num_slots, page_size, device):
    """Per-request slot vectors, indexed by logical position.

    ``page_size=None``: fully scattered (random permutation draw), the worst
    case for coalescing and the one that catches packed-vs-page confusions.
    ``page_size=P``: contiguous runs of up to P slots per request page with
    two gap pages between requests -- the distribution the real allocator
    produces with ``--page-size P``.
    """
    if page_size is None:
        perm = torch.randperm(num_slots, device=device)[: sum(kv_lens)]
        per_request, off = [], 0
        for kv_len in kv_lens:
            per_request.append(perm[off : off + kv_len])
            off += kv_len
    else:
        # Runs are page-aligned and requests never share a page.
        per_request, next_page = [], 0
        for kv_len in kv_lens:
            pages = (kv_len + page_size - 1) // page_size
            slots = []
            for p in range(pages):
                begin = (next_page + p) * page_size
                take = min(page_size, kv_len - p * page_size)
                slots.extend(range(begin, begin + take))
            per_request.append(torch.tensor(slots, dtype=torch.int32, device=device))
            next_page += pages + 2
        max_slot = max(int(s.max()) for s in per_request)
        assert max_slot < num_slots, (
            f"slot plan overflows pool: {max_slot} >= {num_slots}"
        )
    gather_index = (
        per_request[0]
        if len(per_request) == 1
        else torch.cat(per_request).to(torch.long)
    )
    return per_request, gather_index


def _make_case(
    q_lens,
    prefix_lens,
    *,
    dtype,
    page_size=None,
    num_slots=1024,
    req_rows=(3, 5, 1, 7),
    seed=0,
    k_scale=0.25,
    v_scale=0.5,
    tail_slots=False,
):
    """Build pool + tables + packed oracle inputs for one batch.

    Returns a dict with everything the two wrappers need.  The packed K/V is
    densified with ``index_select`` on the joined gather index -- exactly the
    current backend construction.
    """
    device = torch.device("cuda")
    torch.manual_seed(seed)
    kv_lens = [p + q for p, q in zip(prefix_lens, q_lens)]
    total_q, total_k = sum(q_lens), sum(kv_lens)

    q = torch.randn(total_q, 6, HEAD_DIM, dtype=torch.bfloat16, device=device)

    if dtype == torch.bfloat16:
        pool_k = torch.randn(num_slots, 1, HEAD_DIM, dtype=dtype, device=device)
        pool_v = torch.randn(num_slots, 1, HEAD_DIM, dtype=dtype, device=device)
    else:
        pool_k = torch.zeros(num_slots, 1, HEAD_DIM, dtype=dtype, device=device)
        pool_v = torch.zeros(num_slots, 1, HEAD_DIM, dtype=dtype, device=device)

    max_ctx = max(kv_lens)
    req_idx = torch.tensor(
        list(req_rows[: len(kv_lens)]), dtype=torch.int32, device=device
    )
    req_to_token = torch.zeros(
        (len(req_rows) + 4, max_ctx), dtype=torch.int32, device=device
    )

    if tail_slots:
        # Push the referenced slots to the very end of a huge pool so the
        # flat slot*stride index passes 2**31 (int64 arithmetic proof).
        per_request, off = [], 0
        base = num_slots - total_k
        for kv_len in kv_lens:
            per_request.append(
                torch.arange(
                    base + off,
                    base + off + kv_len,
                    dtype=torch.int32,
                    device=device,
                )
            )
            off += kv_len
        gather_index = (
            per_request[0]
            if len(per_request) == 1
            else torch.cat(per_request).to(torch.long)
        )
    else:
        per_request, gather_index = _slot_plan(kv_lens, num_slots, page_size, device)

    for row, (idx, slots) in enumerate(zip(req_idx.tolist(), per_request)):
        req_to_token[idx, : slots.numel()] = slots.to(torch.int32)

    if dtype != torch.bfloat16:
        # Quantise only the referenced rows through the uint8 view (fp8
        # index_put is not implemented): unreferenced pool slots are never
        # loaded by either path, so zeros there are fine.
        ref = gather_index
        raw_k = torch.randn(
            ref.numel(), 1, HEAD_DIM, dtype=torch.float32, device=device
        )
        raw_v = torch.randn(
            ref.numel(), 1, HEAD_DIM, dtype=torch.float32, device=device
        )
        pool_k.view(torch.uint8)[ref] = (raw_k / k_scale).to(dtype).view(torch.uint8)
        pool_v.view(torch.uint8)[ref] = (raw_v / v_scale).to(dtype).view(torch.uint8)

    k_pack = pool_k.index_select(0, gather_index)
    v_pack = pool_v.index_select(0, gather_index)

    indices = torch.full((total_q, TOPK), -1, dtype=torch.int32, device=device)
    row = 0
    for q_len, prefix, kv_len in zip(q_lens, prefix_lens, kv_lens):
        for relative in range(q_len):
            visible = prefix + relative + 1
            count = min(visible, TOPK - 4)
            chosen = torch.randperm(visible, device=device)[:count]
            indices[row, :count] = chosen.to(torch.int32)
            # Invalid markers like the packed-path tests use.
            indices[row, count : count + 4] = torch.tensor(
                [-1, -7, kv_len, kv_len + 19], dtype=torch.int32, device=device
            )
            row += 1

    cu_q = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()], dtype=torch.int32, device=device
    )
    cu_k = torch.tensor(
        [0, *torch.tensor(kv_lens).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32, device=device)

    return {
        "q": q,
        "pool_k": pool_k,
        "pool_v": pool_v,
        "k_pack": k_pack,
        "v_pack": v_pack,
        "req_to_token": req_to_token,
        "req_idx": req_idx,
        "indices": indices,
        "cu_q": cu_q,
        "cu_k": cu_k,
        "kv_lens": kv_lens_tensor,
        "max_q": max(q_lens),
        "gather_index": gather_index,
    }


def _run_pair(case, dtype):
    scales = {} if dtype == torch.bfloat16 else {"k_scale": 0.25, "v_scale": 0.5}
    old = sparse_gqa_fwd_interface_triton_ck(
        case["q"],
        case["k_pack"],
        case["v_pack"],
        case["indices"],
        case["cu_q"],
        case["cu_k"],
        case["kv_lens"],
        SCALE,
        **scales,
        max_q=case["max_q"],
    )
    new = sparse_gqa_fwd_interface_triton_paged(
        case["q"],
        case["pool_k"],
        case["pool_v"],
        case["req_to_token"],
        case["req_idx"],
        case["indices"],
        case["cu_q"],
        case["cu_k"],
        case["kv_lens"],
        SCALE,
        **scales,
        max_q=case["max_q"],
    )
    assert torch.isfinite(new.float()).all()
    assert torch.equal(old, new)


# (id, q_lens, prefix_lens, page_size) -- scattered unless page_size given;
# zero-prefix requests appear as prefix 0 entries and in their own case.
GEOMETRIES = [
    ("single", [9], [3], None),
    ("ragged", [7, 11, 5], [4, 0, 7], None),
    ("zero_prefix", [6, 4], [0, 9], None),
    ("page64", [5, 9, 120], [13, 0, 64], 64),
    ("page128", [5, 9, 120], [13, 0, 64], 128),
]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize(
    "name,q_lens,prefix_lens,page_size",
    [pytest.param(*g, id=g[0]) for g in GEOMETRIES],
)
def test_paged_matches_packed_oracle(name, q_lens, prefix_lens, page_size, dtype):
    case = _make_case(q_lens, prefix_lens, dtype=dtype, page_size=page_size, seed=7)
    _run_pair(case, dtype)


def test_paged_large_pool_int64_flat_index():
    """Slots live at the far end of a >2**31-element fp8 pool.

    Proves the kernel's flat offset arithmetic is int64: an int32 slot*stride
    product would wrap past 2**31 and read (or mask) garbage rows, breaking
    torch.equal against the packed oracle.
    """
    num_slots = LARGE_POOL_SLOTS
    q_lens, prefix_lens = [4, 8], [12, 0]
    case = _make_case(
        q_lens,
        prefix_lens,
        dtype=torch.float8_e4m3fn,
        num_slots=num_slots,
        tail_slots=True,
        seed=11,
    )
    max_slot = int(case["gather_index"].max())
    max_flat_index = max_slot * HEAD_DIM + HEAD_DIM - 1
    print(
        f"large-pool proof: slots up to {max_slot}, "
        f"max flat element index {max_flat_index} "
        f"vs 2**31-1 = {2**31 - 1} (exceeds: {max_flat_index > 2**31 - 1})"
    )
    assert max_flat_index > 2**31 - 1
    assert max_slot * HEAD_DIM * 1 > 2**31 - 1
    _run_pair(case, torch.float8_e4m3fn)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
