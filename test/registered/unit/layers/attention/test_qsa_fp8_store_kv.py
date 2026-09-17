"""CPU coverage for the QSA FP8 KV store contract (``QwenSparseAttnBackend._store_kv``).

The real MHATokenToKVPool cannot be built without a device allocator, so the
pool is emulated on CPU by replicating its fp8 write branch verbatim
(memory_pool.MHATokenToKVPool.set_kv_buffer: in-place ``div_`` of the caller's
tensor when a non-None scale is passed with a dtype mismatch, out-of-place
cast, uint8 byte scatter -- the real pool stores fp8 as uint8 because
index_put/index_copy lacks fp8 kernels). The mock skips the loc-range OOB
checks, the store_cache JIT path and the page/HND layouts, none of which
interact with the scale/clone logic under test.

Asserts the store contract:
  * pool bytes equal the reference ``(x / scale).to(float8_e4m3fn)`` exactly,
  * the caller's bf16 K/V tensors are never mutated (the reason #36644 cloned),
  * the unity-descale path hands the pool the original tensors (no clone),
  * the scaled path quantizes out-of-place into persistent grow-only scratch:
    no per-call allocation once a geometry has been seen.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QwenSparseAttnBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

FP8 = torch.float8_e4m3fn


class EmulatedFP8Pool:
    dtype = FP8
    # The real pool keeps fp8 bytes in uint8 storage ("Store as torch.uint8
    # because Tensor.index_put is not implemented for fp8").
    store_dtype = torch.uint8

    def __init__(self, num_slots: int, row_shape):
        self.k_store = torch.zeros((num_slots,) + row_shape, dtype=torch.uint8)
        self.v_store = torch.zeros((num_slots,) + row_shape, dtype=torch.uint8)
        self.calls = []

    def set_kv_buffer(
        self,
        layer,
        loc,
        cache_k,
        cache_v,
        k_scale=None,
        v_scale=None,
        layer_id_override=None,
        dcp_kv_mask=None,
    ):
        self.calls.append((cache_k, cache_v, k_scale, v_scale))
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)
        self.k_store.index_copy_(0, loc, cache_k.view(self.store_dtype))
        self.v_store.index_copy_(0, loc, cache_v.view(self.store_dtype))


def _make_backend(num_slots=16, row_shape=(2, 8)):
    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    backend._qsa_fp8_store_scratch = {}
    backend._qsa_fp8_store_scratch_retired = []
    backend.token_to_kv_pool = EmulatedFP8Pool(num_slots, row_shape)
    return backend


def _fp8_bytes(x: torch.Tensor) -> torch.Tensor:
    return x.to(FP8).view(torch.uint8)


class TestQsaFp8StoreKv(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.k = torch.randn(4, 2, 8, dtype=torch.bfloat16)
        self.v = torch.randn(4, 2, 8, dtype=torch.bfloat16)
        self.loc = torch.arange(4, dtype=torch.int64)

    def _store(self, backend, k_scale, v_scale, k=None, v=None, loc=None):
        layer = SimpleNamespace(
            layer_id=0,
            k_scale_float=k_scale,
            v_scale_float=v_scale,
        )
        backend._store_kv(
            layer,
            self.loc if loc is None else loc,
            self.k if k is None else k,
            self.v if v is None else v,
        )

    def test_unity_descale_stores_original_tensors_without_clone(self):
        backend = _make_backend()
        # Checkpoint ships no KV scales -> descale None/1.0 -> unity fast path.
        self._store(backend, None, None)
        (stored_k, stored_v, k_scale, v_scale) = backend.token_to_kv_pool.calls[0]
        self.assertIs(stored_k, self.k)
        self.assertIs(stored_v, self.v)
        self.assertIsNone(k_scale)
        self.assertIsNone(v_scale)
        self.assertTrue(
            torch.equal(backend.token_to_kv_pool.k_store[:4], _fp8_bytes(self.k))
        )
        self.assertTrue(
            torch.equal(backend.token_to_kv_pool.v_store[:4], _fp8_bytes(self.v))
        )

    def test_nonunity_scale_never_mutates_inputs_and_matches_reference(self):
        backend = _make_backend()
        k_before, v_before = self.k.clone(), self.v.clone()
        k_scale, v_scale = 0.25, 1.7
        self._store(backend, k_scale, v_scale)
        # The point of the old clone: inputs must be untouched by the store.
        self.assertTrue(torch.equal(self.k, k_before))
        self.assertTrue(torch.equal(self.v, v_before))
        (stored_k, stored_v, passed_k_scale, passed_v_scale) = (
            backend.token_to_kv_pool.calls[0]
        )
        # The pool is handed a pre-quantized buffer and no scale, so its
        # in-place div_ branch cannot fire even for these stores.
        self.assertIsNone(passed_k_scale)
        self.assertIsNone(passed_v_scale)
        self.assertEqual(stored_k.dtype, FP8)
        self.assertEqual(stored_v.dtype, FP8)
        self.assertTrue(
            torch.equal(
                backend.token_to_kv_pool.k_store[:4], _fp8_bytes(self.k / k_scale)
            )
        )
        self.assertTrue(
            torch.equal(
                backend.token_to_kv_pool.v_store[:4], _fp8_bytes(self.v / v_scale)
            )
        )

    def test_mixed_unity_k_nonunity_v(self):
        backend = _make_backend()
        k_before, v_before = self.k.clone(), self.v.clone()
        self._store(backend, 1.0, 0.5)
        self.assertTrue(torch.equal(self.k, k_before))
        self.assertTrue(torch.equal(self.v, v_before))
        self.assertTrue(
            torch.equal(backend.token_to_kv_pool.k_store[:4], _fp8_bytes(self.k))
        )
        self.assertTrue(
            torch.equal(backend.token_to_kv_pool.v_store[:4], _fp8_bytes(self.v / 0.5))
        )

    def test_scaled_store_reuses_persistent_scratch(self):
        backend = _make_backend()
        self._store(backend, 0.25, 0.5)
        first_k = backend.token_to_kv_pool.calls[0][0]
        first_v = backend.token_to_kv_pool.calls[0][1]
        k2 = torch.randn(4, 2, 8, dtype=torch.bfloat16)
        v2 = torch.randn(4, 2, 8, dtype=torch.bfloat16)
        self._store(backend, 0.25, 0.5, k=k2, v=v2, loc=torch.arange(4, 8))
        second_k = backend.token_to_kv_pool.calls[1][0]
        second_v = backend.token_to_kv_pool.calls[1][1]
        # Steady state: same underlying scratch storage (views are new
        # objects each call), zero per-call allocation.
        self.assertEqual(first_k.data_ptr(), second_k.data_ptr())
        self.assertEqual(first_v.data_ptr(), second_v.data_ptr())
        # No stale data: second store bytes are correct and first slots kept.
        self.assertTrue(
            torch.equal(backend.token_to_kv_pool.k_store[4:8], _fp8_bytes(k2 / 0.25))
        )
        self.assertTrue(
            torch.equal(backend.token_to_kv_pool.v_store[4:8], _fp8_bytes(v2 / 0.5))
        )
        self.assertTrue(
            torch.equal(backend.token_to_kv_pool.k_store[:4], _fp8_bytes(self.k / 0.25))
        )

    def test_scratch_growth_keeps_both_stores_correct(self):
        backend = _make_backend()
        self._store(backend, 0.25, 0.5)
        big_k = torch.randn(10, 2, 8, dtype=torch.bfloat16)
        big_v = torch.randn(10, 2, 8, dtype=torch.bfloat16)
        self._store(backend, 0.25, 0.5, k=big_k, v=big_v, loc=torch.arange(6, 16))
        self.assertTrue(
            torch.equal(
                backend.token_to_kv_pool.k_store[6:16], _fp8_bytes(big_k / 0.25)
            )
        )
        self.assertTrue(
            torch.equal(
                backend.token_to_kv_pool.v_store[6:16], _fp8_bytes(big_v / 0.5)
            )
        )
        self.assertTrue(
            torch.equal(backend.token_to_kv_pool.k_store[:4], _fp8_bytes(self.k / 0.25))
        )

    def test_asymmetric_kv_shapes_use_independent_scratch(self):
        backend = _make_backend(row_shape=(2, 8))
        k = torch.randn(4, 2, 8, dtype=torch.bfloat16)
        v = torch.randn(4, 3, 8, dtype=torch.bfloat16)
        # v_store was built for (2, 8) rows; make a matching second pool view
        # by storing V into a v-shaped emulation instead.
        pool = backend.token_to_kv_pool
        pool.v_store = torch.zeros((16, 3, 8), dtype=torch.uint8)
        layer = SimpleNamespace(layer_id=0, k_scale_float=0.3, v_scale_float=0.6)
        backend._store_kv(layer, self.loc, k, v)
        self.assertTrue(torch.equal(pool.k_store[:4], _fp8_bytes(k / 0.3)))
        self.assertTrue(torch.equal(pool.v_store[:4], _fp8_bytes(v / 0.6)))
        # Distinct fp8 output buffers for K and V (alive at the same time).
        (stored_k, stored_v, _, _) = pool.calls[0]
        self.assertIsNot(stored_k, stored_v)

    def test_strided_input_view_is_read_not_mutated(self):
        # QKV-split K/V can be non-contiguous strided views.
        packed = torch.randn(4, 2, 8 * 2, dtype=torch.bfloat16)
        k = packed[:, :, :8]
        backend = _make_backend()
        before = packed.clone()
        layer = SimpleNamespace(layer_id=0, k_scale_float=0.4, v_scale_float=0.4)
        backend._store_kv(layer, self.loc, k, self.v)
        self.assertTrue(torch.equal(packed, before))
        self.assertTrue(
            torch.equal(backend.token_to_kv_pool.k_store[:4], _fp8_bytes(k / 0.4))
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
