"""CPU coverage for the QSA dense chunk-prefill KV memory guards.

Two guards protect ``QwenSparseAttnBackend._forward_chunk_prefill``, the path
that materialises ``sum(full contexts) x kv_heads x head_dim`` of K and V per
QSA layer per forward -- an amount bounded by no scheduler accounting, whose
failure mode used to be a fatal prefill OOM (next9, 2026-10-18 23:26: a 2 GiB
``pack_qsa_prefill_kv`` allocation at ``total_k`` ~= 4.19M took down all four
TP ranks):

* ``_dense_pack_mem_route`` refuses an oversized pack *before* allocating it
  and tiers to the paged kernel, then to a pool-dtype gather;
* ``_chunk_prefill_oom_fallback`` catches the OOM of an estimate that turned
  out wrong and serves the forward from the paged kernel, latching the
  instance onto paged.

The kernels are patched at the module boundary and the backend is built with
``__new__`` plus only the attributes the guard methods read, so none of this
needs a device: it must pass on a host with no visible GPU.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention import qwen_sparse_attn_backend as _mod
from sglang.srt.layers.attention.qsa.sparse_attn import (
    _resolve_chunk_prefill_kv_source,
    qsa_pack_prefill_kv_bytes,
)
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QSA_PACK_HEADROOM_CACHE_KEY,
    QwenSparseAttnBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

_BACKEND = "sglang.srt.layers.attention.qwen_sparse_attn_backend"
_GIB = 2**30

# Captured before setUp patches the module attributes: the real headroom reader,
# so a test can drive the capture path end to end through the patch.
_REAL_DENSE_HEADROOM = _mod._qsa_dense_pack_headroom_bytes

# Qwen3.8-Flash-Next at TP4: one KV head per rank, head_dim 256, bf16 queries,
# so a dense K+V pair costs 1 KiB per summed context token.
_HEADS = 1
_DIM = 256
# ``total_k`` of the batch that OOM'd in prod: 2 GiB per tensor, 4 GiB per pair.
_INCIDENT_TOTAL_K = 4_194_304

_PATCH_TARGETS = {
    "pack": "pack_qsa_prefill_kv",
    "ck": "sparse_gqa_fwd_interface_triton_ck",
    "paged": "sparse_gqa_fwd_interface_triton_paged",
    "headroom": "_qsa_dense_pack_headroom_bytes",
    "static_budget": "_qsa_pack_static_budget_bytes",
}


def _pack_bytes(total_k: int, dtype: torch.dtype) -> int:
    return qsa_pack_prefill_kv_bytes(
        total_k=total_k, heads=_HEADS, dim=_DIM, dtype=dtype
    )


def _log_message(call) -> str:
    """All positional arguments of a logger call, joined (style-agnostic)."""
    return " ".join(str(arg) for arg in call.args)


class _RaisingBuffer:
    """Pool-buffer stand-in whose dense gather raises.

    Duck-typed on purpose: the guards only read ``shape``/``dtype`` and call
    ``index_select``, so this shows the code under test exactly what a real FP8
    pool buffer shows, while forcing the failure in the gather branch.
    """

    def __init__(self, dtype, exc, rows=64, heads=_HEADS, dim=_DIM):
        self.dtype = dtype
        self.shape = torch.Size((rows, heads, dim))
        self._exc = exc
        self.index_select_calls = 0

    def index_select(self, dim, index):
        self.index_select_calls += 1
        raise self._exc


class TestQsaPackBytesFormula(CustomTestCase):
    """The single source of truth for both guards' byte arithmetic."""

    def test_pair_bytes_matches_two_tensors(self):
        for dtype, itemsize in [
            (torch.bfloat16, 2),
            (torch.float16, 2),
            (torch.float8_e4m3fn, 1),
            (torch.float8_e5m2, 1),
            (torch.float32, 4),
        ]:
            with self.subTest(dtype=dtype):
                self.assertEqual(
                    qsa_pack_prefill_kv_bytes(total_k=3, heads=2, dim=128, dtype=dtype),
                    3 * 2 * 128 * itemsize * 2,
                )
                # Prod layout: bytes scale linearly with the summed contexts.
                self.assertEqual(_pack_bytes(3, dtype), 3 * 256 * itemsize * 2)

    def test_zero_rows_cost_nothing(self):
        self.assertEqual(_pack_bytes(0, torch.bfloat16), 0)

    def test_incident_pack_is_the_two_gib_allocation(self):
        # The crash asked for 2.00 GiB per tensor; a guard that priced only one
        # tensor would keep approving what killed the process.
        self.assertEqual(_pack_bytes(_INCIDENT_TOTAL_K, torch.bfloat16) // 2, 2 * _GIB)
        self.assertEqual(_pack_bytes(_INCIDENT_TOTAL_K, torch.bfloat16), 4 * _GIB)


class TestQsaStaticPackBudget(CustomTestCase):
    """Configuration-only budget term (capture-safe: no memory query)."""

    _TOTAL_MEM = 95 * _GIB

    @staticmethod
    def _device_module(total_memory=None):
        module = mock.MagicMock()
        if total_memory is None:
            module.get_device_properties.side_effect = RuntimeError("no device")
        else:
            module.get_device_properties.return_value = SimpleNamespace(
                total_memory=total_memory
            )
        return module

    def test_prod_config_budget_refuses_the_incident_pack(self):
        with (
            mock.patch.object(
                _mod, "get_device_module", return_value=self._device_module(95 * _GIB)
            ),
            mock.patch.object(
                _mod,
                "get_schedule",
                return_value=SimpleNamespace(mem_fraction_static=0.90),
            ),
        ):
            budget = _mod._qsa_pack_static_budget_bytes(device_index=0)
        # 0.1 * 95 GiB of dynamic headroom, 0.2 of it claimable -> 1.9 GiB:
        # under the 0.3 * total cap and under the 4 GiB pair that killed prod.
        self.assertEqual(budget, int(int(self._TOTAL_MEM * 0.1) * 0.2))
        self.assertLess(budget, _pack_bytes(_INCIDENT_TOTAL_K, torch.bfloat16))

    def test_unknown_mem_fraction_falls_back_to_total_fraction(self):
        with (
            mock.patch.object(
                _mod, "get_device_module", return_value=self._device_module(95 * _GIB)
            ),
            mock.patch.object(
                _mod,
                "get_schedule",
                return_value=SimpleNamespace(mem_fraction_static=None),
            ),
        ):
            self.assertEqual(
                _mod._qsa_pack_static_budget_bytes(device_index=0),
                int(self._TOTAL_MEM * 0.3),
            )

    def test_unreadable_schedule_degrades_like_none(self):
        with (
            mock.patch.object(
                _mod, "get_device_module", return_value=self._device_module(95 * _GIB)
            ),
            mock.patch.object(_mod, "get_schedule", side_effect=RuntimeError("no ctx")),
        ):
            self.assertEqual(
                _mod._qsa_pack_static_budget_bytes(device_index=0),
                int(self._TOTAL_MEM * 0.3),
            )

    def test_unreadable_device_properties_give_no_budget(self):
        # An unknown budget must never refuse the pack.
        with mock.patch.object(
            _mod, "get_device_module", return_value=self._device_module()
        ):
            self.assertIsNone(_mod._qsa_pack_static_budget_bytes(device_index=None))

    def test_fractions_are_configurable(self):
        with (
            mock.patch.object(
                _mod, "get_device_module", return_value=self._device_module(8 * _GIB)
            ),
            mock.patch.object(
                _mod,
                "get_schedule",
                return_value=SimpleNamespace(mem_fraction_static=0.5),
            ),
            envs.SGLANG_QSA_CHUNK_PREFILL_BUDGET_TOTAL_FRACTION.override(0.9),
            envs.SGLANG_QSA_CHUNK_PREFILL_BUDGET_FREE_FRACTION.override(0.5),
        ):
            # min(0.5 * (1 - 0.5) * 8 GiB, 0.9 * 8 GiB): the free term binds.
            self.assertEqual(
                _mod._qsa_pack_static_budget_bytes(device_index=0), 2 * _GIB
            )

    def test_headroom_needs_both_readings(self):
        with (
            mock.patch.object(_mod, "_qsa_stream_is_capturing", return_value=True),
            mock.patch.object(torch.cuda, "mem_get_info") as mem_get_info,
            mock.patch.object(torch.cuda, "memory_reserved", return_value=8 * _GIB),
            mock.patch.object(torch.cuda, "memory_allocated", return_value=2 * _GIB),
        ):
            free, cached = _mod._qsa_device_headroom_bytes(None)
            headroom = _mod._qsa_dense_pack_headroom_bytes(None)
        mem_get_info.assert_not_called()
        self.assertIsNone(free)
        self.assertEqual(cached, 6 * _GIB)
        # A partial reading must not be mistaken for usable headroom.
        self.assertIsNone(headroom)

    def test_live_headroom_counts_free_plus_cached(self):
        with (
            mock.patch.object(_mod, "_qsa_stream_is_capturing", return_value=False),
            mock.patch.object(
                torch.cuda, "mem_get_info", return_value=(4 * _GIB, 60 * _GIB)
            ),
            mock.patch.object(torch.cuda, "memory_reserved", return_value=8 * _GIB),
            mock.patch.object(torch.cuda, "memory_allocated", return_value=2 * _GIB),
        ):
            self.assertEqual(_REAL_DENSE_HEADROOM(None), 4 * _GIB + 6 * _GIB)


class _BackendHarness(CustomTestCase):
    """Backend with the production guards over stubbed per-forward state.

    ``_forward_chunk_prefill``, ``_dense_pack_mem_route``,
    ``_qsa_pack_budget_bytes``, ``_paged_chunk_prefill_output``,
    ``_chunk_prefill_oom_fallback``, ``_chunk_prefill_kv_path`` and
    ``_pad_extend_output`` stay real; only the kernels, the hoist builder and
    the layout guard are substituted.  The pack mock returns four rows
    regardless of ``total_k`` so a 4 GiB budget case costs kilobytes of RAM.
    """

    def setUp(self):
        _resolve_chunk_prefill_kv_source.cache_clear()
        self.addCleanup(_resolve_chunk_prefill_kv_source.cache_clear)
        self.pack = mock.MagicMock(name="pack", side_effect=self._packed_result)
        self.ck = mock.MagicMock(name="ck", side_effect=self._ck_result)
        self.paged = mock.MagicMock(name="paged", side_effect=self._paged_result)
        self.headroom = mock.MagicMock(name="headroom", return_value=None)
        self.static_budget = mock.MagicMock(name="static_budget", return_value=None)
        for name in _PATCH_TARGETS:
            patcher = mock.patch(
                f"{_BACKEND}.{_PATCH_TARGETS[name]}", getattr(self, name)
            )
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def _packed_result(**kwargs):
        rows = 4
        dtype = kwargs["output_dtype"]
        return (
            torch.zeros(rows, _HEADS, _DIM, dtype=dtype),
            torch.zeros(rows, _HEADS, _DIM, dtype=dtype),
        )

    @staticmethod
    def _ck_result(*args, **kwargs):
        # Distinct fill so the caller can tell the dense result from the paged one.
        return torch.full((4, _HEADS, _DIM), 3.0, dtype=torch.float32)

    @staticmethod
    def _paged_result(*args, **kwargs):
        return torch.full((4, _HEADS, _DIM), 7.0, dtype=torch.float32)

    def backend(self, guard=(True, None), pool_dtype=torch.bfloat16, rows=64):
        backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
        # What ``__init__`` seeds for these guards -- kept in sync on purpose,
        # so a rename there without one here fails loudly.
        backend._paged_kv_guard_warned = False
        backend._chunk_prefill_kv_forced_paged = False
        backend._qsa_pack_attempts = 0
        backend._qsa_pack_budget_refusals = 0
        backend._qsa_pack_oom_fallbacks = 0
        backend._qsa_pack_budget_warned = {}
        backend._paged_kv_source_supported = mock.MagicMock(return_value=guard)
        backend.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.zeros(8, 512, dtype=torch.int64)
        )
        self.k_buffer = torch.zeros(rows, _HEADS, _DIM, dtype=pool_dtype)
        self.v_buffer = torch.zeros(rows, _HEADS, _DIM, dtype=pool_dtype)
        backend.token_to_kv_pool = SimpleNamespace(
            get_key_buffer=lambda layer_id: self.k_buffer,
            get_value_buffer=lambda layer_id: self.v_buffer,
        )
        backend._kv_descales = lambda layer, dtype: (1.0, 1.0)

        def build(shared, forward_batch, device, *, need_gather):
            total_k = shared["sequence_lens"][0]
            shared.setdefault(
                "sequence_lens_tensor", torch.tensor([total_k], dtype=torch.int32)
            )
            shared.setdefault(
                "cu_seqlens_k", torch.tensor([0, total_k], dtype=torch.int32)
            )
            shared.setdefault("max_q", 4)
            if need_gather:
                shared["gather_index"] = torch.arange(4)

        backend._build_chunk_prefill_shared = mock.MagicMock(side_effect=build)
        return backend

    def forward_args(self, total_k, q_dtype=torch.bfloat16):
        q = torch.zeros(4, 8, _DIM, dtype=q_dtype)
        layer = SimpleNamespace(layer_id=0, scaling=0.0625)
        forward_batch = SimpleNamespace(
            req_pool_indices=torch.tensor([0]),
            seq_lens_cpu=[total_k],
            extend_seq_lens_cpu=[4],
        )
        topk_indices = torch.zeros(4, 8, dtype=torch.int32)
        shared = {
            "extend_lens": [4],
            "sequence_lens": [total_k],
            "prefix_lens": [total_k - 4],
            "cu_seqlens_q": torch.tensor([0, 4], dtype=torch.int32),
            "sequence_lens_tensor": torch.tensor([total_k], dtype=torch.int32),
            "cu_seqlens_k": torch.tensor([0, total_k], dtype=torch.int32),
            "max_q": 4,
        }
        return q, layer, forward_batch, topk_indices, shared

    def run_chunk_prefill(self, backend, total_k, q_dtype=torch.bfloat16):
        q, layer, forward_batch, topk_indices, shared = self.forward_args(
            total_k, q_dtype
        )
        out = backend._forward_chunk_prefill(
            q, layer, forward_batch, topk_indices, shared, num_output_rows=4
        )
        return out, shared


class TestQsaDensePackOomFallback(_BackendHarness):
    def test_pack_oom_serves_the_forward_from_the_paged_kernel(self):
        backend = self.backend(guard=(True, None))
        self.pack.side_effect = torch.cuda.OutOfMemoryError(
            "Tried to allocate 2.00 GiB"
        )
        with mock.patch(f"{_BACKEND}.logger") as fake_logger:
            out, shared = self.run_chunk_prefill(backend, _INCIDENT_TOTAL_K)
        # The paged result, padded -- not the dense one.
        self.assertEqual(out.shape, (4, _HEADS * _DIM))
        self.assertTrue(torch.all(out == 7.0))
        self.paged.assert_called_once()
        self.ck.assert_not_called()
        self.assertIs(self.paged.call_args.args[6], shared["cu_seqlens_q"])
        self.assertIs(self.paged.call_args.args[7], shared["cu_seqlens_k"])
        self.assertEqual(self.paged.call_args.kwargs["max_q"], 4)
        self.assertEqual(backend._qsa_pack_oom_fallbacks, 1)
        self.assertEqual(backend._qsa_pack_attempts, 1)
        self.assertTrue(backend._chunk_prefill_kv_forced_paged)
        # One actionable warning naming the layer, the bytes and the fallback.
        self.assertEqual(fake_logger.warning.call_count, 1)
        message = _log_message(fake_logger.warning.call_args)
        self.assertIn("out of memory", message)
        self.assertIn("4.00 GiB", message)
        self.assertIn(str(_INCIDENT_TOTAL_K), message)
        self.assertIn("layer 0", message)
        self.assertIn("paged chunk-prefill", message)
        self.assertIn("--mem-fraction-static", message)
        self.assertIn("--prefill-max-requests", message)

    def test_latched_instance_skips_the_dense_attempt(self):
        backend = self.backend(guard=(True, None))
        self.pack.side_effect = torch.cuda.OutOfMemoryError(
            "Tried to allocate 2.00 GiB"
        )
        with mock.patch(f"{_BACKEND}.logger") as fake_logger:
            self.run_chunk_prefill(backend, _INCIDENT_TOTAL_K)
            # Every remaining QSA layer of this forward, and every later
            # forward, must route paged without allocating again.
            for _ in range(2):
                out, _ = self.run_chunk_prefill(backend, _INCIDENT_TOTAL_K)
        self.assertEqual(self.pack.call_count, 1)
        self.assertEqual(self.paged.call_count, 3)
        self.assertEqual(fake_logger.warning.call_count, 1)
        self.assertEqual(backend._qsa_pack_oom_fallbacks, 1)
        self.assertEqual(backend._qsa_pack_attempts, 1)
        self.assertTrue(torch.all(out == 7.0))
        self.assertEqual(backend._chunk_prefill_kv_path(), "paged")

    def test_paged_source_builds_hoist_without_a_gather(self):
        backend = self.backend(guard=(True, None))
        q, layer, forward_batch, topk_indices, shared = self.forward_args(1024)
        for key in ("sequence_lens_tensor", "cu_seqlens_k", "max_q"):
            shared.pop(key)
        with envs.SGLANG_QSA_CHUNK_PREFILL_KV_SOURCE.override("paged"):
            out = backend._forward_chunk_prefill(
                q, layer, forward_batch, topk_indices, shared, num_output_rows=4
            )
        self.assertTrue(torch.all(out == 7.0))
        self.pack.assert_not_called()
        self.assertEqual(
            [c.kwargs for c in backend._build_chunk_prefill_shared.call_args_list],
            [{"need_gather": False}],
        )

    def test_guard_refusal_reraises_after_an_actionable_error(self):
        backend = self.backend(guard=(False, "page-major KV layout is enabled"))
        self.pack.side_effect = torch.cuda.OutOfMemoryError(
            "Tried to allocate 2.00 GiB"
        )
        with (
            mock.patch(f"{_BACKEND}.logger") as fake_logger,
            self.assertRaises(torch.cuda.OutOfMemoryError),
        ):
            self.run_chunk_prefill(backend, _INCIDENT_TOTAL_K)
        self.assertEqual(fake_logger.error.call_count, 1)
        self.assertEqual(fake_logger.warning.call_count, 0)
        message = _log_message(fake_logger.error.call_args)
        self.assertIn("page-major KV layout is enabled", message)
        self.assertIn("4.00 GiB", message)
        for lever in (
            "--mem-fraction-static",
            "--context-length",
            "--max-running-requests",
            "--prefill-max-requests",
        ):
            self.assertIn(lever, message)
        # The guard said paged cannot serve, so it must not be suggested.
        self.assertNotIn("SGLANG_QSA_CHUNK_PREFILL_KV_SOURCE", message)
        self.paged.assert_not_called()
        self.assertFalse(backend._chunk_prefill_kv_forced_paged)
        self.assertEqual(backend._qsa_pack_oom_fallbacks, 0)

    def test_kill_switch_off_reraises_without_fallback(self):
        backend = self.backend(guard=(True, None))
        self.pack.side_effect = torch.cuda.OutOfMemoryError(
            "Tried to allocate 2.00 GiB"
        )
        with (
            envs.SGLANG_ENABLE_QSA_CHUNK_PREFILL_OOM_FALLBACK.override(False),
            mock.patch(f"{_BACKEND}.logger") as fake_logger,
            self.assertRaises(torch.cuda.OutOfMemoryError),
        ):
            self.run_chunk_prefill(backend, _INCIDENT_TOTAL_K)
        self.assertEqual(fake_logger.error.call_count, 1)
        message = _log_message(fake_logger.error.call_args)
        self.assertIn("SGLANG_ENABLE_QSA_CHUNK_PREFILL_OOM_FALLBACK is off", message)
        # Paged is supported here, so the log has to offer it.
        self.assertIn("SGLANG_QSA_CHUNK_PREFILL_KV_SOURCE=paged", message)
        self.paged.assert_not_called()
        self.assertFalse(backend._chunk_prefill_kv_forced_paged)

    def test_scaled_pool_gather_oom_also_falls_back(self):
        # The index_select branch is inside the same try, so a scaled FP8 pool
        # (which never uses the pack) is covered too.
        backend = self.backend(guard=(True, None))
        backend._kv_descales = lambda layer, dtype: (2.0, 4.0)
        self.k_buffer = _RaisingBuffer(
            torch.float8_e4m3fn, torch.cuda.OutOfMemoryError("boom")
        )
        self.v_buffer = _RaisingBuffer(
            torch.float8_e4m3fn, torch.cuda.OutOfMemoryError("boom")
        )
        with mock.patch(f"{_BACKEND}.logger"):
            out, _ = self.run_chunk_prefill(backend, 1024)
        self.assertTrue(torch.all(out == 7.0))
        self.assertEqual(self.k_buffer.index_select_calls, 1)
        self.pack.assert_not_called()
        self.assertTrue(backend._chunk_prefill_kv_forced_paged)


class TestQsaDensePackBudgetRoute(_BackendHarness):
    """Tiered routing when the dense K/V exceeds the budget."""

    def test_over_budget_routes_to_paged(self):
        backend = self.backend(guard=(True, None))
        self.static_budget.return_value = 1 * _GIB
        out, shared = self.run_chunk_prefill(backend, _INCIDENT_TOTAL_K)
        self.assertTrue(torch.all(out == 7.0))
        self.pack.assert_not_called()
        self.ck.assert_not_called()
        self.paged.assert_called_once()
        self.assertEqual(backend._qsa_pack_budget_refusals, 1)
        self.assertEqual(backend._qsa_pack_attempts, 0)
        # No device on this host, so the dynamic term is unknown and cached as such.
        self.assertIsNone(shared[QSA_PACK_HEADROOM_CACHE_KEY])
        self.assertEqual(self.headroom.call_count, 1)

    def test_warns_once_per_tier_with_the_numbers(self):
        backend = self.backend(guard=(True, None))
        self.static_budget.return_value = 1 * _GIB
        with mock.patch(f"{_BACKEND}.logger") as fake_logger:
            for _ in range(3):
                self.run_chunk_prefill(backend, _INCIDENT_TOTAL_K)
        self.assertEqual(fake_logger.warning.call_count, 1)
        message = _log_message(fake_logger.warning.call_args)
        self.assertIn("4.00 GiB", message)
        self.assertIn("1.00 GiB", message)
        self.assertIn("SGLANG_QSA_CHUNK_PREFILL_KV_SOURCE=paged", message)
        self.assertEqual(backend._qsa_pack_budget_refusals, 3)

    def test_over_budget_fp8_pool_routes_to_pool_dtype_gather(self):
        # The paged guard refuses (page-major / dequant-read pools), but the FP8
        # pool halves the bytes and fits the budget: gather in pool dtype.
        backend = self.backend(guard=(False, "page-major KV layout is enabled"))
        self.k_buffer = torch.zeros(64, _HEADS, _DIM, dtype=torch.float8_e4m3fn)
        self.v_buffer = torch.zeros(64, _HEADS, _DIM, dtype=torch.float8_e4m3fn)
        self.static_budget.return_value = 3 * _GIB  # between 2 GiB (fp8) and 4 GiB
        out, shared = self.run_chunk_prefill(backend, _INCIDENT_TOTAL_K)
        self.assertTrue(torch.all(out == 3.0))
        self.pack.assert_not_called()
        self.paged.assert_not_called()
        self.ck.assert_called_once()
        gathered_k, gathered_v = self.ck.call_args.args[1:3]
        self.assertEqual(gathered_k.dtype, torch.float8_e4m3fn)
        self.assertEqual(gathered_v.dtype, torch.float8_e4m3fn)
        # The descales still ride along: the kernel dequantises what it reads.
        self.assertEqual(
            {k: v for k, v in self.ck.call_args.kwargs.items() if k != "max_q"},
            {"k_scale": 1.0, "v_scale": 1.0},
        )
        self.assertEqual(
            backend._build_chunk_prefill_shared.call_args.kwargs,
            {"need_gather": True},
        )
        self.assertEqual(backend._qsa_pack_attempts, 1)
        self.assertEqual(backend._qsa_pack_budget_refusals, 1)

    def test_pool_dtype_gather_used_only_when_it_fits(self):
        backend = self.backend(guard=(False, "page-major KV layout is enabled"))
        self.k_buffer = torch.zeros(64, _HEADS, _DIM, dtype=torch.float8_e4m3fn)
        self.v_buffer = torch.zeros(64, _HEADS, _DIM, dtype=torch.float8_e4m3fn)
        self.static_budget.return_value = 1 * _GIB  # below even the FP8 pair
        with mock.patch(f"{_BACKEND}.logger") as fake_logger:
            out, _ = self.run_chunk_prefill(backend, _INCIDENT_TOTAL_K)
        # Tier (c): attempt anyway -- refusing outright would be worse.
        self.assertEqual(self.pack.call_count, 1)
        self.assertTrue(torch.all(out == 3.0))
        self.assertEqual(backend._qsa_pack_budget_refusals, 0)
        self.assertEqual(backend._qsa_pack_attempts, 1)
        message = _log_message(fake_logger.warning.call_args)
        self.assertIn("attempting the pack", message)
        self.assertIn("page-major KV layout is enabled", message)

    def test_non_fp8_pool_over_budget_attempts_pack(self):
        backend = self.backend(guard=(False, "page-major KV layout is enabled"))
        self.static_budget.return_value = 1 * _GIB
        with mock.patch(f"{_BACKEND}.logger") as fake_logger:
            out, _ = self.run_chunk_prefill(backend, _INCIDENT_TOTAL_K)
        self.assertEqual(self.pack.call_count, 1)
        self.assertEqual(backend._qsa_pack_attempts, 1)
        self.assertIn(
            "neither fallback applies", _log_message(fake_logger.warning.call_args)
        )

    def test_under_budget_skips_every_device_query(self):
        backend = self.backend(guard=(True, None))
        # A 4 GiB pack under a 4 GiB floor is ordinary work: the guard must not
        # run at all, and the pack must be called exactly as before it existed.
        with envs.SGLANG_QSA_CHUNK_PREFILL_PACK_BUDGET_MIN_BYTES.override(4 * _GIB):
            out, shared = self.run_chunk_prefill(backend, _INCIDENT_TOTAL_K)
        self.assertEqual(self.pack.call_count, 1)
        self.static_budget.assert_not_called()
        self.headroom.assert_not_called()
        self.assertEqual(backend._qsa_pack_attempts, 1)
        self.assertNotIn(QSA_PACK_HEADROOM_CACHE_KEY, shared)
        self.assertEqual(self.pack.call_args.kwargs["output_dtype"], torch.bfloat16)
        self.assertEqual(self.pack.call_args.kwargs["total_k"], _INCIDENT_TOTAL_K)
        self.assertEqual(self.pack.call_args.kwargs["max_k"], _INCIDENT_TOTAL_K)
        self.assertTrue(torch.all(out == 3.0))

    def test_budget_kill_switch_skips_gate(self):
        backend = self.backend(guard=(True, None))
        self.static_budget.return_value = 1 * _GIB
        with envs.SGLANG_ENABLE_QSA_CHUNK_PREFILL_PACK_BUDGET.override(False):
            out, _ = self.run_chunk_prefill(backend, _INCIDENT_TOTAL_K)
        self.assertEqual(self.pack.call_count, 1)
        self.static_budget.assert_not_called()
        self.headroom.assert_not_called()
        self.assertTrue(torch.all(out == 3.0))

    def test_headroom_is_queried_once_per_forward(self):
        backend = self.backend(guard=(True, None))
        self.static_budget.return_value = None
        self.headroom.return_value = 2 * _GIB
        q, _, _, _, shared = self.forward_args(_INCIDENT_TOTAL_K)
        for _ in range(3):
            backend._dense_pack_mem_route(
                q=q,
                k_buffer=self.k_buffer,
                total_k=_INCIDENT_TOTAL_K,
                use_pack=True,
                shared=shared,
            )
        self.assertEqual(self.headroom.call_count, 1)
        self.assertEqual(shared[QSA_PACK_HEADROOM_CACHE_KEY], 2 * _GIB)
        # 0.2 * 2 GiB of headroom is well under the 4 GiB pack -> all refused.
        self.assertEqual(backend._qsa_pack_budget_refusals, 3)

    def test_headroom_budget_refuses_without_static_term(self):
        backend = self.backend(guard=(True, None))
        self.static_budget.return_value = None
        self.headroom.return_value = 2 * _GIB  # budget = 0.2 * 2 GiB
        out, _ = self.run_chunk_prefill(backend, _INCIDENT_TOTAL_K)
        self.paged.assert_called_once()
        self.pack.assert_not_called()
        self.assertTrue(torch.all(out == 7.0))

    def test_capture_budget_uses_only_the_static_term(self):
        backend = self.backend(guard=(True, None))
        self.headroom.side_effect = _REAL_DENSE_HEADROOM
        self.static_budget.return_value = 1 * _GIB
        with (
            mock.patch.object(_mod, "_qsa_stream_is_capturing", return_value=True),
            mock.patch.object(torch.cuda, "mem_get_info") as mem_get_info,
        ):
            out, shared = self.run_chunk_prefill(backend, _INCIDENT_TOTAL_K)
        mem_get_info.assert_not_called()
        self.assertIsNone(shared[QSA_PACK_HEADROOM_CACHE_KEY])
        # 4 GiB needed against the 1 GiB static budget: routed, not attempted.
        self.paged.assert_called_once()
        self.pack.assert_not_called()
        self.assertTrue(torch.all(out == 7.0))

    def test_non_token_major_layout_skips_gate(self):
        backend = self.backend(guard=(True, None))
        self.static_budget.return_value = 1 * _GIB
        self.k_buffer = SimpleNamespace(
            dtype=torch.bfloat16, shape=torch.Size((64, _HEADS * _DIM))
        )
        out, _ = self.run_chunk_prefill(backend, _INCIDENT_TOTAL_K)
        self.assertEqual(self.pack.call_count, 1)
        self.static_budget.assert_not_called()
        self.assertTrue(torch.all(out == 3.0))


if __name__ == "__main__":
    unittest.main()
