"""CPU coverage for the QSA chunked-prefill KV source selector and dispatch.

The resolver half lives in ``sparse_attn``; the dispatch half under test here
is ``QwenSparseAttnBackend._chunk_prefill_kv_path`` (env value + layout guard
+ one-shot WARNING fallback) and ``_paged_kv_source_supported`` (the guard's
real predicates).  Neither touches CUDA, so the backend instance is built
with ``__new__`` and only the attributes the methods read.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.environ import envs
from sglang.srt.layers.attention.qsa.sparse_attn import (
    _resolve_chunk_prefill_kv_source,
)
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QwenSparseAttnBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_BACKEND = "sglang.srt.layers.attention.qwen_sparse_attn_backend"


class TestQsaChunkPrefillKvSource(unittest.TestCase):
    def setUp(self):
        _resolve_chunk_prefill_kv_source.cache_clear()

    def tearDown(self):
        _resolve_chunk_prefill_kv_source.cache_clear()

    def test_auto_resolves_to_packed(self):
        # The paged path is unproven; the default must stay on the packed
        # gather path.
        with envs.SGLANG_QSA_CHUNK_PREFILL_KV_SOURCE.override("auto"):
            self.assertEqual(_resolve_chunk_prefill_kv_source(), "packed")

    def test_unset_env_defaults_to_packed(self):
        self.assertEqual(_resolve_chunk_prefill_kv_source(), "packed")

    def test_explicit_values(self):
        for value in ("packed", "paged"):
            with self.subTest(value=value):
                _resolve_chunk_prefill_kv_source.cache_clear()
                with envs.SGLANG_QSA_CHUNK_PREFILL_KV_SOURCE.override(value):
                    self.assertEqual(_resolve_chunk_prefill_kv_source(), value)

    def test_value_is_trimmed_and_lowercased(self):
        with envs.SGLANG_QSA_CHUNK_PREFILL_KV_SOURCE.override(" Paged "):
            self.assertEqual(_resolve_chunk_prefill_kv_source(), "paged")

    def test_invalid_value_raises_and_names_all_valid_values(self):
        with (
            envs.SGLANG_QSA_CHUNK_PREFILL_KV_SOURCE.override("dense"),
            self.assertRaises(ValueError) as ctx,
        ):
            _resolve_chunk_prefill_kv_source()
        message = str(ctx.exception)
        self.assertIn("SGLANG_QSA_CHUNK_PREFILL_KV_SOURCE", message)
        for value in ("auto", "packed", "paged"):
            self.assertIn(value, message)
        self.assertIn("dense", message)
        # A rejected value must not poison the cache for later resolutions.
        _resolve_chunk_prefill_kv_source.cache_clear()
        with envs.SGLANG_QSA_CHUNK_PREFILL_KV_SOURCE.override("paged"):
            self.assertEqual(_resolve_chunk_prefill_kv_source(), "paged")

    def test_resolution_is_cached_until_cache_clear(self):
        with envs.SGLANG_QSA_CHUNK_PREFILL_KV_SOURCE.override("paged"):
            self.assertEqual(_resolve_chunk_prefill_kv_source(), "paged")
            # The first resolution stays cached while the env var says
            # something else...
            with envs.SGLANG_QSA_CHUNK_PREFILL_KV_SOURCE.override("packed"):
                self.assertEqual(_resolve_chunk_prefill_kv_source(), "paged")
                # ...and the clear hook lets the second override take effect.
                _resolve_chunk_prefill_kv_source.cache_clear()
                self.assertEqual(_resolve_chunk_prefill_kv_source(), "packed")

    def test_resolved_choice_is_logged_once(self):
        with (
            envs.SGLANG_QSA_CHUNK_PREFILL_KV_SOURCE.override("paged"),
            mock.patch(
                "sglang.srt.layers.attention.qsa.sparse_attn.logger"
            ) as fake_logger,
        ):
            self.assertEqual(_resolve_chunk_prefill_kv_source(), "paged")
            self.assertEqual(_resolve_chunk_prefill_kv_source(), "paged")
            self.assertEqual(fake_logger.info.call_count, 1)
            logged = fake_logger.info.call_args_list[0].args
            self.assertIn("SGLANG_QSA_CHUNK_PREFILL_KV_SOURCE", logged[0])
            self.assertEqual(logged[1:], ("paged", "paged"))
            # Re-resolution after a clear logs the new choice once again.
            _resolve_chunk_prefill_kv_source.cache_clear()
            self.assertEqual(_resolve_chunk_prefill_kv_source(), "paged")
            self.assertEqual(fake_logger.info.call_count, 2)


class TestQsaChunkPrefillDispatch(unittest.TestCase):
    """Env value + layout guard -> which path the backend selects.

    Only the guard is mocked here (its own predicates get real-logic tests
    below); the dispatch, the one-shot WARNING and the fallback must be the
    production methods.  with-bodies stay one statement because a failing
    assert inside ``envs....override()`` leaks the env var.
    """

    def setUp(self):
        _resolve_chunk_prefill_kv_source.cache_clear()

    def tearDown(self):
        _resolve_chunk_prefill_kv_source.cache_clear()

    @staticmethod
    def _backend(guard_result=(True, None)):
        backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
        backend._paged_kv_guard_warned = False
        backend.token_to_kv_pool = None
        backend._paged_kv_source_supported = mock.MagicMock(return_value=guard_result)
        return backend

    def test_guard_pass_selects_paged(self):
        backend = self._backend((True, None))
        with envs.SGLANG_QSA_CHUNK_PREFILL_KV_SOURCE.override("paged"):
            path = backend._chunk_prefill_kv_path()
        self.assertEqual(path, "paged")
        self.assertEqual(backend._paged_kv_source_supported.call_count, 1)

    def test_guard_fail_selects_packed_and_warns_once(self):
        backend = self._backend((False, "page-major KV layout is enabled"))
        with (
            envs.SGLANG_QSA_CHUNK_PREFILL_KV_SOURCE.override("paged"),
            mock.patch(f"{_BACKEND}.logger") as fake_logger,
        ):
            # Per-layer calls must not turn the fallback into a warning storm.
            paths = [backend._chunk_prefill_kv_path() for _ in range(3)]
        self.assertEqual(paths, ["packed", "packed", "packed"])
        self.assertEqual(fake_logger.warning.call_count, 1)
        message = fake_logger.warning.call_args.args[0]
        self.assertIn("SGLANG_QSA_CHUNK_PREFILL_KV_SOURCE=paged", message)
        self.assertIn("packed", message)
        self.assertEqual(
            fake_logger.warning.call_args.args[1:],
            ("page-major KV layout is enabled",),
        )

    def test_packed_never_consults_the_guard(self):
        backend = self._backend()
        with envs.SGLANG_QSA_CHUNK_PREFILL_KV_SOURCE.override("packed"):
            path = backend._chunk_prefill_kv_path()
        self.assertEqual(path, "packed")
        backend._paged_kv_source_supported.assert_not_called()

    def test_auto_never_consults_the_guard(self):
        backend = self._backend()
        with envs.SGLANG_QSA_CHUNK_PREFILL_KV_SOURCE.override("auto"):
            path = backend._chunk_prefill_kv_path()
        self.assertEqual(path, "packed")
        backend._paged_kv_source_supported.assert_not_called()


class TestQsaPagedLayoutGuard(unittest.TestCase):
    """Real predicate logic of ``_paged_kv_source_supported``; the fakes
    mirror exactly what MHATokenToKVPool exposes (is_quantized_kv_cache is a
    property there, quant_method.needs_plain_kv_dequant_read only exists on
    packings that need dequantised plain reads)."""

    @staticmethod
    def _backend_with_pool(pool):
        backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
        backend.token_to_kv_pool = pool
        return backend

    def test_plain_token_major_pool_supported(self):
        pool = SimpleNamespace(is_quantized_kv_cache=False, quant_method=object())
        backend = self._backend_with_pool(pool)
        with mock.patch(
            f"{_BACKEND}.get_memory",
            return_value=SimpleNamespace(enable_page_major_kv_layout=False),
        ):
            supported, reason = backend._paged_kv_source_supported()
        self.assertTrue(supported)
        self.assertIsNone(reason)

    def test_page_major_flag_rejects_and_names_the_flag(self):
        pool = SimpleNamespace(is_quantized_kv_cache=False, quant_method=object())
        backend = self._backend_with_pool(pool)
        with mock.patch(
            f"{_BACKEND}.get_memory",
            return_value=SimpleNamespace(enable_page_major_kv_layout=True),
        ):
            supported, reason = backend._paged_kv_source_supported()
        self.assertFalse(supported)
        self.assertIn("--enable-page-major-kv-layout", reason)

    def test_dequant_read_pool_rejects_and_names_the_predicate(self):
        class FakeFP4LikePool:
            is_quantized_kv_cache = True

            class quant_method:
                @staticmethod
                def needs_plain_kv_dequant_read():
                    return True

        backend = self._backend_with_pool(FakeFP4LikePool())
        with mock.patch(
            f"{_BACKEND}.get_memory",
            return_value=SimpleNamespace(enable_page_major_kv_layout=False),
        ):
            supported, reason = backend._paged_kv_source_supported()
        self.assertFalse(supported)
        self.assertIn("needs_plain_kv_dequant_read", reason)
        self.assertIn("FakeFP4LikePool", reason)

    def test_quantized_pool_without_the_predicate_is_supported(self):
        # fp8-e4m3 stores via store_dtype; its kv-cache method has no
        # dequant-read hook, so the getattr default must keep it supported.
        pool = SimpleNamespace(is_quantized_kv_cache=True, quant_method=object())
        backend = self._backend_with_pool(pool)
        with mock.patch(
            f"{_BACKEND}.get_memory",
            return_value=SimpleNamespace(enable_page_major_kv_layout=False),
        ):
            supported, reason = backend._paged_kv_source_supported()
        self.assertTrue(supported)
        self.assertIsNone(reason)


if __name__ == "__main__":
    unittest.main()
