"""CPU coverage for the QSA chunked-prefill KV source selector."""

import unittest
from unittest import mock

from sglang.srt.environ import envs
from sglang.srt.layers.attention.qsa.sparse_attn import (
    _resolve_chunk_prefill_kv_source,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


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


if __name__ == "__main__":
    unittest.main()
