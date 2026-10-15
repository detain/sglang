# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Validation for --enable-linear-lossless-verify.

The flag replaces the per-draft SSM snapshots that --enable-linear-replayssm
(and -spec) also own, so the combination must fail closed at arg resolution --
independently of CLI flag order -- and the remaining gates (spec required,
linear chain, Triton verify kernel, no unified memory) must reject up front
instead of at the first verify step.

The dummy model path short-circuits the resolution pipeline before the
speculative hooks, so the hook under test is driven directly on CLI-parsed
ServerArgs (the pattern other server_args unit tests use).

    python -m pytest test/registered/unit/server_args/test_linear_lossless_verify_validation.py -v
"""

import inspect
import unittest

from sglang.srt.arg_groups.attention_hook import validate_linear_lossless_verify
from sglang.srt.arg_groups.pipeline import run_resolution_pipeline
from sglang.srt.server_args import prepare_server_args
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestLinearLosslessVerifyValidation(CustomTestCase):
    def _args(self, *cli_tokens):
        base = ["--model-path", "dummy", "--enable-linear-lossless-verify"]
        return prepare_server_args(base + list(cli_tokens))

    def test_off_by_default_never_raises(self):
        args = prepare_server_args(["--model-path", "dummy"])
        self.assertFalse(args.enable_linear_lossless_verify)
        validate_linear_lossless_verify(args)  # no-op

    def test_clean_linear_chain_resolves(self):
        # DFLASH pins the linear chain (topk == 1, restated here because the
        # dummy path skips the spec hook) with the triton verify kernel.
        args = self._args(
            "--speculative-algorithm", "DFLASH", "--speculative-eagle-topk", "1"
        )
        validate_linear_lossless_verify(args)
        self.assertTrue(args.enable_linear_lossless_verify)

    def test_requires_speculative_algorithm(self):
        args = self._args()
        with self.assertRaisesRegex(
            ValueError, "only changes speculative target-verify.*speculative-algorithm"
        ):
            validate_linear_lossless_verify(args)

    def test_mutually_exclusive_replayssm_spec_both_orders(self):
        for extra in (
            (
                "--speculative-algorithm",
                "DFLASH",
                "--speculative-eagle-topk",
                "1",
                "--enable-linear-replayssm-spec",
            ),
            (
                "--enable-linear-replayssm-spec",
                "--speculative-algorithm",
                "DFLASH",
                "--speculative-eagle-topk",
                "1",
            ),
        ):
            with self.subTest(flags=extra):
                args = self._args(*extra)
                with self.assertRaisesRegex(
                    ValueError, "mutually exclusive with .*replayssm"
                ):
                    validate_linear_lossless_verify(args)

    def test_mutually_exclusive_replayssm_both_orders(self):
        for extra in (
            (
                "--speculative-algorithm",
                "DFLASH",
                "--speculative-eagle-topk",
                "1",
                "--enable-linear-replayssm",
            ),
            (
                "--enable-linear-replayssm",
                "--speculative-algorithm",
                "DFLASH",
                "--speculative-eagle-topk",
                "1",
            ),
        ):
            with self.subTest(flags=extra):
                args = self._args(*extra)
                with self.assertRaisesRegex(
                    ValueError, "mutually exclusive with .*replayssm"
                ):
                    validate_linear_lossless_verify(args)

    def test_rejects_tree_topk(self):
        args = self._args(
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-draft-model-path",
            "dummy",
            "--speculative-eagle-topk",
            "2",
        )
        with self.assertRaisesRegex(ValueError, "requires a linear draft chain"):
            validate_linear_lossless_verify(args)

    def test_rejects_flashinfer_verify_backend(self):
        args = self._args(
            "--speculative-algorithm",
            "DFLASH",
            "--speculative-eagle-topk",
            "1",
            "--linear-attn-verify-backend",
            "flashinfer",
        )
        with self.assertRaisesRegex(ValueError, "Triton GDN verify kernel"):
            validate_linear_lossless_verify(args)

    def test_rejects_flashinfer_decode_default_verify(self):
        # verify unset follows the decode backend; flashinfer decode would pick
        # the flashinfer verify kernel, which silently drops the tuple ring.
        args = self._args(
            "--speculative-algorithm",
            "DFLASH",
            "--speculative-eagle-topk",
            "1",
            "--linear-attn-decode-backend",
            "flashinfer",
        )
        with self.assertRaisesRegex(ValueError, "Triton GDN verify kernel"):
            validate_linear_lossless_verify(args)

    def test_rejects_unified_memory(self):
        args = self._args(
            "--speculative-algorithm",
            "DFLASH",
            "--speculative-eagle-topk",
            "1",
            "--enable-unified-memory",
        )
        with self.assertRaisesRegex(
            ValueError, "not supported with.*enable-unified-memory"
        ):
            validate_linear_lossless_verify(args)

    def test_ragged_verify_env_gate(self):
        from sglang.srt.environ import envs

        args = self._args(
            "--speculative-algorithm", "DFLASH", "--speculative-eagle-topk", "1"
        )
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("compact"):
            with self.assertRaisesRegex(ValueError, "SGLANG_RAGGED_VERIFY_MODE"):
                validate_linear_lossless_verify(args)

    def test_hook_wired_after_speculative_resolution(self):
        # The gate reads the FINAL speculative_eagle_topk (DFLASH/DSPARK pin it
        # to 1 inside handle_speculative_decoding), so it must run after it.
        src = inspect.getsource(run_resolution_pipeline)
        spec_at = src.index("run_hook(handle_speculative_decoding")
        lossless_at = src.index("run_hook(validate_linear_lossless_verify")
        self.assertGreater(lossless_at, spec_at)


if __name__ == "__main__":
    unittest.main()
