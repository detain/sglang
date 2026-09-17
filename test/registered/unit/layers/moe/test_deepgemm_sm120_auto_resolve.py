"""CPU guards for `auto` MoE runner resolution onto the SM120 DeepGEMM path."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe.moe_runner import deep_gemm_sm120
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
from sglang.test.test_utils import CustomTestCase

BLOCK_SIZE = [128, 128]


def _resolve(a2a: MoeA2ABackend, backend: MoeRunnerBackend, block_size):
    return Fp8MoEMethod.is_deepgemm_moe_runner_backend_enabled(
        moe_runner_backend=backend,
        moe_a2a_backend=a2a,
        weight_block_size=block_size,
    )


class TestSm120AutoDeepgemmResolve(CustomTestCase):
    def _check(self, sm, on_sm120_family, wheel_has_api, a2a, block_size):
        p_sm120 = patch.object(deep_gemm_sm120, "_is_sm120", on_sm120_family)
        p_enable = patch.object(deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", wheel_has_api)
        p_dev = patch("sglang.srt.utils.get_device_sm", return_value=sm)
        with p_sm120, p_enable, p_dev:
            return _resolve(a2a, MoeRunnerBackend.AUTO, block_size)

    def test_sm120_auto_none_blockwise_resolves_deepgemm(self):
        self.assertTrue(self._check(120, True, True, MoeA2ABackend.NONE, BLOCK_SIZE))

    def test_sm120_auto_none_without_api_falls_back(self):
        self.assertFalse(self._check(120, True, False, MoeA2ABackend.NONE, BLOCK_SIZE))

    def test_sm100_auto_none_stays_triton(self):
        self.assertFalse(self._check(100, False, True, MoeA2ABackend.NONE, BLOCK_SIZE))

    def test_sm90_auto_none_stays_triton(self):
        self.assertFalse(self._check(90, False, True, MoeA2ABackend.NONE, BLOCK_SIZE))

    def test_sm121_auto_none_not_auto_selected(self):
        # The wrapper's contiguous-API probe is SM120-only; keep GB10 on Triton.
        self.assertFalse(self._check(121, True, True, MoeA2ABackend.NONE, BLOCK_SIZE))

    def test_sm120_auto_none_non_blockwise_not_auto_selected(self):
        self.assertFalse(self._check(120, True, True, MoeA2ABackend.NONE, [1, 16]))
        self.assertFalse(self._check(120, True, True, MoeA2ABackend.NONE, None))

    def test_deepep_branch_unchanged(self):
        self.assertTrue(
            self._check(120, True, True, MoeA2ABackend.DEEPEP, BLOCK_SIZE)
        )
        self.assertFalse(
            self._check(120, True, False, MoeA2ABackend.DEEPEP, BLOCK_SIZE)
        )

    def test_explicit_deepgemm_unconditional(self):
        p_sm120 = patch.object(deep_gemm_sm120, "_is_sm120", False)
        p_enable = patch.object(deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", False)
        with p_sm120, p_enable:
            self.assertTrue(
                _resolve(MoeA2ABackend.NONE, MoeRunnerBackend.DEEP_GEMM, BLOCK_SIZE)
            )

    def test_is_supported_semantics(self):
        with (
            patch.object(deep_gemm_sm120, "_is_sm120", False),
            patch.object(deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", False),
        ):
            self.assertTrue(deep_gemm_sm120.is_supported())
        with (
            patch.object(deep_gemm_sm120, "_is_sm120", True),
            patch.object(deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", False),
        ):
            self.assertFalse(deep_gemm_sm120.is_supported())
        with (
            patch.object(deep_gemm_sm120, "_is_sm120", True),
            patch.object(deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", True),
        ):
            self.assertTrue(deep_gemm_sm120.is_supported())


if __name__ == "__main__":
    unittest.main()
