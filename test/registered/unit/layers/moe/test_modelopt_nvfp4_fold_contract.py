"""Contract probe: #44901's fold predicate on the flashinfer_cutlass NVFP4 route.

WHAT WE NOW BELIEVE, pinned at import/config level with zero device work. The
real end-to-end double-scale check is GPU-window debt; this file documents the
decision every CUDA FP4 boot takes today.

Our production FP4 arm is ModelOptNvFp4FusedMoEMethod on a FLASHINFER_CUTLASS
runner -- the backend `auto` resolves to on SM110/SM120 (and the value entry
230's fork patch publishes for it, so the dispatcher's expert-id convention
matches). For that combination:

  1. FusedMoE._fuses_routed_scaling_factor_in_topk (fused_moe_triton/layer.py)
     answers True: ModelOpt NVFP4 folds routed_scaling_factor into the top-k
     weights on EVERY non-Marlin runner -- the predicate list #44901's shared-
     slot fix keys off. Marlin answers False (it scales in its epilogue).
  2. The block passes that verdict as apply_routed_scaling_factor_on_output, so
     topk.is_routed_scaling_factor_fused_in_topk (#44901) also answers True.
  3. Consequently topk.fused_shared_expert_weight (#44901, consumed by
     biased_grouped_topk_gpu's CUDA append in #44904 and by
     remap_topk_for_per_rank_shared_slots) writes the fused shared column with
     1.0 -- NOT 1/routed_scaling_factor. An unfused route compensates by
     1/routed_scaling_factor instead, so exactly one scale lands on the shared
     contribution either way.

A flip of any answer below is either a fold-predicate change or a double-count
regression; both change model output on the NVFP4 arm.
"""

import unittest

from sglang.srt.layers.moe.fused_moe_triton.layer import (
    _fuses_routed_scaling_factor_in_topk,
)
from sglang.srt.layers.moe.topk import (
    TopKConfig,
    fused_shared_expert_weight,
    is_routed_scaling_factor_fused_in_topk,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptNvFp4FusedMoEMethod,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _nvfp4_stub(backend: MoeRunnerBackend) -> ModelOptNvFp4FusedMoEMethod:
    """A ModelOptNvFp4FusedMoEMethod that only knows its resolved runner.

    Built with object.__new__: the layer predicate reads exactly two things
    off the quant method -- the (absent) class-level
    fuse_routed_scaling_factor_in_topk flag and _moe_runner_backend -- so no
    config, weights, or device are involved.
    """
    method = object.__new__(ModelOptNvFp4FusedMoEMethod)
    method._moe_runner_backend = backend
    return method


class TestFoldPredicateNvfp4Cutlass(CustomTestCase):
    def test_nvfp4_class_has_no_blanket_fold_flag(self):
        # The verdict comes from the runner arm, not a class default; a
        # future class-level True would silently fold on Marlin too.
        self.assertFalse(
            hasattr(ModelOptNvFp4FusedMoEMethod, "fuse_routed_scaling_factor_in_topk")
        )

    def test_flashinfer_cutlass_nvfp4_folds_in_topk(self):
        # OUR ARM: fold == True, so the epilogue must not re-apply the factor.
        self.assertTrue(
            _fuses_routed_scaling_factor_in_topk(
                _nvfp4_stub(MoeRunnerBackend.FLASHINFER_CUTLASS)
            )
        )

    def test_marlin_nvfp4_does_not_fold_in_topk(self):
        self.assertFalse(
            _fuses_routed_scaling_factor_in_topk(_nvfp4_stub(MoeRunnerBackend.MARLIN))
        )

    def test_config_predicate_confirms_the_folded_verdict(self):
        # The block forwards the layer verdict as apply_routed_scaling_factor_on_output.
        self.assertTrue(
            is_routed_scaling_factor_fused_in_topk(
                TopKConfig(top_k=10, apply_routed_scaling_factor_on_output=True)
            )
        )

    def test_config_predicate_rejects_an_unfolded_softmax_route(self):
        # Ungrouped softmax without the flag: no fold, on CUDA or aiter alike
        # (the predicate's aiter arms need grouped+bias or sqrtsoftplus).
        self.assertFalse(
            is_routed_scaling_factor_fused_in_topk(
                TopKConfig(top_k=10, apply_routed_scaling_factor_on_output=False)
            )
        )

    def test_shared_slot_weight_on_the_folded_nvfp4_arm(self):
        # The decision that #44901 fixed: folded => plain 1.0, never 1/rsf.
        self.assertEqual(
            fused_shared_expert_weight(routed_scaling_factor=2.5, fused_in_topk=True),
            1.0,
        )

    def test_shared_slot_weight_compensates_when_unfolded(self):
        # Unfused => the shared column pre-divides the epilogue's multiply.
        self.assertEqual(
            fused_shared_expert_weight(routed_scaling_factor=2.5, fused_in_topk=False),
            1.0 / 2.5,
        )

    def test_shared_slot_weight_without_a_factor_is_one(self):
        self.assertEqual(
            fused_shared_expert_weight(routed_scaling_factor=None, fused_in_topk=False),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
