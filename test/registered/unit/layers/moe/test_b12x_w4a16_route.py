"""CPU coverage for the flashinfer SM120 b12x W4A16 MoE route.

The kernel itself only runs on an SM120 device with a CUDA 13 toolkit, so this
file pins everything that decides *what gets passed to it*: the two contract
adaptations (``source_format`` and the global scales), the gate/up row-order
string, the EP expert map, the launch argument set against the installed
flashinfer's real signature, the actionable gate message, and the fact that
nothing selects the route unless the operator names it.
"""

import dataclasses
import inspect
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import sglang.srt.layers.moe.moe_runner.flashinfer_cutlass as fic
import sglang.srt.layers.quantization.modelopt_quant as modelopt
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig, FusedOpPool
from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
    B12X_W4A16_MIN_CUDA_MAJOR,
    B12X_W4A16_SOURCE_FORMAT,
    B12X_W4A16_W13_LAYOUT,
    B12X_W4A16_W13_LAYOUT_UP_FIRST,
    FlashInferCutlassMoeQuantInfo,
    _b12x_w4a16_expert_map,
    _run_flashinfer_b12x_w4a16,
    b12x_w4a16_token_capacities,
    flashinfer_b12x_w4a16_unavailable_reason,
    prewarm_flashinfer_b12x_w4a16,
)
from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.quantization.modelopt_quant import (
    b12x_w4a16_w13_layout,
    prepare_flashinfer_b12x_w4a16_weights,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

# The flashinfer 0.7.2 wheel imports these SM12x leaves on a CPU-only host, so
# its tables and signatures can be checked directly. A wheel without them skips
# the tests that consult it rather than failing.
try:
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import (
        moe_w4a16_prepare as _w4a16_prepare,
    )
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_prepare import (
        _SOURCE_FORMATS,
    )

    _HAS_SM12X_B12X = True
except Exception:  # pragma: no cover - wheel without SM12x b12x
    _w4a16_prepare = None
    _SOURCE_FORMATS = {}
    _HAS_SM12X_B12X = False

requires_sm12x_b12x = unittest.skipUnless(
    _HAS_SM12X_B12X, "installed flashinfer has no SM12x b12x module"
)

_PREPARER = (
    "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_prepare."
    "prepare_w4a16_packed_weights"
)
_FC = "sglang.srt.layers.moe.moe_runner.flashinfer_cutlass."
_MO = "sglang.srt.layers.quantization.modelopt_quant."

# The real nvidia/Qwen3.8-Flash-Next-NVFP4 export carries per-tensor calibrated
# activation scales of this order (gate/up 1.87e-3, down 2.41e-3). That is
# precisely the number that must NOT appear in a W4A16 alpha.
_CKPT_INPUT_SCALE = 0.00187174
_CKPT_WEIGHT_SCALE_2 = 0.5


class _PatchHelper(CustomTestCase):
    """Reversible patching with both ``return_value`` and ``new`` semantics."""

    def _patch(self, target, value):
        """``target`` keeps being callable and returns ``value``."""
        patcher = patch(target, return_value=value)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _patch_new(self, target, replacement):
        """``target`` becomes ``replacement`` (for substituting a function)."""
        patcher = patch(target, new=replacement)
        patcher.start()
        self.addCleanup(patcher.stop)


def _loaded_nvfp4_tensors(layer):
    """Stand in for the weight loader: block scales, global scales, input scales."""
    for prefix in ("w13", "w2"):
        scale = getattr(layer, prefix + "_weight_scale")
        values = (
            (torch.arange(scale.numel(), dtype=torch.float32).reshape(scale.shape) % 32)
            + 1
        ).to(torch.float8_e4m3fn)
        scale.data.copy_(values)
        getattr(layer, f"{prefix}_weight_scale_2").data.fill_(_CKPT_WEIGHT_SCALE_2)
        getattr(layer, f"{prefix}_input_scale").data.fill_(_CKPT_INPUT_SCALE)
        getattr(layer, f"{prefix}_weight").data.zero_()


class TestRouteIsOptIn(_PatchHelper):
    """Nothing selects the new backend on its own; nothing else changes either."""

    def test_backend_is_a_named_choice(self):
        self.assertEqual(
            MoeRunnerBackend.FLASHINFER_B12X_W4A16.value, "flashinfer_b12x_w4a16"
        )
        from sglang.srt.arg_groups.choices import MOE_RUNNER_BACKEND_CHOICES

        self.assertIn("flashinfer_b12x_w4a16", MOE_RUNNER_BACKEND_CHOICES)

    def test_predicates_do_not_leak_into_the_cutlass_route(self):
        """Every existing ``is_flashinfer_cutlass()`` reader must stay put.

        The b12x route deliberately reuses flashinfer_cutlass.py's quant-info
        type, its dispatch entry point and the fused-func registry, so a
        predicate leak would silently move the FP8/CUTLASS paths (dispatcher
        expert-id convention, draft-backend downgrade, LoRA mem pool, ...).
        """
        b12x = MoeRunnerBackend.FLASHINFER_B12X_W4A16
        self.assertTrue(b12x.is_flashinfer_b12x_w4a16())
        self.assertFalse(b12x.is_flashinfer_cutlass())
        self.assertFalse(b12x.is_flashinfer_cutedsl())
        self.assertFalse(b12x.is_auto())
        self.assertFalse(b12x.is_triton())
        cutlass = MoeRunnerBackend.FLASHINFER_CUTLASS
        self.assertTrue(cutlass.is_flashinfer_cutlass())
        self.assertFalse(cutlass.is_flashinfer_b12x_w4a16())

    def test_method_route_flag_follows_the_named_backend_only(self):
        for backend, expect_b12x in [
            (MoeRunnerBackend.FLASHINFER_B12X_W4A16, True),
            (MoeRunnerBackend.FLASHINFER_CUTLASS, False),
            (MoeRunnerBackend.AUTO, False),
            (MoeRunnerBackend.TRITON, False),
        ]:
            method = object.__new__(modelopt.ModelOptNvFp4FusedMoEMethod)
            method._moe_runner_backend = backend
            with self.subTest(backend=backend):
                self.assertEqual(method._use_flashinfer_b12x_w4a16, expect_b12x)
                self.assertEqual(
                    method.enable_flashinfer_cutlass_moe,
                    backend is MoeRunnerBackend.FLASHINFER_CUTLASS,
                )

    def test_up_proj_order_is_not_flipped_for_b12x(self):
        """b12x must not inherit the [Up, Gate] load order of flashinfer_cutlass."""
        for backend, up_first in [
            (MoeRunnerBackend.FLASHINFER_CUTLASS, True),
            (MoeRunnerBackend.FLASHINFER_B12X_W4A16, False),
        ]:
            with self.subTest(backend=backend):
                method = object.__new__(modelopt.ModelOptNvFp4FusedMoEMethod)
                method._moe_runner_backend = backend
                method.moe_runner_config = MoeRunnerConfig(is_gated=True)
                self.assertEqual(method.load_up_proj_weight_first, up_first)
                # The layout string is derived from that same predicate.
                self.assertEqual(
                    b12x_w4a16_w13_layout(method.load_up_proj_weight_first),
                    B12X_W4A16_W13_LAYOUT_UP_FIRST
                    if up_first
                    else B12X_W4A16_W13_LAYOUT,
                )

    def test_runner_raises_the_gate_reason(self):
        self._patch(
            _FC + "flashinfer_b12x_w4a16_unavailable_reason",
            f"requires CUDA {B12X_W4A16_MIN_CUDA_MAJOR} or later",
        )
        with self.assertRaises(ValueError) as ctx:
            MoeRunner(MoeRunnerBackend.FLASHINFER_B12X_W4A16, MoeRunnerConfig())
        message = str(ctx.exception)
        self.assertIn("--moe-runner-backend flashinfer_b12x_w4a16", message)
        self.assertIn(f"CUDA {B12X_W4A16_MIN_CUDA_MAJOR}", message)

    def test_fused_func_is_registered_for_a2a_none(self):
        self.assertIsNotNone(
            FusedOpPool.get_fused_func(
                MoeA2ABackend.NONE.value, MoeRunnerBackend.FLASHINFER_B12X_W4A16.value
            )
        )


class TestHardwareGateMessage(_PatchHelper):
    """The refusal must name the fix, because the box looks CUDA-13 already."""

    def _fake_device(self, capability):
        self._patch_new("torch.cuda.is_available", lambda: True)
        self._patch_new("torch.cuda.get_device_capability", lambda *a: capability)

    def test_no_device(self):
        self._patch_new("torch.cuda.is_available", lambda: False)
        reason = flashinfer_b12x_w4a16_unavailable_reason()
        self.assertIsNotNone(reason)
        self.assertIn("CUDA device", reason)

    def test_other_capability(self):
        self._fake_device((10, 0))
        reason = flashinfer_b12x_w4a16_unavailable_reason()
        self.assertIsNotNone(reason)
        self.assertIn("SM120", reason)
        self.assertIn("sm100", reason)

    def test_hopper_is_also_rejected(self):
        self._fake_device((9, 0))
        self.assertIn("sm90", flashinfer_b12x_w4a16_unavailable_reason())

    @requires_sm12x_b12x
    def test_cuda_12_toolkit_prefix_explains_cuda_home(self):
        self._fake_device((12, 0))
        self._patch(_FC + "_b12x_w4a16_cuda_major", 12)
        reason = flashinfer_b12x_w4a16_unavailable_reason()
        self.assertIsNotNone(reason)
        # Version, real cause (nvcc, not the torch build), fix, one-time cost,
        # and the fallback backend.
        for needle in (
            f"requires CUDA {B12X_W4A16_MIN_CUDA_MAJOR}",
            "$CUDA_HOME/bin/nvcc",
            "CUDA_HOME",
            "one-time",
            "--moe-runner-backend flashinfer_cutlass",
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, reason)

    @requires_sm12x_b12x
    def test_satisfied_gate_returns_none(self):
        self._fake_device((12, 0))
        self._patch(_FC + "_b12x_w4a16_cuda_major", B12X_W4A16_MIN_CUDA_MAJOR)
        self.assertIsNone(flashinfer_b12x_w4a16_unavailable_reason())

    def test_unknown_toolkit_version_does_not_claim_a_cuda_problem(self):
        """If flashinfer's own probe fails, do not invent a version number."""
        self._fake_device((12, 0))
        self._patch(_FC + "_b12x_w4a16_cuda_major", None)
        self.assertIsNone(flashinfer_b12x_w4a16_unavailable_reason())

    def test_gate_never_imports_flashinfer_eagerly(self):
        """Selection-time only: the b12x import stays behind the cheap checks."""
        self._patch_new("torch.cuda.is_available", lambda: False)
        with patch.object(fic, "b12x_w4a16_launch") as launch:
            flashinfer_b12x_w4a16_unavailable_reason()
            launch.assert_not_called()


class TestContractAdaptations(_PatchHelper):
    """Adaptation 1 (``source_format``) and the row-order string, vs the wheel."""

    @requires_sm12x_b12x
    def test_source_format_is_a_real_wheel_key(self):
        self.assertIn(B12X_W4A16_SOURCE_FORMAT, _SOURCE_FORMATS)
        self.assertEqual(
            _w4a16_prepare._normalize_source_format(B12X_W4A16_SOURCE_FORMAT),
            "modelopt",
        )

    @requires_sm12x_b12x
    def test_upstream_pr_spelling_is_rejected_by_this_wheel(self):
        """#45298 passes "modelopt_e4m3_k32" (from flashinfer #6824): absent here."""
        self.assertNotIn("modelopt_e4m3_k32", _SOURCE_FORMATS)
        with self.assertRaises(ValueError):
            _w4a16_prepare._normalize_source_format("modelopt_e4m3_k32")

    @requires_sm12x_b12x
    def test_modelopt_is_the_k16_e4m3_grid_our_checkpoint_has(self):
        """K/32-E8M0 would read the wrong scale grid; modelopt means K/16 E4M3."""
        self.assertNotEqual(B12X_W4A16_SOURCE_FORMAT, "fp4_e8m0_k32")
        doc = _w4a16_prepare.prepare_w4a16_modelopt_nvfp4_weights.__doc__ or ""
        self.assertIn("K/16", doc)
        self.assertIn("swizzled", doc)

    @requires_sm12x_b12x
    def test_compressed_tensors_would_invert_the_global_scale(self):
        """The choice is not cosmetic: compressed_tensors reciprocates it."""
        source = torch.tensor([0.25], dtype=torch.float32)
        self.assertEqual(
            float(
                _w4a16_prepare._source_global_scale(
                    source, source_format=B12X_W4A16_SOURCE_FORMAT
                )[0]
            ),
            0.25,
        )
        self.assertEqual(
            float(
                _w4a16_prepare._source_global_scale(
                    source, source_format="compressed_tensors"
                )[0]
            ),
            4.0,
        )

    def test_layout_strings_are_the_w31_w13_pair(self):
        self.assertEqual(B12X_W4A16_W13_LAYOUT, "w31")
        self.assertEqual(B12X_W4A16_W13_LAYOUT_UP_FIRST, "w13")
        self.assertEqual(b12x_w4a16_w13_layout(False), "w31")
        self.assertEqual(b12x_w4a16_w13_layout(True), "w13")

    @requires_sm12x_b12x
    def test_layout_strings_mean_what_this_route_needs(self):
        """Wheel-side: "w31" is already kernel-native gate/up, "w13" needs a swap."""
        normalize = _w4a16_prepare._normalize_w13_layout
        self.assertEqual(normalize("w31"), "w31")
        self.assertEqual(normalize("gate_up"), "w31")
        self.assertEqual(normalize("w13"), "w13")
        self.assertEqual(normalize("up_gate"), "w13")
        # Our layer is gate-first, so it must ask for the no-rotation spelling.
        self.assertEqual(b12x_w4a16_w13_layout(False), normalize("gate_up"))


class TestGlobalScaleDerivation(_PatchHelper):
    """Adaptation 2: the alpha must be weight-only, and that is arithmetic."""

    def setUp(self):
        super().setUp()
        # w13_weight_scale_2 is [E, 2]: col 0 gate, col 1 up. The route's input
        # scale arrives 1-D over experts (layer.w13_input_scale.max(dim=-1)).
        self.ws2 = torch.tensor([[0.5, 0.5], [0.25, 0.25]], dtype=torch.float32)
        self.ones = torch.ones(2, dtype=torch.float32)
        self.real = torch.full((2,), _CKPT_INPUT_SCALE, dtype=torch.float32)

    def test_ones_input_scale_yields_weight_only_alpha(self):
        gate, up = modelopt._compute_gemm1_alphas(self.ws2, self.ones, True)
        self.assertTrue(torch.equal(gate, self.ws2[:, 0]))
        self.assertTrue(torch.equal(up, self.ws2[:, 1]))

    def test_weight_only_equals_folded_product_divided_by_input_scale(self):
        """The relationship the route relies on, stated as an identity.

        sglang folds ``g1_alphas = w13_input_scale * w13_weight_scale_2``. W4A16
        wants only ``w13_weight_scale_2``, so either side of this equation is a
        correct way to obtain it -- the port chooses to neutralize the input
        scale so the file keeps one meaning for ``g1_alphas``.
        """
        folded_gate, folded_up = modelopt._compute_gemm1_alphas(
            self.ws2, self.real, True
        )
        weight_only_gate, weight_only_up = modelopt._compute_gemm1_alphas(
            self.ws2, self.ones, True
        )
        self.assertTrue(
            torch.allclose(folded_gate / self.real, weight_only_gate, atol=1e-7)
        )
        self.assertTrue(
            torch.allclose(folded_up / self.real, weight_only_up, atol=1e-7)
        )
        # And how wrong the folded value is: a factor of the input scale (~534x).
        self.assertAlmostEqual(
            float(folded_gate[0] / weight_only_gate[0]), _CKPT_INPUT_SCALE, places=7
        )
        self.assertAlmostEqual(
            float(weight_only_gate[0]), _CKPT_WEIGHT_SCALE_2, places=7
        )

    def test_shared_scale_checkpoint_has_equal_gate_and_up(self):
        gate, up = modelopt._compute_gemm1_alphas(self.ws2, self.ones, True)
        self.assertTrue(torch.equal(gate, up))

    @requires_sm12x_b12x
    def test_w4a16_never_folds_the_activation_scale(self):
        """Wheel-side half of the same claim."""
        doc = _w4a16_prepare.prepare_w4a16_modelopt_nvfp4_weights.__doc__ or ""
        self.assertIn("input scales are not folded into W4A16", doc)
        # And the global scales it takes are documented as raw weight scales.
        self.assertIn("raw ModelOpt weight global scales", doc)


@dataclasses.dataclass(frozen=True)
class _LocalPrepared:
    """Stand-in for the wheel's W4A16PackedWeights when flashinfer is absent.

    Only the fields the port touches: a frozen dataclass, because the real one
    is and the route's storage reuse goes through ``dataclasses.replace``.
    """

    w13: torch.Tensor
    w13_scale: torch.Tensor
    w13_global_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    w2_global_scale: torch.Tensor
    workspace: torch.Tensor
    hidden_size: int
    intermediate_size: int
    num_experts: int
    is_gated: bool
    params_dtype: torch.dtype
    source_format: str = "modelopt"
    w13_layout: str = "w13"
    weight_layout: str = "packed"
    scale_format: str = "e4m3_k16"


def _make_prepared(**fields):
    """Build the real wheel record when the wheel is importable."""
    cls = _w4a16_prepare.W4A16PackedWeights if _HAS_SM12X_B12X else _LocalPrepared
    fields.setdefault("workspace", torch.zeros(4, dtype=torch.int32))
    fields.setdefault("hidden_size", 128)
    fields.setdefault("intermediate_size", 128)
    fields.setdefault("num_experts", 2)
    fields.setdefault("is_gated", True)
    fields.setdefault("params_dtype", torch.bfloat16)
    return cls(**fields)


def _fake_preparer(calls):
    """Stand-in for flashinfer's preparer: returns views, records the call."""

    def prepare_w4a16_packed_weights(*args, **kwargs):
        merged = dict(kwargs)
        for index, name in enumerate(
            (
                "w13_fp4",
                "w13_scale",
                "w13_global_scale",
                "w2_fp4",
                "w2_scale",
                "w2_global_scale",
            )
        ):
            if name not in merged:
                merged[name] = args[index] if index < len(args) else None
        calls.append(merged)
        return _make_prepared(
            w13=merged["w13_fp4"],
            w2=merged["w2_fp4"],
            # The real preparer permutes the grid but never changes its volume,
            # which is what the in-place storage reuse depends on.
            w13_scale=merged["w13_scale"].detach().clone().view(torch.uint8),
            w2_scale=merged["w2_scale"].detach().clone().view(torch.uint8),
            w13_global_scale=merged["w13_global_scale"].detach().clone(),
            w2_global_scale=merged["w2_global_scale"].detach().clone(),
            hidden_size=int(merged["w2_fp4"].shape[2]) * 2,
            intermediate_size=int(merged["w13_fp4"].shape[2]),
            num_experts=int(merged["w13_fp4"].shape[0]),
            source_format=merged.get("source_format"),
            w13_layout=merged.get("w13_layout"),
        )

    return prepare_w4a16_packed_weights


class TestWeightPreparation(_PatchHelper):
    """The load-time preparation: ordering, argument set, storage reuse."""

    def setUp(self):
        super().setUp()
        self.calls = []

    def _stub_layer_and_method(self, backend):
        """A serialized NVFP4 MoE layer that has gone through the loader."""
        method = object.__new__(modelopt.ModelOptNvFp4FusedMoEMethod)
        method.enable_flashinfer_trtllm_moe = False
        method.quant_config = SimpleNamespace(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            use_per_token_activation=False,
            get_name=lambda: "modelopt",
        )
        method.prepare_weight_loader = Mock(return_value=None)

        layer = torch.nn.Module()
        layer.num_experts = layer.num_local_experts = 2
        layer.num_fused_shared_experts = 0
        layer.num_global_routed_experts = 2
        layer.moe_ep_size = 1
        layer.moe_ep_rank = 0
        layer.moe_runner_config = MoeRunnerConfig(
            is_gated=True,
            activation="silu",
            top_k=2,
            num_experts=2,
            num_local_experts=2,
        )
        layer.dispatcher = Mock()
        method.moe_runner_config = layer.moe_runner_config

        self._patch("sglang.srt.layers.moe.get_moe_runner_backend", backend)
        self._patch(_MO + "get_moe_runner_backend", backend)
        self._patch(_MO + "get_moe_a2a_backend", MoeA2ABackend.NONE)
        self._patch(_MO + "_use_nvfp4_dispatch", False)

        with torch.device("cpu"):
            method.create_weights(layer, 2, 128, 128, torch.bfloat16)
        _loaded_nvfp4_tensors(layer)
        return method, layer

    def _run_pwal(self, backend, *, scale_2_cols=None):
        method, layer = self._stub_layer_and_method(backend)
        if scale_2_cols is not None:
            layer.w13_weight_scale_2.data.copy_(scale_2_cols)
        with patch(_PREPARER, _fake_preparer(self.calls), create=False):
            method.process_weights_after_loading(layer)
        return method, layer

    @requires_sm12x_b12x
    def test_b12x_passes_weight_only_alpha_and_the_modelopt_format(self):
        _, layer = self._run_pwal(MoeRunnerBackend.FLASHINFER_B12X_W4A16)
        self.assertEqual(len(self.calls), 1)
        call = self.calls[0]
        # The alphas handed to the preparer are weight-only even though the
        # checkpoint's input scale is 1.87e-3 -- adaptation 2, end to end.
        self.assertTrue(
            torch.allclose(call["w13_global_scale"], layer.w13_weight_scale_2[:, 0])
        )
        self.assertTrue(
            torch.allclose(call["w2_global_scale"], layer.w2_weight_scale_2.reshape(2))
        )
        self.assertNotAlmostEqual(
            float(call["w13_global_scale"][0]),
            _CKPT_INPUT_SCALE * _CKPT_WEIGHT_SCALE_2,
            places=6,
        )
        self.assertEqual(call["source_format"], "modelopt")
        self.assertEqual(call["w13_layout"], "w31")
        self.assertEqual(call["activation"], "silu")
        self.assertIs(call["reuse_input_storage"], True)
        self.assertIs(call["params_dtype"], torch.bfloat16)
        # And the layer's own g1_alphas are weight-only, not the folded product.
        self.assertTrue(
            torch.allclose(
                layer.g1_alphas.detach().float(), layer.w13_weight_scale_2[:, 0]
            )
        )

    @requires_sm12x_b12x
    def test_preparation_consumes_the_rebuilt_swizzled_scales(self):
        """Swizzle timing: entry #42452 defers them, so b12x must run after."""
        _, layer = self._run_pwal(MoeRunnerBackend.FLASHINFER_B12X_W4A16)
        self.assertIsNotNone(layer.w13_blockscale_swizzled)
        self.assertIs(self.calls[0]["w13_scale"], layer.w13_blockscale_swizzled)
        self.assertEqual(layer.w13_blockscale_swizzled.dtype, torch.float8_e4m3fn)
        # Storage reuse: the prepared grid must be a VIEW of the layer's own
        # scale bytes (a second allocation is the whole thing this avoids), so
        # the data pointers match even though the tensor objects differ.
        prepared = layer._flashinfer_b12x_w4a16_prepared_weights
        self.assertEqual(
            prepared.w13_scale.data_ptr(), layer.w13_blockscale_swizzled.data_ptr()
        )
        self.assertEqual(
            prepared.w2_scale.data_ptr(), layer.w2_blockscale_swizzled.data_ptr()
        )
        self.assertEqual(prepared.w13_scale.dtype, torch.uint8)
        self.assertEqual(
            prepared.w13_scale.numel(), layer.w13_blockscale_swizzled.numel()
        )
        self.assertEqual(prepared.w13.data_ptr(), layer.w13_weight.data_ptr())
        self.assertEqual(prepared.w2.data_ptr(), layer.w2_weight.data_ptr())

    def test_cutlass_route_never_prepares_and_keeps_the_folded_alpha(self):
        """Gate off by default: the same stub through the existing route."""
        method, layer = self._stub_layer_and_method(MoeRunnerBackend.FLASHINFER_CUTLASS)
        with patch(_PREPARER, _fake_preparer(self.calls), create=False):
            method.process_weights_after_loading(layer)
        self.assertEqual(self.calls, [])
        self.assertIsNone(
            getattr(layer, "_flashinfer_b12x_w4a16_prepared_weights", None)
        )
        # g1_alphas keeps input_scale * weight_scale_2: today's behaviour, and
        # exactly the value that would be a ~534x error for a W4A16 kernel.
        expected = torch.full(
            (2,), _CKPT_INPUT_SCALE * _CKPT_WEIGHT_SCALE_2, dtype=torch.float32
        )
        self.assertTrue(torch.allclose(layer.g1_alphas.detach().float(), expected))

    @requires_sm12x_b12x
    def test_distinct_gate_up_global_scales_are_rejected(self):
        """A single-alpha route cannot represent per-half global scales."""
        bad = torch.tensor([[0.5, 0.125], [0.5, 0.125]], dtype=torch.float32)
        with self.assertRaises(ValueError) as ctx:
            self._run_pwal(MoeRunnerBackend.FLASHINFER_B12X_W4A16, scale_2_cols=bad)
        self.assertIn("gate/up", str(ctx.exception))
        self.assertEqual(self.calls, [])

    def test_prepare_refuses_before_the_swizzle_exists(self):
        layer = SimpleNamespace(
            w13_blockscale_swizzled=None,
            w2_blockscale_swizzled=Mock(),
            w13_weight=Mock(),
        )
        with self.assertRaises(RuntimeError) as ctx:
            prepare_flashinfer_b12x_w4a16_weights(layer, up_first=False)
        self.assertIn("w13_blockscale_swizzled", str(ctx.exception))

    def test_scale_storage_size_mismatch_raises(self):
        source = torch.zeros(8, dtype=torch.float8_e4m3fn)
        packed = torch.zeros(4, dtype=torch.uint8)
        with self.assertRaises(ValueError) as ctx:
            modelopt._reuse_b12x_scale_storage(source, packed, name="W13")
        self.assertIn("W13", str(ctx.exception))

    @requires_sm12x_b12x
    def test_warm_payload_is_the_apply_payload(self):
        method, layer = self._run_pwal(MoeRunnerBackend.FLASHINFER_B12X_W4A16)
        quant_info, runner_config = method.b12x_w4a16_warm_payload(layer)
        self.assertIsInstance(quant_info, FlashInferCutlassMoeQuantInfo)
        self.assertEqual(quant_info.quant_type, "w4a16_b12x")
        self.assertEqual(len(quant_info.quant_scales), 4)
        self.assertIs(
            quant_info.prepared_weights, layer._flashinfer_b12x_w4a16_prepared_weights
        )
        self.assertIs(quant_info.w13_weight, layer.w13_weight)
        self.assertIs(runner_config, method.moe_runner_config)
        # The other backend never yields a b12x payload.
        method._moe_runner_backend = MoeRunnerBackend.FLASHINFER_CUTLASS
        self.assertEqual(method.b12x_w4a16_warm_payload(layer), (None, None))

    @requires_sm12x_b12x
    def test_contract_guard_rejects_the_wrong_group_size(self):
        method, layer = self._run_pwal(MoeRunnerBackend.FLASHINFER_B12X_W4A16)
        method.quant_config.group_size = 32
        reason = method._b12x_w4a16_contract_violation(layer)
        self.assertIsNotNone(reason)
        self.assertIn("group_size=32", reason)
        method.quant_config.group_size = 16
        self.assertIsNone(method._b12x_w4a16_contract_violation(layer))
        layer.num_fused_shared_experts = 1
        self.assertIn("shared experts", method._b12x_w4a16_contract_violation(layer))


def _b12x_dispatch_output(tokens=3, hidden=8, dtype=torch.bfloat16):
    return SimpleNamespace(
        hidden_states=torch.zeros(tokens, hidden, dtype=dtype),
        topk_output=SimpleNamespace(
            topk_ids=torch.arange(tokens * 2, dtype=torch.int64).reshape(tokens, 2) % 4,
            topk_weights=torch.full((tokens, 2), 0.5, dtype=torch.float32),
        ),
    )


def _b12x_runner_config(**overrides):
    base = dict(
        num_experts=4,
        num_local_experts=4,
        top_k=2,
        moe_tp_size=1,
        moe_tp_rank=0,
        moe_ep_size=1,
        moe_ep_rank=0,
        activation="silu",
        is_gated=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _capture_launch(calls):
    def launch(*args, **kwargs):
        calls.append((args, kwargs))
        return kwargs["scatter_output"]

    return launch


class TestLaunchContract(_PatchHelper):
    """What actually reaches ``launch_sm120_moe``."""

    def setUp(self):
        super().setUp()
        self.calls = []
        self._patch(_FC + "get_moe_a2a_backend", MoeA2ABackend.NONE)
        self._patch_new(_FC + "use_symmetric_memory", lambda *a, **k: nullcontext())
        self._patch(_FC + "is_allocation_symmetric", False)
        self._patch(_FC + "get_parallel", Mock())
        self._patch(_FC + "b12x_w4a16_launch", _capture_launch(self.calls))
        self.quant_info = FlashInferCutlassMoeQuantInfo(
            quant_type="w4a16_b12x",
            w13_weight=torch.zeros(4, 16, 4, dtype=torch.uint8),
            w2_weight=torch.zeros(4, 8, 4, dtype=torch.uint8),
            output_dtype=torch.bfloat16,
            quant_scales=[
                torch.zeros(4, 16, 1, dtype=torch.uint8),
                torch.ones(4, dtype=torch.float32),
                torch.zeros(4, 8, 1, dtype=torch.uint8),
                torch.ones(4, dtype=torch.float32),
            ],
            apply_routed_scaling_factor=False,
            prepared_weights=SimpleNamespace(tag="prepared"),
        )

    def _run(
        self, runner_config=None, *, output=None, alltoall=False, dtype=torch.bfloat16
    ):
        return _run_flashinfer_b12x_w4a16(
            dispatch_output=_b12x_dispatch_output(dtype=dtype),
            quant_info=self.quant_info,
            runner_config=runner_config or _b12x_runner_config(),
            output=output,
            enable_alltoall=alltoall,
        )

    @requires_sm12x_b12x
    def test_kwargs_are_real_wheel_parameters(self):
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
            launch_sm120_moe,
        )

        self._run()
        self.assertEqual(len(self.calls), 1)
        _, kwargs = self.calls[0]
        accepted = set(inspect.signature(launch_sm120_moe).parameters)
        self.assertFalse(set(kwargs) - accepted, sorted(set(kwargs) - accepted))
        for required in (
            "a",
            "w1_weight",
            "w1_weight_sf",
            "w1_alpha",
            "w2_weight",
            "w2_weight_sf",
            "w2_alpha",
            "topk_ids",
            "topk_weights",
            "num_experts",
            "num_local_experts",
            "top_k",
            "scatter_output",
            "expert_map",
            "activation",
            "quant_mode",
            "source_format",
            "_prepared_weights",
        ):
            with self.subTest(arg=required):
                self.assertIn(required, kwargs)

    def test_route_arguments(self):
        self._run()
        _, kwargs = self.calls[0]
        self.assertEqual(kwargs["quant_mode"], "w4a16")
        self.assertEqual(kwargs["source_format"], B12X_W4A16_SOURCE_FORMAT)
        self.assertEqual(kwargs["activation"], "silu")
        self.assertIs(kwargs["_prepared_weights"], self.quant_info.prepared_weights)
        # The prepared grid, not the layer's logical scales, and one alpha pair.
        w13_scale, w13_global, w2_scale, w2_global = self.quant_info.quant_scales
        self.assertIs(kwargs["w1_weight_sf"], w13_scale)
        self.assertIs(kwargs["w1_alpha"], w13_global)
        self.assertIs(kwargs["w2_weight_sf"], w2_scale)
        self.assertIs(kwargs["w2_alpha"], w2_global)
        self.assertEqual(kwargs["num_experts"], 4)
        self.assertEqual(kwargs["num_local_experts"], 4)
        self.assertEqual(kwargs["top_k"], 2)
        # int32 ids: the wheel's own route packs to torch.int32 (:1540 of
        # moe_dispatch.py) and the kernels specialize on that width.
        self.assertIs(kwargs["topk_ids"].dtype, torch.int32)

    def test_no_ep_passes_no_map_and_reuses_the_supplied_output(self):
        out = torch.zeros(3, 8, dtype=torch.bfloat16)
        result = self._run(output=out)
        _, kwargs = self.calls[0]
        self.assertIsNone(kwargs["expert_map"])
        self.assertIs(kwargs["scatter_output"], out)
        self.assertIs(result, out)

    def test_ep_passes_a_map_and_zeroes_the_output(self):
        rc = _b12x_runner_config(
            num_experts=8, num_local_experts=2, moe_ep_size=4, moe_ep_rank=1
        )
        self._run(rc)
        _, kwargs = self.calls[0]
        expert_map = kwargs["expert_map"]
        self.assertEqual(expert_map.numel(), 8)
        self.assertTrue(
            torch.equal(expert_map[2:4], torch.tensor([0, 1], dtype=torch.int32))
        )
        # EP partial-sum all-reduce: rows this rank does not own must be zero,
        # not allocator garbage.
        self.assertTrue(torch.equal(kwargs["scatter_output"], torch.zeros(3, 8)))

    def test_without_ep_the_output_is_not_zero_filled(self):
        self._run()
        out = self.calls[0][1]["scatter_output"]
        self.assertEqual(tuple(out.shape), (3, 8))
        self.assertIsNone(self.calls[0][1]["expert_map"])

    def test_rejects_alltoall(self):
        with self.assertRaises(NotImplementedError) as ctx:
            self._run(alltoall=True)
        self.assertIn("all-to-all", str(ctx.exception))

    def test_rejects_a_non_none_a2a_backend(self):
        self._patch(_FC + "get_moe_a2a_backend", MoeA2ABackend.DEEPEP)
        with self.assertRaises(NotImplementedError) as ctx:
            self._run()
        self.assertIn("--moe-a2a-backend none", str(ctx.exception))

    def test_rejects_unsupported_contract_shapes(self):
        cases = {
            "activation": _b12x_runner_config(activation="relu2"),
            "ungated": _b12x_runner_config(is_gated=False),
            "moe_tp": _b12x_runner_config(moe_tp_size=2),
            "missing_top_k": _b12x_runner_config(top_k=None),
            "missing_num_experts": _b12x_runner_config(num_experts=None),
        }
        for name, rc in cases.items():
            with self.subTest(case=name):
                with self.assertRaises((NotImplementedError, ValueError)):
                    self._run(rc)

    def test_requires_prepared_weights(self):
        self.quant_info.prepared_weights = None
        with self.assertRaises(RuntimeError) as ctx:
            self._run()
        self.assertIn("prepared weights", str(ctx.exception))

    def test_requires_four_scales(self):
        self.quant_info.quant_scales = self.quant_info.quant_scales[:3]
        with self.assertRaises(ValueError) as ctx:
            self._run()
        self.assertIn("four prepared scales", str(ctx.exception))

    def test_requires_bf16_activations(self):
        with self.assertRaises(ValueError) as ctx:
            self._run(dtype=torch.float16)
        self.assertIn("bf16", str(ctx.exception))

    def test_dispatch_delegates_on_quant_type_before_touching_cutlass(self):
        seen = []

        def fake(**kwargs):
            seen.append(kwargs)
            return kwargs["output"]

        self._patch_new(_FC + "_run_flashinfer_b12x_w4a16", fake)
        out = torch.zeros(3, 8, dtype=torch.bfloat16)
        result = fic._run_flashinfer_cutlass(
            dispatch_output=_b12x_dispatch_output(),
            quant_info=self.quant_info,
            runner_config=_b12x_runner_config(),
            output=out,
            enable_alltoall=False,
        )
        self.assertIs(result, out)
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0]["quant_info"], self.quant_info)

    def test_fused_func_returns_a_combine_input(self):
        result = fic.fused_experts_none_to_flashinfer_b12x_w4a16(
            dispatch_output=_b12x_dispatch_output(),
            quant_info=self.quant_info,
            runner_config=_b12x_runner_config(),
        )
        self.assertIsInstance(result, StandardCombineInput)
        self.assertEqual(tuple(result.hidden_states.shape), (3, 8))

    def test_fused_func_rejects_a_foreign_quant_type(self):
        self.quant_info.quant_type = "fp4"
        with self.assertRaises(AssertionError):
            fic.fused_experts_none_to_flashinfer_b12x_w4a16(
                dispatch_output=_b12x_dispatch_output(),
                quant_info=self.quant_info,
                runner_config=_b12x_runner_config(),
            )


class TestArgumentValidation(_PatchHelper):
    """Server-arg resolution refuses a b12x launch that cannot run."""

    def _view(self, **overrides):
        base = dict(
            moe_runner_backend="flashinfer_b12x_w4a16",
            quantization="modelopt_fp4",
            moe_a2a_backend="none",
            ep_size=4,
            moe_dp_size=1,
            lora_paths=None,
            enable_lora=False,
            enable_flashinfer_moe_autotune=False,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def _validate(self, **overrides):
        from sglang.srt.arg_groups.moe_hook import _validate_flashinfer_b12x_w4a16

        cfg = SimpleNamespace(tp_size=4)
        return _validate_flashinfer_b12x_w4a16(self._view(**overrides), cfg)

    @requires_sm12x_b12x
    def test_a_valid_configuration_passes_when_the_gate_passes(self):
        self._patch(_FC + "flashinfer_b12x_w4a16_unavailable_reason", None)
        self.assertIsNone(self._validate())

    def test_wrong_quantization(self):
        for quantization in ("fp8", "modelopt_fp8", None):
            with self.subTest(quantization=quantization):
                with self.assertRaises(AssertionError) as ctx:
                    self._validate(quantization=quantization)
                self.assertIn("NVFP4", str(ctx.exception))

    def test_requires_a2a_none(self):
        for a2a in ("deepep", "flashinfer", "mori"):
            with self.subTest(a2a=a2a):
                with self.assertRaises(AssertionError) as ctx:
                    self._validate(moe_a2a_backend=a2a)
                self.assertIn("--moe-a2a-backend none", str(ctx.exception))

    def test_ep_size_must_be_one_or_tp(self):
        with self.assertRaises(AssertionError):
            self._validate(ep_size=2)
        self._patch(_FC + "flashinfer_b12x_w4a16_unavailable_reason", None)
        self.assertIsNone(self._validate(ep_size=1))

    def test_expert_data_parallelism_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self._validate(moe_dp_size=2)
        self.assertIn("data parallelism", str(ctx.exception))

    def test_lora_is_rejected(self):
        for overrides in ({"lora_paths": ["a"]}, {"enable_lora": True}):
            with self.subTest(**overrides):
                with self.assertRaises(ValueError) as ctx:
                    self._validate(**overrides)
                self.assertIn("flashinfer_cutlass", str(ctx.exception))

    def test_autotune_is_warned_about_not_rejected(self):
        self._patch(_FC + "flashinfer_b12x_w4a16_unavailable_reason", None)
        with self.assertLogs("sglang.srt.arg_groups.moe_hook", level="WARNING") as logs:
            self._validate(enable_flashinfer_moe_autotune=True)
        self.assertTrue(any("b12x W4A16 route" in line for line in logs.output))

    def test_config_errors_are_reported_before_the_hardware_gate(self):
        """A bad flag combination must not be blamed on the toolkit."""
        with self.assertRaises(AssertionError) as ctx:
            self._validate(moe_a2a_backend="deepep")
        self.assertIn("--moe-a2a-backend none", str(ctx.exception))

    def test_gate_refusal_names_the_backend_and_stops_the_boot(self):
        self._patch(
            _FC + "flashinfer_b12x_w4a16_unavailable_reason",
            "requires CUDA 13 or later",
        )
        with self.assertRaises(ValueError) as ctx:
            self._validate()
        message = str(ctx.exception)
        self.assertIn("--moe-runner-backend flashinfer_b12x_w4a16", message)
        self.assertIn("explicit opt-in", message)

    def test_fusion_is_disabled_for_this_runner(self):
        """The packed FP4 weights cannot carry a fused shared expert."""
        from sglang.srt.arg_groups.overrides import _moe_runner_fusion_disable

        self.assertEqual(
            _moe_runner_fusion_disable(
                SimpleNamespace(moe_runner_backend="flashinfer_b12x_w4a16")
            ),
            {"disable_shared_experts_fusion": True},
        )
        self.assertEqual(
            _moe_runner_fusion_disable(SimpleNamespace(moe_runner_backend="triton")), {}
        )

    @requires_sm12x_b12x
    def test_auto_never_resolves_to_b12x(self):
        """SM120 + NVFP4 keeps resolving to the existing CUTLASS route."""
        from sglang.srt.arg_groups.overrides import (
            _moe_runner_backend_quant_constraints,
        )

        view = SimpleNamespace(
            moe_runner_backend="auto",
            quantization="modelopt_fp4",
            moe_a2a_backend="none",
        )
        platform = SimpleNamespace(
            is_sm100=False, is_npu=False, is_hip=False, is_sm120=True, is_sm110=False
        )
        with patch(
            "sglang.srt.arg_groups.overrides.get_platform", return_value=platform
        ):
            result = _moe_runner_backend_quant_constraints(view)
        self.assertEqual(
            result,
            {"moe_runner_backend": MoeRunnerBackend.FLASHINFER_CUTLASS.value},
        )
        # An explicit b12x request is passed through untouched (no auto route
        # ever lands on it, and naming it is what the validation gate checks).
        self.assertEqual(
            _moe_runner_backend_quant_constraints(
                SimpleNamespace(
                    moe_runner_backend="flashinfer_b12x_w4a16",
                    quantization="modelopt_fp4",
                    moe_a2a_backend="none",
                )
            ),
            {},
        )


class TestExpertMap(unittest.TestCase):
    def test_no_ep_passes_no_map(self):
        self.assertIsNone(
            _b12x_w4a16_expert_map(
                num_experts=8, num_local_experts=8, ep_rank=0, ep_size=1, device="cpu"
            )
        )

    def test_prod_geometry(self):
        """512 experts at EP4 -> 128 local ids, matching the live TP4/EP4 launch."""
        for rank in range(4):
            expert_map = _b12x_w4a16_expert_map(
                num_experts=512,
                num_local_experts=128,
                ep_rank=rank,
                ep_size=4,
                device="cpu",
            )
            with self.subTest(rank=rank):
                self.assertEqual(expert_map.dtype, torch.int32)
                self.assertEqual(expert_map.numel(), 512)
                self.assertTrue(expert_map.is_contiguous())
                self.assertTrue(
                    torch.equal(
                        expert_map[rank * 128 : (rank + 1) * 128],
                        torch.arange(128, dtype=torch.int32),
                    )
                )
                rest = torch.cat(
                    [expert_map[: rank * 128], expert_map[(rank + 1) * 128 :]]
                )
                self.assertTrue(torch.equal(rest, torch.full_like(rest, -1)))
                # Every local id used exactly once, i.e. what flashinfer's
                # prepare_ep_expert_map validates for the sibling package.
                self.assertEqual(
                    sorted(int(i) for i in expert_map if i >= 0), list(range(128))
                )

    def test_indivisible_split_is_asserted(self):
        """A clipped slice would silently make trailing experts unroutable."""
        with self.assertRaises(AssertionError):
            _b12x_w4a16_expert_map(
                num_experts=512,
                num_local_experts=100,
                ep_rank=0,
                ep_size=4,
                device="cpu",
            )


class TestPrewarmCapacities(_PatchHelper):
    def test_ladder_covers_every_route_specialization(self):
        self.assertEqual(b12x_w4a16_token_capacities(1), [1])
        self.assertEqual(b12x_w4a16_token_capacities(8), [1, 2, 4, 8])
        self.assertEqual(b12x_w4a16_token_capacities(0), [1])
        caps = b12x_w4a16_token_capacities(4096)
        self.assertEqual(caps[:4], [1, 2, 4, 8])
        self.assertEqual(caps[-1], 4096)
        self.assertEqual(len(set(caps)), len(caps))
        # A non-power-of-two max warms its own bucket on top of the ladder.
        self.assertEqual(b12x_w4a16_token_capacities(300)[-2:], [256, 300])

    def test_warm_capacity_is_the_largest_captured_graph(self):
        args = SimpleNamespace(
            speculative_algorithm="NEXTN",
            speculative_eagle_topk=1,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(max_bs=48)),
            cuda_graph_max_bs=48,
            max_running_requests=96,
        )
        with patch.object(fic, "get_server_args", return_value=args):
            self.assertEqual(fic._b12x_warm_token_capacity(), 48 * 4)
        with patch.object(fic, "get_server_args", return_value=None):
            self.assertEqual(fic._b12x_warm_token_capacity(), 0)

    def test_a_wider_draft_tree_widens_the_ladder(self):
        args = SimpleNamespace(
            speculative_algorithm="EAGLE",
            speculative_eagle_topk=4,
            speculative_num_steps=2,
            speculative_num_draft_tokens=None,
            cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(max_bs=8)),
            cuda_graph_max_bs=8,
            max_running_requests=None,
        )
        with patch.object(fic, "get_server_args", return_value=args):
            self.assertEqual(fic._b12x_warm_token_capacity(), 8 * (1 + 2 * 4))

    def _model_with_prepared_layer(self, payload, runner_config):
        model = torch.nn.Module()
        layer = torch.nn.Module()
        layer.quant_method = Mock()
        layer.quant_method.b12x_w4a16_warm_payload = Mock(
            return_value=(payload, runner_config)
        )
        model.add_module("layer0", layer)
        return model

    def _payload(self):
        return FlashInferCutlassMoeQuantInfo(
            quant_type="w4a16_b12x",
            w13_weight=torch.zeros(4, 16, 4, dtype=torch.uint8),
            w2_weight=torch.zeros(4, 8, 4, dtype=torch.uint8),
            output_dtype=torch.bfloat16,
            quant_scales=[Mock(), Mock(), Mock(), Mock()],
            prepared_weights=SimpleNamespace(tag="prepared"),
        )

    def test_prewarm_is_inert_when_the_route_was_not_selected(self):
        """A normal boot must not pay an nvcc probe or a flashinfer import."""
        self._patch(_FC + "get_moe_runner_backend", MoeRunnerBackend.FLASHINFER_CUTLASS)
        with (
            patch.object(fic, "precompile_flashinfer_b12x_w4a16_routes") as pre,
            patch.object(fic, "flashinfer_b12x_w4a16_unavailable_reason") as gate,
        ):
            self.assertEqual(prewarm_flashinfer_b12x_w4a16(Mock()), 0)
            pre.assert_not_called()
            gate.assert_not_called()

    def test_prewarm_stops_when_the_gate_refuses(self):
        self._patch(
            _FC + "get_moe_runner_backend", MoeRunnerBackend.FLASHINFER_B12X_W4A16
        )
        self._patch(_FC + "flashinfer_b12x_w4a16_unavailable_reason", "no CUDA 13")
        with patch.object(fic, "precompile_flashinfer_b12x_w4a16_routes") as pre:
            self.assertEqual(prewarm_flashinfer_b12x_w4a16(Mock()), 0)
            pre.assert_not_called()

    def test_prewarm_drives_the_real_launcher_per_capacity(self):
        self._patch(
            _FC + "get_moe_runner_backend", MoeRunnerBackend.FLASHINFER_B12X_W4A16
        )
        self._patch(_FC + "flashinfer_b12x_w4a16_unavailable_reason", None)
        self._patch(_FC + "_b12x_warm_token_capacity", 8)
        self._patch_new(_FC + "use_symmetric_memory", lambda *a, **k: nullcontext())
        self._patch(_FC + "is_allocation_symmetric", False)
        self._patch(_FC + "get_parallel", Mock())
        calls = []
        self._patch(_FC + "b12x_w4a16_launch", _capture_launch(calls))
        model = self._model_with_prepared_layer(self._payload(), _b12x_runner_config())
        warmed = prewarm_flashinfer_b12x_w4a16(model)
        # 8 tokens -> ladder [1, 2, 4, 8], each warmed once, largest first.
        self.assertEqual(warmed, 4)
        self.assertEqual([c[1]["a"].shape[0] for c in calls], [8, 4, 2, 1])
        self.assertTrue(all(c[1]["quant_mode"] == "w4a16" for c in calls))

    def test_prewarm_survives_a_launch_failure(self):
        """A prewarm must never fail a boot: the first error stops the ladder."""
        payload = self._payload()
        with patch.object(
            fic,
            "_run_flashinfer_b12x_w4a16",
            side_effect=RuntimeError("W4A16 workspace is not initialized"),
        ) as run:
            warmed = fic.precompile_flashinfer_b12x_w4a16_routes(
                payload=payload,
                runner_config=_b12x_runner_config(),
                token_capacities=[4, 2, 1],
            )
        self.assertEqual(warmed, 0)
        self.assertEqual(run.call_count, 1)  # stopped at the first failure

    def test_precompile_skips_an_unprepared_payload(self):
        payload = self._payload()
        payload.prepared_weights = None
        self.assertEqual(
            fic.precompile_flashinfer_b12x_w4a16_routes(
                payload=payload,
                runner_config=_b12x_runner_config(),
                token_capacities=[4],
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()
