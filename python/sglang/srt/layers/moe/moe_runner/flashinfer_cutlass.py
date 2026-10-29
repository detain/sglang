"""FlashInfer CUTLASS MoE fused funcs.

This module owns the FlashInfer ``cutlass_fused_moe`` calls used by the
unquantized, ModelOpt FP8, ModelOpt NVFP4, and CUTLASS MXFP4 MoE paths, plus
the shared ``flashinfer_mxfp4`` dispatcher. Quantization methods prepare a
small quant_info payload and route through ``MoeRunner``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.kernels.ops.quantization.fp8_kernel import scaled_fp8_quant
from sglang.kernels.selector import get_kernel
from sglang.kernels.spec import KernelBackend
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    deployment_attn_dp_size,
    is_allocation_symmetric,
)
from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    register_fused_func,
)
from sglang.srt.layers.moe.utils import (
    get_moe_a2a_backend,
    get_moe_runner_backend,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
    _is_stream_capturing,
)
from sglang.srt.runtime_context import (
    get_disagg,
    get_parallel,
    get_schedule,
    get_server_args,
    max_prefill_buffer_tokens,
)
from sglang.srt.utils import is_flashinfer_available
from sglang.srt.utils.common import is_sm120, next_power_of_2

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.kernels.ops.moe.nvfp4_moe_sm120 import Nvfp4MoeWorkspace
    from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
        FlashinferCombineInput,
        FlashinferDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


@dataclass
class FlashInferCutlassMoeQuantInfo(MoeQuantInfo):
    """Payload for FlashInfer CUTLASS fused MoE.

    ``quant_type`` selects the input/weight conventions:
      - ``"bf16"``: unquantized weights, BF16/FP16 input, no quant scales.
      - ``"fp8"``: FP8 weights, FP8-quantized input, per-tensor scales.
      - ``"fp4"``: NVFP4 packed weights and optional NVFP4 packed input.
      - ``"w4a16_b12x"``: SM120 b12x route -- packed NVFP4 weights with the
        E4M3 K/16 scale grid and BF16 activations (nothing is quantized on the
        activation side). ``quant_scales`` carries the four *prepared* tensors
        (block scales + weight global scales) and ``prepared_weights`` carries
        the object flashinfer's preparer returned at load time, which is what the
        launch actually consumes.
    """

    quant_type: str
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    quant_scales: Optional[list[torch.Tensor]] = None
    output_dtype: Optional[torch.dtype] = None
    # Optional per-expert SwiGLU overrides, fp32 [num_local_experts].
    swiglu_alpha: Optional[torch.Tensor] = None
    swiglu_beta: Optional[torch.Tensor] = None
    swiglu_limit: Optional[torch.Tensor] = None
    apply_routed_scaling_factor: bool = True
    # #44994 paired-NVFP4 FC2 weight: legal only on the sm120 native MTP-MMA
    # route below; any other path that receives one fails loud.
    pair_w2_weight: Optional[torch.Tensor] = None
    g1_alpha_up: Optional[torch.Tensor] = None
    smallm_workspace: Optional[Nvfp4MoeWorkspace] = None
    smallm_global_routed_experts: Optional[int] = None
    smallm_local_routed_experts: Optional[int] = None
    smallm_local_expert_start: Optional[int] = None
    # flashinfer W4A16PackedWeights, built once per layer at weight-load time.
    # The b12x launch ignores the weight/scale/alpha arguments when this is set:
    # the prepared object already embeds the repacked weights, the permuted
    # scales and the folded weight-only global scale.
    prepared_weights: Optional[Any] = None


B12X_W4A16_MIN_CUDA_MAJOR = 13
# The route's contract, both derived from the installed flashinfer 0.7.2 wheel:
# ``source_format`` is a key of blackwell_sm12x/moe_w4a16_prepare.py
# ``_SOURCE_FORMATS`` -- "modelopt" is the plain ModelOpt NVFP4 export (E2M1
# weights with the normal K/16 E4M3 scale grid in swizzled storage). The
# "w4a16" preparer never folds activation input scales, so this is the correct
# spelling for a ModelOpt A4 checkpoint whose activations stay BF16.
B12X_W4A16_SOURCE_FORMAT = "modelopt"
# ``w13_layout`` names the ORDER THE ROWS ARRIVE IN, not the kernel's internal
# order: "w13" == [Up, Gate] (the preparer rotates the halves), "w31" ==
# [Gate, Up]. sglang only stores [Up, Gate] when
# ``ModelOptNvFp4FusedMoEMethod.load_up_proj_weight_first`` is true, which is
# keyed on the flashinfer_cutlass / cutedsl-v2 runners -- so this route, which
# is neither, gets the default [Gate, Up] and must say "w31". Passing "w13"
# would rotate a gate-first tensor and swap gate/up inside SwiGLU.
B12X_W4A16_W13_LAYOUT = "w31"
B12X_W4A16_W13_LAYOUT_UP_FIRST = "w13"


def b12x_w4a16_launch():
    """Import ``flashinfer.fused_moe..launch_sm120_moe``, or raise a named error."""
    try:
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
            launch_sm120_moe,
        )
    except Exception as error:  # ImportError, or a wheel built without SM12x
        raise RuntimeError(
            "FlashInfer b12x W4A16 MoE needs "
            "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch."
            "launch_sm120_moe, which the installed flashinfer does not provide "
            f"({type(error).__name__}: {error}). Install flashinfer >= 0.7.2 "
            "with SM12x b12x MoE support."
        ) from error
    return launch_sm120_moe


def _b12x_w4a16_cuda_major() -> Optional[int]:
    """CUDA major version flashinfer's JIT gate will see, or None if unknown.

    Deliberately asks flashinfer rather than ``torch.version.cuda``: the wheel's
    b12x gate calls ``flashinfer.jit.cpp_ext.get_cuda_version()``, which shells
    out to ``$CUDA_HOME/bin/nvcc`` and only falls back to torch's build version
    when nvcc is missing. A torch cu130 runtime with a CUDA 12.9 toolkit on
    ``CUDA_HOME`` therefore fails the wheel's gate, and this has to agree with
    it or the check would be fiction.
    """
    try:
        from flashinfer.jit.cpp_ext import get_cuda_version

        return int(get_cuda_version().major)
    except Exception:
        return None


def flashinfer_b12x_w4a16_unavailable_reason() -> Optional[str]:
    """Why the b12x W4A16 MoE route cannot run here, or ``None`` when it can.

    Checked at backend selection (server-args resolution and again in
    ``MoeRunner``) so an unsupported configuration fails with the fix in the
    message instead of deep inside a kernel launch or, worse, a CUDA graph
    capture. Never allocates device memory.
    """
    if not torch.cuda.is_available():
        return "requires a CUDA device (no GPU is visible to this process)"
    try:
        capability = torch.cuda.get_device_capability()
    except Exception as error:  # pragma: no cover - driver query failure
        return f"could not read the CUDA device capability ({error})"
    if capability != (12, 0):
        return (
            "is implemented only for SM120 (compute capability 12.0, e.g. RTX "
            f"PRO 6000 Blackwell); this device is sm{capability[0]}{capability[1]}"
        )
    try:
        b12x_w4a16_launch()
    except RuntimeError as error:
        return f"cannot be imported: {error}"
    major = _b12x_w4a16_cuda_major()
    if major is not None and major < B12X_W4A16_MIN_CUDA_MAJOR:
        return (
            f"requires CUDA {B12X_W4A16_MIN_CUDA_MAJOR} or later, but flashinfer's "
            f"JIT gate sees CUDA {major}. flashinfer gates every b12x/SM12x "
            "kernel on the version reported by `$CUDA_HOME/bin/nvcc`, not on the "
            "CUDA the torch build links, so a cu130 torch with a CUDA 12.9 "
            "toolkit prefix still fails. Fix: export `CUDA_HOME` (and put "
            "`$CUDA_HOME/bin` first on PATH) to a CUDA "
            f"{B12X_W4A16_MIN_CUDA_MAJOR}.x toolkit before launching -- expect a "
            "one-time flashinfer re-JIT for the new toolkit (~100 MB of cached "
            "objects, several minutes) and rebuild any flashinfer autotune cache "
            "captured under the old one. Or use "
            "--moe-runner-backend flashinfer_cutlass."
        )
    return None


class _B12xWarmTopkOutput:
    """The two fields of a ``TopKOutput`` the b12x warm launches read."""

    __slots__ = ("topk_ids", "topk_weights")

    def __init__(self, *, topk_ids: torch.Tensor, topk_weights: torch.Tensor):
        self.topk_ids = topk_ids
        self.topk_weights = topk_weights


def b12x_w4a16_token_capacities(max_tokens: int) -> list[int]:
    """Power-of-two token ladder up to ``max_tokens``, plus ``max_tokens`` itself.

    The packed route table is built for ``route_pack_token_capacity(tokens)``,
    which rounds the token count up to the next power of two, so this ladder
    enumerates the distinct specializations the route can be asked for.
    """
    max_tokens = max(1, int(max_tokens))
    capacities = []
    capacity = 1
    while capacity < max_tokens:
        capacities.append(capacity)
        capacity *= 2
    capacities.append(max_tokens)
    return capacities


def _b12x_warm_token_capacity() -> int:
    """Largest MoE token count a captured decode/verify CUDA graph can carry."""
    server_args = get_server_args()
    if server_args is None:  # pragma: no cover - only outside a server
        return 0

    tokens_per_request = 1
    if server_args.speculative_algorithm:
        # A verify batch carries one drafted tree per request: `num_draft_tokens`
        # rows for a chain (top_k 1), 1 + num_steps * top_k for a wider tree.
        top_k = max(1, int(server_args.speculative_eagle_topk or 1))
        steps = int(server_args.speculative_num_steps or 0)
        draft = int(server_args.speculative_num_draft_tokens or 0)
        tokens_per_request = max(1, draft, 1 + steps * top_k)

    configs = getattr(server_args, "cuda_graph_config", None)
    decode_bs = getattr(getattr(configs, "decode", None), "max_bs", None)
    if not decode_bs:
        decode_bs = server_args.cuda_graph_max_bs or 0
    decode_bs = int(decode_bs)
    if server_args.max_running_requests:
        decode_bs = (
            min(decode_bs, int(server_args.max_running_requests))
            if decode_bs
            else int(server_args.max_running_requests)
        )
    return decode_bs * tokens_per_request


@torch.inference_mode()
def precompile_flashinfer_b12x_w4a16_routes(
    *,
    payload: FlashInferCutlassMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    token_capacities: list[int],
) -> int:
    """Launch the b12x W4A16 route once per token capacity, largest first.

    The route has two lazy, shape-specialized steps that a CUDA graph capture
    must never meet for the first time:

    1. the packed route table and its kernels, which specialize on the
       power-of-two token capacity and on the routed block size that capacity
       selects (``select_route_block_size_m``), and
    2. the W4A16 workspace, whose cache raises ``W4A16 workspace is not
       initialized for CUDA graph capture`` if it has to allocate while a graph
       is being captured.

    The workspace cache keys on geometry only (experts, hidden, intermediate,
    top-k, device) and keeps the largest capacity it has seen, reusing it for
    every smaller call, so walking the ladder in descending order means one
    allocation covers all of it.

    Each warm runs the real fused func on zero activations, so it exercises the
    same expert map, output allocation and launch arguments as serving. Returns
    the number of capacities warmed; the first failure stops the ladder and is
    logged rather than raised -- a prewarm must not fail a boot, and the launch
    that needs the shape reports its own error.
    """
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput

    # Built positionally: StandardDispatchOutput is a NamedTuple whose
    # hidden_states_scale field is required (it has no default), so a
    # keyword-only construction of the two fields the route reads would
    # TypeError -- caught here rather than in a log line nobody reads.
    if payload.prepared_weights is None or not runner_config.top_k:
        return 0
    hidden_size = int(payload.w13_weight.shape[2]) * 2
    device = payload.w13_weight.device
    top_k = int(runner_config.top_k)
    warmed = 0
    for tokens in sorted(
        {int(t) for t in token_capacities if int(t) > 0}, reverse=True
    ):
        try:
            _run_flashinfer_b12x_w4a16(
                dispatch_output=StandardDispatchOutput(
                    torch.zeros(
                        tokens, hidden_size, dtype=torch.bfloat16, device=device
                    ),
                    None,
                    _B12xWarmTopkOutput(
                        topk_ids=torch.zeros(
                            (tokens, top_k), dtype=torch.int64, device=device
                        ),
                        topk_weights=torch.zeros(
                            (tokens, top_k), dtype=torch.float32, device=device
                        ),
                    ),
                ),
                quant_info=payload,
                runner_config=runner_config,
                output=None,
                enable_alltoall=False,
            )
        except Exception as error:
            detail = str(error).strip().splitlines()
            logger.warning(
                "b12x W4A16 prewarm stopped at %d tokens (%s: %s)",
                tokens,
                type(error).__name__,
                detail[0] if detail else "no detail",
            )
            break
        warmed += 1
    if warmed:
        logger.info(
            "b12x W4A16 prewarm: warmed %d token capacities (hidden %d, top_k %d, "
            "%d local experts, ep_size %d) on %s",
            warmed,
            hidden_size,
            top_k,
            int(payload.w13_weight.shape[0]),
            runner_config.moe_ep_size,
            device,
        )
    return warmed


def prewarm_flashinfer_b12x_w4a16(model: torch.nn.Module) -> int:
    """Model-hook entry point: warm the b12x route for the captured shapes.

    Called from a model's ``precompile_kernels_after_loading``, i.e. after the
    weights (and so the one-time in-place preparation) are in place and before
    the KV pools are committed and CUDA graphs captured -- the only window with
    free device memory to allocate the workspace in. Every MoE layer shares one
    geometry, so warming through the first prepared layer fills the cache for
    all of them.

    Prefill is not capture-constrained, so its workspace is left to allocate
    lazily on the first large chunk unless the operator asks for it up front with
    ``SGLANG_B12X_W4A16_PREWARM_PREFILL_TOKENS``: the intermediate caches scale as
    tokens * top_k * intermediate_size, so a 16K-token warm costs hundreds of MB
    per rank and is never released.
    """
    # Cheap first, and before the gate: the reason check imports flashinfer
    # and shells out to nvcc, so it must not run on a boot that never selected
    # this backend.
    if not get_moe_runner_backend().is_flashinfer_b12x_w4a16():
        return 0
    if flashinfer_b12x_w4a16_unavailable_reason() is not None:
        return 0

    max_tokens = _b12x_warm_token_capacity()
    prefill_tokens = int(envs.SGLANG_B12X_W4A16_PREWARM_PREFILL_TOKENS.get())
    if prefill_tokens > 0:
        max_tokens = max(max_tokens, prefill_tokens)
    if max_tokens <= 0:
        return 0

    for module in model.modules():
        build_payload = getattr(
            getattr(module, "quant_method", None), "b12x_w4a16_warm_payload", None
        )
        if build_payload is None:
            continue
        payload, runner_config = build_payload(module)
        if payload is None:
            continue
        return precompile_flashinfer_b12x_w4a16_routes(
            payload=payload,
            runner_config=runner_config,
            token_capacities=b12x_w4a16_token_capacities(max_tokens),
        )
    return 0


def materialize_swiglu_params_for_cutlass(
    runner_config: MoeRunnerConfig,
    num_local_experts: int,
    device: torch.device,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Per-expert SwiGLU (alpha, beta, limit) tensors for the CUTLASS kernel.

    Returns all-None unless a clamp limit is configured. ``gemm1_alpha``
    implies the GPT-OSS-style ``+1`` up term, so beta then defaults to 1.0;
    without alpha, alpha/beta stay silu-neutral at 1.0 / 0.0. The kernel
    clamps dequantized values, so the limit is in physical units (no
    g1_alphas conversion, unlike trtllm-gen). SiTU carries its clamp in the
    activation itself.
    """
    if runner_config.activation == "situ":
        return None, None, None
    clamp_limit = runner_config.gemm1_clamp_limit or runner_config.swiglu_limit
    if clamp_limit is None:
        return None, None, None
    alpha = runner_config.gemm1_alpha if runner_config.gemm1_alpha is not None else 1.0
    beta = runner_config.gemm1_beta
    if beta is None:
        beta = 1.0 if runner_config.gemm1_alpha is not None else 0.0
    return (
        torch.full(
            (num_local_experts,), float(alpha), dtype=torch.float32, device=device
        ),
        torch.full(
            (num_local_experts,), float(beta), dtype=torch.float32, device=device
        ),
        torch.full(
            (num_local_experts,),
            float(clamp_limit),
            dtype=torch.float32,
            device=device,
        ),
    )


@dataclass
class FlashInferCutlassMxfp4MoeQuantInfo(MoeQuantInfo):
    """Quantization payload for CUTLASS MXFP4 MoE.

    SM90 consumes either W4A16-interleaved weights/scales or the Humming-style
    W4A8 layouts. SM120 consumes packed MXFP4 weights and block-interleaved
    scales with MXFP8 activations.
    """

    # SM90 weights are interleaved; SM120 weights remain checkpoint-packed.
    w13_weight: torch.Tensor  # [E, 2*N, K/2]
    w2_weight: torch.Tensor  # [E, K, N/2]

    # E8M0 block scales in the layout selected by the quantization method.
    w13_weight_scale: torch.Tensor  # [E, 2*N, K/32]
    w2_weight_scale: torch.Tensor  # [E, K, N/32]

    # A non-None global scale selects the SM120 MXFP8 activation path.
    mxfp4_weight_global_scale: Optional[torch.Tensor] = None

    # A complete non-None triplet selects the SM90 Humming W4A8 path. The
    # residuals are FP32 [num_local_experts] and already include the fixed 2^6
    # compensation required by FlashInfer's epilogue.
    w13_humming_residual_scale: Optional[torch.Tensor] = None
    w2_humming_residual_scale: Optional[torch.Tensor] = None
    humming_fc2_act_scale: Optional[torch.Tensor] = None

    # Per-expert bias. GPT-OSS has both; DSv4 leaves both None.
    w13_bias: Optional[torch.Tensor] = None  # bf16 [E, 2*N]
    w2_bias: Optional[torch.Tensor] = None  # bf16 [E, K]

    # Optional per-expert SwiGLU overrides, fp32 [E].
    swiglu_alpha: Optional[torch.Tensor] = None
    swiglu_beta: Optional[torch.Tensor] = None
    swiglu_limit: Optional[torch.Tensor] = None

    # Bailing clamps after SiLU, which the kernel only implements in its
    # SwigluStep variant.
    use_swiglu_step: bool = False

    # GPT-OSS pads its input hidden dim up to the (pre-padded) loaded weight
    # width and trims the output back. DSv4 leaves this as ``None`` (no pad).
    padded_hidden: Optional[int] = None


def _flashinfer_cutlass_fused_moe():
    if not is_flashinfer_available():
        raise RuntimeError(
            "flashinfer_cutlass MoE runner backend requires flashinfer to be installed."
        )
    from flashinfer.fused_moe import cutlass_fused_moe
    from flashinfer.fused_moe.core import ActivationType

    return cutlass_fused_moe, ActivationType


# Matches the default tune_max_num_tokens of FlashInfer's cutlass_fused_moe.
_FLASHINFER_DEFAULT_TUNE_MAX_NUM_TOKENS = 8192


# Tune range: the DP-gathered prefill chunk, capped at FlashInfer's default;
# PD decode servers run no prefill, so they size it per call.
def _cutlass_moe_tune_max_num_tokens(num_tokens: int) -> int:
    if get_disagg().disaggregation_mode == "decode":
        return next_power_of_2(num_tokens)
    prefill_tokens = max_prefill_buffer_tokens() or get_schedule().max_prefill_tokens
    return min(
        prefill_tokens * deployment_attn_dp_size(),
        _FLASHINFER_DEFAULT_TUNE_MAX_NUM_TOKENS,
    )


def _activation_type(runner_config: MoeRunnerConfig):
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import get_activation_type

    _, ActivationType = _flashinfer_cutlass_fused_moe()
    activation = ActivationType(
        get_activation_type(
            runner_config.activation,
            is_gated=runner_config.is_gated,
        )
    )
    supported = {
        ActivationType.Swiglu,
        ActivationType.Geglu,
        ActivationType.Relu2,
        ActivationType.Identity,
    }
    assert activation in supported, (
        f"Activation {runner_config.activation!r} "
        f"(is_gated={runner_config.is_gated}) maps to {activation.name}, "
        "which is not supported by flashinfer cutlass moe."
    )
    return activation


def _maybe_apply_routed_scaling_factor(
    output: torch.Tensor,
    quant_info: FlashInferCutlassMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> torch.Tensor:
    if (
        quant_info.apply_routed_scaling_factor
        and runner_config.routed_scaling_factor is not None
    ):
        output.mul_(runner_config.routed_scaling_factor)
    return output


def _prescale_router_weight_on_input(
    dispatch_output: StandardDispatchOutput | FlashinferDispatchOutput,
    runner_config: MoeRunnerConfig,
) -> StandardDispatchOutput | FlashinferDispatchOutput:
    if not runner_config.apply_router_weight_on_input:
        return dispatch_output

    topk_output = dispatch_output.topk_output
    topk_weights = topk_output.topk_weights

    if dispatch_output.hidden_states_scale is not None:
        raise NotImplementedError(
            "apply_router_weight_on_input is not supported when activations are "
            "quantized before dispatch (flashinfer_cutlass fp4 all-gather path)."
        )

    assert topk_weights.dim() == 2 and topk_weights.shape[-1] == 1, (
        "apply_router_weight_on_input requires topk=1"
    )

    hidden_states = dispatch_output.hidden_states * topk_weights.to(
        dispatch_output.hidden_states.dtype
    )
    unit_scales = torch.ones_like(topk_weights, dtype=torch.float32)
    return dispatch_output._replace(
        hidden_states=hidden_states,
        topk_output=topk_output._replace(topk_weights=unit_scales),
    )


def _prepare_input(
    dispatch_output,
    quant_info: FlashInferCutlassMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.dtype, int]:
    x = dispatch_output.hidden_states
    x_sf = dispatch_output.hidden_states_scale

    if quant_info.quant_type == "fp8":
        assert quant_info.quant_scales is not None and len(quant_info.quant_scales) == 4
        x, _ = scaled_fp8_quant(x, quant_info.quant_scales[3])
        x_sf = None
        output_dtype = quant_info.output_dtype or dispatch_output.hidden_states.dtype
        output_col = dispatch_output.hidden_states.shape[1]
    elif quant_info.quant_type == "fp4":
        output_dtype = quant_info.output_dtype or torch.bfloat16
        output_col = x.shape[1]
        if x_sf is not None and runner_config.is_gated:
            output_col *= 2
    else:
        assert quant_info.quant_type == "bf16"
        output_dtype = quant_info.output_dtype or x.dtype
        output_col = x.shape[1]

    return x, x_sf, output_dtype, output_col


_logged_smallm_decisions: set[str] = set()


def _log_smallm_decision(message: str) -> None:
    if message in _logged_smallm_decisions:
        return
    _logged_smallm_decisions.add(message)
    logger.info("SM120 NVFP4 small-row MoE: %s", message)


def _warn_smallm_unexpected(message: str) -> None:
    """Report an internal small-row MoE fault once per distinct text.

    Choosing between the small-row kernel and CUTLASS is an optimisation, so a
    raise inside it must never fail a forward -- but it is also a bug, so it
    earns a warning rather than silence. Shares the decision log's dedupe set:
    this runs per layer per forward, and one report is enough.
    """
    if message in _logged_smallm_decisions:
        return
    _logged_smallm_decisions.add(message)
    logger.warning("SM120 NVFP4 small-row MoE: %s", message)


def _smallm_ineligibility_reason(
    *,
    quant_info: FlashInferCutlassMoeQuantInfo,
    workspace: Optional[Nvfp4MoeWorkspace],
    x: torch.Tensor,
    x_sf: Optional[torch.Tensor],
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    output_dtype: torch.dtype,
    quant_scales: Optional[list[torch.Tensor]],
    runner_config: MoeRunnerConfig,
    output_supplied: bool,
    enable_alltoall: bool,
    capturing: Optional[bool] = None,
) -> Optional[str]:
    """Why the small-row kernel cannot serve this call, or None if it can.

    The MoE placement is read from ``runner_config`` -- a view of the layer's
    own ``moe_{tp,ep}_{size,rank}`` -- not from ``quant_info``: upstream #42348
    moved every placement consumer there, and the quant info carries weights
    plus the global/local expert topology only.
    """
    if quant_info.quant_type != "fp4":
        return "quantization is not NVFP4"
    if workspace is None:
        return "workspace is disabled or the layer shape is unsupported"
    if quant_info.g1_alpha_up is None:
        return "the up-projection alpha is unavailable"
    if any(
        value is None
        for value in (
            quant_info.smallm_global_routed_experts,
            quant_info.smallm_local_routed_experts,
            quant_info.smallm_local_expert_start,
        )
    ):
        return "expert topology is unavailable"
    if output_supplied or enable_alltoall:
        return "the all-to-all output contract requires CUTLASS"
    if x_sf is not None:
        return "the input is already quantized"
    if capturing is None:
        capturing = torch.cuda.is_available() and _is_stream_capturing(
            torch.cuda.current_stream(x.device)
        )
    if capturing and workspace.graph_capture_supported is not True:
        return "cooperative graph capture is unavailable"
    if not 0 < x.shape[0] <= workspace.max_tokens:
        return "token count is outside the small-row range"
    if (
        x.dtype != torch.bfloat16
        or topk_ids.dtype != torch.int32
        or topk_weights.dtype != torch.float32
    ):
        return "input or routing dtypes are unsupported"
    if not (
        x.is_contiguous() and topk_ids.is_contiguous() and topk_weights.is_contiguous()
    ):
        return "input or routing tensors are not contiguous"
    if (
        x.dim() != 2
        or topk_ids.dim() != 2
        or topk_weights.shape != topk_ids.shape
        or topk_ids.shape[0] != x.shape[0]
        or x.shape[1] != workspace.hidden_size
        or topk_ids.shape[1] != workspace.top_k
    ):
        return "input or routing shapes do not match the workspace"
    if not runner_config.is_gated or runner_config.activation not in (
        "silu",
        "swiglu",
    ):
        return "the activation is unsupported"
    if any(
        value is not None
        for value in (
            runner_config.gemm1_alpha,
            runner_config.gemm1_beta,
            runner_config.gemm1_clamp_limit,
            runner_config.swiglu_limit,
        )
    ):
        return "the activation modifiers are unsupported"
    if output_dtype != torch.bfloat16:
        return "the output dtype is unsupported"
    if quant_scales is None or len(quant_scales) != 6:
        return "the NVFP4 scale set is incomplete"
    if runner_config.moe_tp_size < 1 or runner_config.moe_ep_size < 1:
        return "the MoE parallel topology is invalid"

    global_routed = quant_info.smallm_global_routed_experts
    local_routed = quant_info.smallm_local_routed_experts
    local_start = quant_info.smallm_local_expert_start
    if (
        global_routed is None
        or local_routed is None
        or local_start is None
        or global_routed <= 0
        or local_routed <= 0
        or local_start < 0
        or not 0 <= runner_config.moe_ep_rank < runner_config.moe_ep_size
        or global_routed != local_routed * runner_config.moe_ep_size
        or local_start != runner_config.moe_ep_rank * local_routed
        or local_routed > quant_info.w13_weight.shape[0]
    ):
        return "the expert topology is not a uniform contiguous EP shard"
    return None


def _sm120_mtp_mma_supported() -> bool:
    import os

    # Matches only FlashInfer's default FP4 quantizer with the fused FC2 epilogue.
    return (
        is_sm120()
        and envs.SGLANG_FLASHINFER_MOE_FUSED_FINALIZE.get()
        and not envs.FLASHINFER_NVFP4_4OVER6.get()
        and os.environ.get("FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH", "0") == "0"
        and os.environ.get("TRTLLM_DISABLE_FP4_QUANT_FAST_MATH", "0") == "0"
    )


def _flashinfer_autotune_active() -> bool:
    from flashinfer.autotuner import AutoTuner

    return AutoTuner.get().is_tuning_mode


def _run_flashinfer_cutlass(
    *,
    dispatch_output,
    quant_info: FlashInferCutlassMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    output: Optional[torch.Tensor] = None,
    enable_alltoall: bool = False,
) -> torch.Tensor:
    if quant_info.quant_type == "w4a16_b12x":
        # The b12x route is its own kernel family (not CUTLASS grouped FP4) and
        # reads none of the CUTLASS input preparation, so branch before any of it
        # runs rather than threading a fourth quant_type through below.
        return _run_flashinfer_b12x_w4a16(
            dispatch_output=dispatch_output,
            quant_info=quant_info,
            runner_config=runner_config,
            output=output,
            enable_alltoall=enable_alltoall,
        )

    flashinfer_cutlass_fused_moe, _ = _flashinfer_cutlass_fused_moe()

    dispatch_output = _prescale_router_weight_on_input(dispatch_output, runner_config)
    topk_output = dispatch_output.topk_output
    topk_weights = topk_output.topk_weights.to(torch.float32)
    topk_ids = topk_output.topk_ids
    x, x_sf, output_dtype, output_col = _prepare_input(
        dispatch_output, quant_info, runner_config
    )

    if (
        quant_info.quant_type == "fp4"
        and not runner_config.is_gated
        and runner_config.activation == "relu2"
        and runner_config.moe_ep_size == runner_config.moe_tp_size == 1
        and x_sf is None
        and not enable_alltoall
        and is_sm120()
    ):
        from sglang.kernels.ops.gemm.sm120_nvfp4_moe import try_nvfp4_moe

        result = try_nvfp4_moe(
            x,
            quant_info.w13_weight,
            quant_info.w2_weight,
            quant_info.quant_scales,
            topk_ids,
            topk_weights,
            output_dtype=output_dtype,
            out=output,
            pair_w2_weight=quant_info.pair_w2_weight,
            # Preserve FlashInfer target/draft tactic discovery during autotuning.
            use_mtp=_sm120_mtp_mma_supported() and not _flashinfer_autotune_active(),
        )
        if result is not None:
            return _maybe_apply_routed_scaling_factor(result, quant_info, runner_config)
    if quant_info.pair_w2_weight is not None:
        raise RuntimeError("A paired NVFP4 weight reached an unsupported MoE path")

    workspace = quant_info.smallm_workspace
    quant_scales = quant_info.quant_scales
    capturing = False
    if workspace is not None and quant_info.quant_type == "fp4":
        capturing = torch.cuda.is_available() and _is_stream_capturing(
            torch.cuda.current_stream(x.device)
        )
    smallm_reason = None
    if workspace is not None:
        try:
            smallm_reason = _smallm_ineligibility_reason(
                quant_info=quant_info,
                workspace=workspace,
                x=x,
                x_sf=x_sf,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                output_dtype=output_dtype,
                quant_scales=quant_scales,
                runner_config=runner_config,
                output_supplied=output is not None,
                enable_alltoall=enable_alltoall,
                capturing=capturing,
            )
        except Exception as error:
            # Which kernel may run is an optimisation decision, so a raise here
            # must degrade to CUTLASS rather than escape a fused func and take
            # down a scheduler. It is still a bug: report it once at warning
            # level instead of folding it silently into the fallback reason.
            detail = str(error).strip().splitlines()
            _warn_smallm_unexpected(
                "using CUTLASS because the eligibility check raised "
                f"{type(error).__name__}: {detail[0] if detail else 'no detail'}"
            )
            smallm_reason = "the eligibility check raised (see its warning)"
    use_smallm = workspace is not None and smallm_reason is None
    if not use_smallm and workspace is not None and quant_info.quant_type == "fp4":
        _log_smallm_decision(f"using CUTLASS because {smallm_reason}")

    if output is None:
        with use_symmetric_memory(
            get_parallel().tp_group, disabled=not is_allocation_symmetric()
        ):
            output = torch.empty(
                x.shape[0],
                output_col,
                dtype=output_dtype,
                device=x.device,
            )

    if use_smallm:
        launch_error = None
        try:
            launched = get_kernel("moe.nvfp4_fused_experts", KernelBackend.JIT)(
                x=x,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                w13_weight=quant_info.w13_weight,
                w2_weight=quant_info.w2_weight,
                w13_scale=quant_scales[1],
                w2_scale=quant_scales[4],
                input_scale_1=quant_scales[0],
                input_scale_2=quant_scales[3],
                g1_alpha=quant_scales[2],
                g1_alpha_up=quant_info.g1_alpha_up,
                g2_alpha=quant_scales[5],
                global_routed_experts=quant_info.smallm_global_routed_experts,
                local_routed_experts=quant_info.smallm_local_routed_experts,
                local_expert_start=quant_info.smallm_local_expert_start,
                output=output,
                workspace=workspace,
            )
        except Exception as error:
            if capturing:
                raise RuntimeError(
                    "SM120 NVFP4 cooperative launch failed during CUDA graph capture"
                ) from error
            launched = False
            launch_error = error
            error_text = str(error).strip()
            detail = error_text.splitlines()[0] if error_text else "no detail"
            _log_smallm_decision(
                "using CUTLASS because the SM120 JIT launch failed: "
                f"{type(error).__name__}: {detail}"
            )
        if launched:
            _log_smallm_decision("selected")
            return output
        if launch_error is None:
            if capturing:
                raise RuntimeError(
                    "SM120 NVFP4 cooperative launch failed during CUDA graph capture"
                )
            _log_smallm_decision(
                "using CUTLASS because the SM120 launch is unavailable"
            )

    if output is None:
        raise RuntimeError("FlashInfer CUTLASS MoE output allocation failed")

    w13_weight = quant_info.w13_weight
    w2_weight = quant_info.w2_weight
    if quant_info.quant_type == "fp4":
        w13_weight = w13_weight.view(torch.long)
        w2_weight = w2_weight.view(torch.long)
        if quant_scales is None or len(quant_scales) != 6:
            raise ValueError("NVFP4 CUTLASS MoE requires six quantization scales")
        quant_scales = [
            quant_scales[0],
            quant_scales[1].view(torch.int32),
            quant_scales[2],
            quant_scales[3],
            quant_scales[4].view(torch.int32),
            quant_scales[5],
        ]

    output = flashinfer_cutlass_fused_moe(
        output=output,
        input=x,
        token_selected_experts=topk_ids.to(torch.int),
        token_final_scales=topk_weights,
        fc1_expert_weights=w13_weight,
        fc2_expert_weights=w2_weight,
        output_dtype=output_dtype,
        input_sf=x_sf,
        quant_scales=quant_scales,
        swiglu_alpha=quant_info.swiglu_alpha,
        swiglu_beta=quant_info.swiglu_beta,
        swiglu_limit=quant_info.swiglu_limit,
        ep_size=runner_config.moe_ep_size,
        ep_rank=runner_config.moe_ep_rank,
        tp_size=runner_config.moe_tp_size,
        tp_rank=runner_config.moe_tp_rank,
        tune_max_num_tokens=_cutlass_moe_tune_max_num_tokens(x.shape[0]),
        activation_type=_activation_type(runner_config),
        enable_alltoall=enable_alltoall,
        use_fused_finalize=envs.SGLANG_FLASHINFER_MOE_FUSED_FINALIZE.get(),
    )[0]

    if quant_info.quant_type in ("bf16", "fp8"):
        _maybe_apply_routed_scaling_factor(output, quant_info, runner_config)
    return output


@register_fused_func("none", "flashinfer_cutlass")
def fused_experts_none_to_flashinfer_cutlass(
    dispatch_output: StandardDispatchOutput,
    quant_info: MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    assert isinstance(quant_info, FlashInferCutlassMoeQuantInfo), (
        f"Unexpected quant_info type for flashinfer_cutlass: {type(quant_info)}"
    )

    output = _run_flashinfer_cutlass(
        dispatch_output=dispatch_output,
        quant_info=quant_info,
        runner_config=runner_config,
    )
    return StandardCombineInput(hidden_states=output)


@register_fused_func("flashinfer", "flashinfer_cutlass")
def fused_experts_flashinfer_to_flashinfer_cutlass(
    dispatch_output: FlashinferDispatchOutput | StandardDispatchOutput,
    quant_info: MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> FlashinferCombineInput | StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
        FlashinferCombineInput,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput

    # All-gather prefill must use ordinary expert execution and reduce-scatter
    # combine, without the decode A2A workspace or kernel layout.
    if isinstance(dispatch_output, StandardDispatchOutput):
        return fused_experts_none_to_flashinfer_cutlass(
            dispatch_output, quant_info, runner_config
        )

    assert isinstance(quant_info, FlashInferCutlassMoeQuantInfo), (
        f"Unexpected quant_info type for flashinfer_cutlass: {type(quant_info)}"
    )

    output = _run_flashinfer_cutlass(
        dispatch_output=dispatch_output,
        quant_info=quant_info,
        runner_config=runner_config,
        output=dispatch_output.moe_output,
        enable_alltoall=True,
    )
    return FlashinferCombineInput(hidden_states=output)


def _b12x_w4a16_expert_map(
    *, num_experts: int, num_local_experts: int, ep_rank: int, ep_size: int, device
) -> Optional[torch.Tensor]:
    """Global-id -> local-id map for this EP rank, or None when there is no EP.

    flashinfer translates the (global) top-k ids itself when given a map and
    silently falls back to a rank-0 prefix map otherwise, so an EP caller must
    always pass its own. sglang without EPLB gives rank r the contiguous chunk
    ``[r * num_local_experts, (r + 1) * num_local_experts)``, which is what this
    encodes; the MoE layer then all-reduces the per-rank partials.

    The map relies on the chunk covering the whole expert range: a non-divisible
    split would clip the slice (torch clamps), silently leaving trailing experts
    unroutable, so that is asserted rather than assumed.
    ``flashinfer.experimental.b12x.moe.ep_moe.prepare_ep_expert_map`` validates
    exactly this shape (int32, contiguous, rank-1, every local id used once), but
    for the sibling ``b12x.*`` package that ``launch_sm120_moe`` does not consume,
    so the equivalent construction is spelled out here.
    """
    if ep_size <= 1:
        return None
    assert ep_size * num_local_experts == num_experts, (
        f"b12x W4A16 expects a divisible EP split: {ep_size} * "
        f"{num_local_experts} != {num_experts}"
    )
    expert_map = torch.full((num_experts,), -1, dtype=torch.int32, device=device)
    local_start = ep_rank * num_local_experts
    expert_map[local_start : local_start + num_local_experts] = torch.arange(
        num_local_experts, dtype=torch.int32, device=device
    )
    return expert_map


def _run_flashinfer_b12x_w4a16(
    *,
    dispatch_output,
    quant_info: FlashInferCutlassMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    output: Optional[torch.Tensor],
    enable_alltoall: bool,
) -> torch.Tensor:
    """FP4 weights, BF16 activations on flashinfer's SM120 b12x kernels.

    Because the activations are never quantized, no activation scale may reach
    the kernel: the global scales the load-time preparation folded into the
    packed scale grid are weight-only (``w13_weight_scale_2`` /
    ``w2_weight_scale_2``), which is why ``ModelOptNvFp4FusedMoEMethod``
    neutralizes the checkpoint's input scales for this route. Passing the usual
    ``g1_alphas = input_scale * weight_scale_2`` product would apply the
    activation scale twice and silently rescale every expert output.
    """
    if enable_alltoall:
        raise NotImplementedError(
            "FlashInfer b12x W4A16 does not support expert all-to-all dispatch; "
            "run expert parallelism with --moe-a2a-backend none."
        )
    if not get_moe_a2a_backend().is_none():
        raise NotImplementedError(
            "FlashInfer b12x W4A16 requires --moe-a2a-backend none, got "
            f"'{get_moe_a2a_backend().value}'."
        )
    if runner_config.activation != "silu" or not runner_config.is_gated:
        raise NotImplementedError(
            "FlashInfer b12x W4A16 supports gated SiLU (SwiGLU) MoE only, got "
            f"activation={runner_config.activation!r}, "
            f"is_gated={runner_config.is_gated}."
        )
    if runner_config.moe_tp_size != 1:
        raise NotImplementedError(
            "FlashInfer b12x W4A16 has no inner-dimension sharding of an expert: "
            f"moe_tp_size must be 1, got {runner_config.moe_tp_size}."
        )
    for name in ("num_experts", "num_local_experts", "top_k"):
        if getattr(runner_config, name) is None:
            raise ValueError(f"b12x W4A16 requires {name} in the runner config")
    if quant_info.prepared_weights is None:
        raise RuntimeError(
            "FlashInfer b12x W4A16 requires load-time prepared weights: this "
            "layer did not go through process_weights_after_loading for the "
            "b12x route (or its tensors were replaced afterwards)."
        )
    if quant_info.quant_scales is None or len(quant_info.quant_scales) != 4:
        got = None if quant_info.quant_scales is None else len(quant_info.quant_scales)
        raise ValueError(
            "b12x W4A16 expects four prepared scales: w13 block scales, w13 "
            f"global scale, w2 block scales, w2 global scale, got {got}."
        )

    x = dispatch_output.hidden_states
    if x.dtype != torch.bfloat16:
        # The route is validated for bf16 activations only, and the output
        # allocation below is fixed to bf16; fail closed rather than guessing.
        raise ValueError(
            f"FlashInfer b12x W4A16 expects bf16 activations, got {x.dtype}"
        )

    topk_output = dispatch_output.topk_output
    if output is None:
        with use_symmetric_memory(
            get_parallel().tp_group, disabled=not is_allocation_symmetric()
        ):
            if runner_config.moe_ep_size > 1:
                # EP zero-fill: a rank writes only rows with at least one local
                # expert, so unwritten rows would carry allocator garbage into
                # the layer all-reduce that sums the per-rank partials. With EP
                # off this rank writes every row, so empty_like skips zeroing.
                output = torch.zeros_like(x, dtype=torch.bfloat16)
            else:
                output = torch.empty_like(x, dtype=torch.bfloat16)

    expert_map = _b12x_w4a16_expert_map(
        num_experts=int(runner_config.num_experts),
        num_local_experts=int(runner_config.num_local_experts),
        ep_rank=int(runner_config.moe_ep_rank),
        ep_size=int(runner_config.moe_ep_size),
        device=x.device,
    )

    launch_sm120_moe = b12x_w4a16_launch()
    w13_scale, w13_global_scale, w2_scale, w2_global_scale = quant_info.quant_scales
    return launch_sm120_moe(
        a=x,
        w1_weight=quant_info.w13_weight,
        w1_weight_sf=w13_scale,
        w1_alpha=w13_global_scale,
        w2_weight=quant_info.w2_weight,
        w2_weight_sf=w2_scale,
        w2_alpha=w2_global_scale,
        topk_ids=topk_output.topk_ids.to(torch.int),
        topk_weights=topk_output.topk_weights,
        num_experts=int(runner_config.num_experts),
        num_local_experts=int(runner_config.num_local_experts),
        top_k=int(runner_config.top_k),
        scatter_output=output,
        expert_map=expert_map,
        activation="silu",
        quant_mode="w4a16",
        source_format=B12X_W4A16_SOURCE_FORMAT,
        _prepared_weights=quant_info.prepared_weights,
    )


@register_fused_func("none", "flashinfer_b12x_w4a16")
def fused_experts_none_to_flashinfer_b12x_w4a16(
    dispatch_output: StandardDispatchOutput,
    quant_info: MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    assert isinstance(quant_info, FlashInferCutlassMoeQuantInfo), (
        f"Unexpected quant_info type for flashinfer_b12x_w4a16: {type(quant_info)}"
    )
    assert quant_info.quant_type == "w4a16_b12x", (
        f"Unexpected quant_type {quant_info.quant_type!r} for "
        "flashinfer_b12x_w4a16; the quantization method must build the b12x "
        "payload."
    )

    output = _run_flashinfer_b12x_w4a16(
        dispatch_output=dispatch_output,
        quant_info=quant_info,
        runner_config=runner_config,
        output=None,
        enable_alltoall=False,
    )
    return StandardCombineInput(hidden_states=output)


@register_fused_func("none", "flashinfer_mxfp4")
def fused_experts_none_to_flashinfer_mxfp4(
    dispatch_output: StandardDispatchOutput,
    quant_info: MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    """Dispatch flashinfer_mxfp4 by quant-info type.

    Both mxfp4 paths register under this single ``("none", "flashinfer_mxfp4")``
    key but call different kernels.
    """
    if isinstance(quant_info, FlashInferCutlassMxfp4MoeQuantInfo):
        return _fused_experts_flashinfer_mxfp4_cutlass(
            dispatch_output, quant_info, runner_config
        )

    # Keep one fused-op registration for the shared backend while loading the
    # TRT-LLM implementation only when its quant-info type is dispatched.
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        FlashInferTrtllmGenMxfp4MoeQuantInfo,
        _fused_experts_flashinfer_mxfp4_sm100_trtllm_gen,
    )

    if isinstance(quant_info, FlashInferTrtllmGenMxfp4MoeQuantInfo):
        return _fused_experts_flashinfer_mxfp4_sm100_trtllm_gen(
            dispatch_output, quant_info, runner_config
        )
    raise TypeError(
        f"Unexpected quant_info type for flashinfer_mxfp4: {type(quant_info)}"
    )


def _fused_experts_flashinfer_mxfp4_cutlass(
    dispatch_output: StandardDispatchOutput,
    quant_info: FlashInferCutlassMxfp4MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    x = dispatch_output.hidden_states
    if x.shape[0] == 0:
        return StandardCombineInput(hidden_states=x)

    flashinfer_cutlass_fused_moe, ActivationType = _flashinfer_cutlass_fused_moe()

    topk_output = dispatch_output.topk_output

    # Under ``--moe-runner-backend flashinfer_mxfp4`` topk may be in bypassed
    # form (the SM100 trtllm-gen path does routing internally). The CUTLASS
    # SM90 path needs explicit topk_ids / topk_weights; materialize here.
    if TopKOutputChecker.format_is_bypassed(topk_output):
        topk_output = topk_output.to_standard()
    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights

    # GPT-OSS: pad input hidden dim up to the loaded weight width. DSv4
    # leaves padded_hidden as None (or equal to origin_hidden), no pad.
    origin_hidden = x.shape[-1]
    padded_hidden = quant_info.padded_hidden
    do_pad = padded_hidden is not None and padded_hidden != origin_hidden
    if do_pad:
        x = torch.nn.functional.pad(
            x,
            (0, padded_hidden - origin_hidden),
            mode="constant",
            value=0.0,
        )

    weight_global_scale = quant_info.mxfp4_weight_global_scale
    use_mxfp8_act_scaling = weight_global_scale is not None
    w13_humming_residual_scale = quant_info.w13_humming_residual_scale
    w2_humming_residual_scale = quant_info.w2_humming_residual_scale
    humming_fc2_act_scale = quant_info.humming_fc2_act_scale
    humming_scales = (
        w13_humming_residual_scale,
        w2_humming_residual_scale,
        humming_fc2_act_scale,
    )
    use_wfp4afp8_humming = any(scale is not None for scale in humming_scales)
    if use_wfp4afp8_humming and not all(scale is not None for scale in humming_scales):
        raise ValueError(
            "SM90 Humming MXFP4 MoE requires both expert residual scales "
            "and the FC2 activation scale."
        )
    if use_wfp4afp8_humming and use_mxfp8_act_scaling:
        raise ValueError("SM90 Humming and SM120 MXFP8 scaling are mutually exclusive.")
    input_sf = None
    fc1_expert_weights = quant_info.w13_weight
    fc2_expert_weights = quant_info.w2_weight
    if weight_global_scale is not None:
        from flashinfer import mxfp8_quantize

        x, input_sf = mxfp8_quantize(
            x,
            is_sf_swizzled_layout=True,
            alignment=32,
        )
        fc1_expert_weights = fc1_expert_weights.view(torch.int64)
        fc2_expert_weights = fc2_expert_weights.view(torch.int64)
        quant_scales = [
            quant_info.w13_weight_scale.view(torch.int32),
            weight_global_scale,
            quant_info.w2_weight_scale.view(torch.int32),
            weight_global_scale,
        ]
    elif use_wfp4afp8_humming:
        assert w13_humming_residual_scale is not None
        assert w2_humming_residual_scale is not None
        assert humming_fc2_act_scale is not None
        quant_scales = [
            quant_info.w13_weight_scale.view(torch.int32),
            w13_humming_residual_scale,
            humming_fc2_act_scale,
            quant_info.w2_weight_scale.view(torch.int32),
            w2_humming_residual_scale,
        ]
    else:
        quant_scales = [
            quant_info.w13_weight_scale.view(torch.int32),
            quant_info.w2_weight_scale.view(torch.int32),
        ]

    out_hidden = padded_hidden if do_pad else origin_hidden
    output_dtype = torch.bfloat16
    # FlashInfer 0.6.17 intentionally reverted the Humming API. Do not pass the
    # new keyword at all on the existing W4A16/MXFP8 paths, so those paths keep
    # working with SGLang's currently pinned release.
    humming_kwargs = {"use_wfp4afp8_humming": True} if use_wfp4afp8_humming else {}
    with use_symmetric_memory(
        get_parallel().tp_group, disabled=not is_allocation_symmetric()
    ):
        out = torch.empty(x.shape[0], out_hidden, dtype=output_dtype, device=x.device)

    flashinfer_cutlass_fused_moe(
        input=x,
        token_selected_experts=topk_ids.to(torch.int32),
        token_final_scales=topk_weights,
        fc1_expert_weights=fc1_expert_weights,
        fc2_expert_weights=fc2_expert_weights,
        output_dtype=output_dtype,
        quant_scales=quant_scales,
        input_sf=input_sf,
        fc1_expert_biases=quant_info.w13_bias,
        fc2_expert_biases=quant_info.w2_bias,
        swiglu_alpha=quant_info.swiglu_alpha,
        swiglu_beta=quant_info.swiglu_beta,
        swiglu_limit=quant_info.swiglu_limit,
        tp_size=runner_config.moe_tp_size,
        tp_rank=runner_config.moe_tp_rank,
        ep_size=runner_config.moe_ep_size,
        ep_rank=runner_config.moe_ep_rank,
        use_w4_group_scaling=not use_mxfp8_act_scaling,
        use_mxfp8_act_scaling=use_mxfp8_act_scaling,
        activation_type=(
            ActivationType.SwigluStep
            if quant_info.use_swiglu_step
            else ActivationType.Swiglu
        ),
        tune_max_num_tokens=_cutlass_moe_tune_max_num_tokens(x.shape[0]),
        output=out,
        use_fused_finalize=envs.SGLANG_FLASHINFER_MOE_FUSED_FINALIZE.get(),
        **humming_kwargs,
    )

    if do_pad:
        out = out[:, :origin_hidden].contiguous()

    return StandardCombineInput(hidden_states=out)


# merge-#5 flipped flashinfer_cutlass.py from take-upstream to fork-owned: the b12x
# W4A16 route (entry 247) is fork-only (origin/main has ZERO b12x refs). #44993/
# #44994's upstream-side delta was re-woven over the fork in B5 (pair_w2_weight +
# the sm120 native MTP-MMA route + its fail-loud paired-weight guard + the
# all-gather-prefill Standard delegation); the tune-range helper had already
# arrived at the merge. D-3 re-verify stays closed: 534x/415x alpha-fold,
# w13_layout and source_format spelling vs flashinfer 0.7.2 are in the fork here.
