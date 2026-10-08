# SPDX-License-Identifier: Apache-2.0
import dataclasses

from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import LTX2PipelineConfig


@dataclasses.dataclass
class LTXVideoPipelineConfig(LTX2PipelineConfig):
    """Config slot for LTX-Video v1.

    v1's ``model_index.json`` declares the diffusers ``LTXPipeline``, which has
    no native sglang counterpart, so resolution falls back to the diffusers
    backend and only ``task_type`` is consumed from this class (identical to
    LTX-2's TI2V). It exists as a distinct type so the v1 registration does not
    share the LTX-2 config object.
    """


def register():
    from sglang.multimodal_gen.configs.sample.ltx_video import (
        LTXVideoSamplingParams,
    )
    from sglang.multimodal_gen.registry import register_configs

    # Exact-path match is consulted before detectors, so this outranks ltx_2's
    # broad "ltx"+"video" detector, which used to resolve v1 to
    # LTX2SamplingParams and its 254-token (128-truncated) negative prompt.
    register_configs(
        sampling_param_cls=LTXVideoSamplingParams,
        pipeline_config_cls=LTXVideoPipelineConfig,
        hf_model_paths=["Lightricks/LTX-Video"],
    )
