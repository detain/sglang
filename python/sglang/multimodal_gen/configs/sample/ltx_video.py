# SPDX-License-Identifier: Apache-2.0
import dataclasses

from sglang.multimodal_gen.configs.sample.ltx_2 import LTX2SamplingParams


@dataclasses.dataclass
class LTXVideoSamplingParams(LTX2SamplingParams):
    """Sampling parameters for LTX-Video v1.

    v1 has no native pipeline yet and is served through the diffusers
    ``LTXPipeline`` fallback, whose T5 tokenizer caps inputs at
    ``model_max_length=128``. The LTX-2 family default negative is 254 T5
    tokens (it targets the Gemma encoder with 1024-token capacity), so diffusers
    silently drops half of it and warns on every warmup. Keep this negative
    comfortably under 128 T5 tokens -- measured 107 with v1's shipped tokenizer.
    """

    # Official v1 default frame rate (diffusers LTXPipeline exports at 25 fps);
    # the inherited 24 is the LTX-2 reference default.
    fps: int = 25

    negative_prompt: str = (
        "worst quality, inconsistent motion, blurry, jittery, distorted, watermark, "
        "text, low resolution, jpeg artifacts, deformed, disfigured, bad anatomy, "
        "extra limbs, incorrect anatomy, unrealistic materials, 3D CGI look, "
        "cartoonish rendering, oversaturated, underexposed, overexposed, flickering, "
        "noisy, grainy, compression artifacts, ugly, duplicate, morbid, mutilated, "
        "poorly drawn"
    )
