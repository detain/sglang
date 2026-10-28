# SPDX-License-Identifier: Apache-2.0
"""LTX-Video v1 registry wiring.

v1 (`Lightricks/LTX-Video`) has no native pipeline and runs through the
diffusers `LTXPipeline` fallback, whose T5 tokenizer caps inputs at 128
tokens. Before the dedicated v1 registration, ltx_2's broad
"ltx"+"video" detector resolved it to `LTX2SamplingParams`, whose 1078-char
default negative is 254 T5 tokens -- diffusers silently truncated it in
half on every request. CPU-only; no weights, no network.
"""

import unittest


class TestLTXVideoV1Resolution(unittest.TestCase):
    def _resolve(self, model_path):
        # Not `get_model_info`: it also reads `model_index.json` from the Hub.
        from sglang.multimodal_gen.registry import _get_config_info

        return _get_config_info(model_path)

    def test_v1_resolves_to_dedicated_sampling_params(self):
        from sglang.multimodal_gen.configs.sample.ltx_video import (
            LTXVideoSamplingParams,
        )

        config = self._resolve("Lightricks/LTX-Video")
        self.assertIs(config.sampling_param_cls, LTXVideoSamplingParams)

        params = config.sampling_param_cls()
        self.assertIsInstance(params.negative_prompt, str)
        # v1's official default frame rate, not LTX-2's 24.
        self.assertEqual(params.fps, 25)

    def test_derived_v1_repos_keep_the_v1_config(self):
        from sglang.multimodal_gen.configs.sample.ltx_video import (
            LTXVideoSamplingParams,
        )

        # Partial match on the registered "LTX-Video" stem beats detectors.
        config = self._resolve("myorg/LTX-Video-finetune")
        self.assertIs(config.sampling_param_cls, LTXVideoSamplingParams)

    def test_v1_negative_prompt_fits_the_128_token_t5_budget(self):
        from sglang.multimodal_gen.configs.sample.ltx_2 import LTX2SamplingParams
        from sglang.multimodal_gen.configs.sample.ltx_video import (
            LTXVideoSamplingParams,
        )

        negative = LTXVideoSamplingParams().negative_prompt
        # 107 T5 tokens (incl. special tokens) with v1's shipped tokenizer.
        # Bound chosen from LTX-2's observed density (1078 chars = 254 tokens,
        # ~4.24 chars/token): 540 chars is at most ~127 tokens at that density,
        # so the truncation regression cannot silently come back via additions.
        self.assertLessEqual(len(negative), 540)
        self.assertLess(len(negative), len(LTX2SamplingParams().negative_prompt))

    def test_ltx2_family_resolution_unchanged(self):
        from sglang.multimodal_gen.configs.sample.ltx_2 import (
            LTX2SamplingParams,
            LTX23SamplingParams,
        )
        from sglang.multimodal_gen.configs.sample.ltx_2_5 import LTX25SamplingParams

        self.assertIs(
            self._resolve("Lightricks/LTX-2").sampling_param_cls, LTX2SamplingParams
        )
        self.assertIs(
            self._resolve("Lightricks/LTX-2.3").sampling_param_cls, LTX23SamplingParams
        )
        self.assertIs(
            self._resolve("Lightricks/LTX-2.5-Diffusers").sampling_param_cls,
            LTX25SamplingParams,
        )
        # The LTX-2 negative (Gemma encoder, 1024-token capacity) stays intact.
        self.assertEqual(len(LTX2SamplingParams().negative_prompt), 1078)


if __name__ == "__main__":
    unittest.main()
