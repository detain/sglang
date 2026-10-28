# SPDX-License-Identifier: Apache-2.0
"""max_sequence_length forwarding in the diffusers fallback.

``batch.max_sequence_length`` is populated from the request's top-level
``max_sequence_length`` field (or from ``diffusers_kwargs``), but
``DiffusersExecutionStage._build_pipeline_kwargs`` used to ignore it, so on
fallback-served models (e.g. LTX-Video v1 -> diffusers ``LTXPipeline``, whose
T5 encoder caps prompts at 128 tokens) the field was a silent no-op and long
prompts kept getting truncated. The only working knob was nesting the value
under ``diffusers_kwargs``. CPU-only; no weights, no network.
"""

import unittest


class _LtxLikePipe:
    """Mirrors diffusers LTXPipeline: accepts max_sequence_length."""

    def __call__(
        self,
        prompt=None,
        negative_prompt=None,
        num_inference_steps=50,
        guidance_scale=3.0,
        height=128,
        width=128,
        num_frames=97,
        max_sequence_length=128,
    ):
        return {}


class _NoMslPipe:
    """Fallback pipeline without a max_sequence_length kwarg."""

    def __call__(
        self,
        prompt=None,
        negative_prompt=None,
        num_inference_steps=50,
        height=480,
        width=832,
        num_frames=81,
    ):
        return {}


class _VarKwPipe:
    def __call__(self, prompt=None, **kwargs):
        return {}


def _make_stage(pipe):
    from sglang.multimodal_gen.runtime.pipelines.diffusers_pipeline import (
        DiffusersExecutionStage,
    )

    return DiffusersExecutionStage(pipe)


def _make_batch():
    from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

    sp = SamplingParams(
        prompt="a long pixel-art prompt",
        negative_prompt="",
        height=512,
        width=512,
        num_frames=97,
        num_inference_steps=40,
        guidance_scale=3.0,
    )
    sp.seed = None  # type: ignore[assignment]  # keep the stage from building a device generator
    return Req(sampling_params=sp)


class TestMaxSequenceLengthForwarding(unittest.TestCase):
    def test_unset_is_not_forwarded(self):
        stage = _make_stage(_LtxLikePipe())
        kwargs = stage._build_pipeline_kwargs(_make_batch())
        self.assertNotIn("max_sequence_length", kwargs)

    def test_set_is_forwarded_when_pipeline_accepts(self):
        stage = _make_stage(_LtxLikePipe())
        batch = _make_batch()
        batch.max_sequence_length = 256
        kwargs = stage._build_pipeline_kwargs(batch)
        self.assertEqual(kwargs["max_sequence_length"], 256)

    def test_set_is_skipped_when_pipeline_rejects(self):
        stage = _make_stage(_NoMslPipe())
        batch = _make_batch()
        batch.max_sequence_length = 256
        kwargs = stage._build_pipeline_kwargs(batch)
        self.assertNotIn("max_sequence_length", kwargs)

    def test_var_keyword_pipeline_accepts(self):
        stage = _make_stage(_VarKwPipe())
        batch = _make_batch()
        batch.max_sequence_length = 192
        kwargs = stage._build_pipeline_kwargs(batch)
        self.assertEqual(kwargs["max_sequence_length"], 192)

    def test_diffusers_kwargs_still_win(self):
        stage = _make_stage(_LtxLikePipe())
        batch = _make_batch()
        batch.max_sequence_length = 256
        batch.extra["diffusers_kwargs"] = {"max_sequence_length": 512}
        kwargs = stage._build_pipeline_kwargs(batch)
        self.assertEqual(kwargs["max_sequence_length"], 512)

    def test_forwarded_kwarg_survives_filter(self):
        # strict=True turns a signature mismatch into a test failure.
        stage = _make_stage(_LtxLikePipe())
        batch = _make_batch()
        batch.max_sequence_length = 256
        kwargs = stage._build_pipeline_kwargs(batch)
        filtered, ignored = stage._filter_pipeline_kwargs(kwargs, strict=True)
        self.assertEqual(filtered["max_sequence_length"], 256)
        self.assertEqual(ignored, [])


if __name__ == "__main__":
    unittest.main()
