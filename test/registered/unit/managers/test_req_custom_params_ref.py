import gc
import unittest
import weakref
from array import array

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.custom_logit_processor import (
    Qwen3ThinkingBudgetLogitProcessor,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_req(custom_logit_processor=None, origin_input_ids=(1, 2)):
    return Req(
        rid="r",
        origin_input_text="",
        origin_input_ids=array("q", origin_input_ids),
        sampling_params=SamplingParams(custom_params={"thinking_budget": 8}),
        custom_logit_processor=custom_logit_processor,
    )


class TestReqCustomParamsRef(CustomTestCase):
    def test_req_without_a_processor_is_freed_by_refcount(self):
        """custom_params without a custom logit processor must not make the
        request reference itself; refcount alone frees it."""
        gc.disable()
        try:
            req = _make_req()
            self.assertNotIn("__req__", req.sampling_params.custom_params)
            ref = weakref.ref(req)
            del req
            self.assertIsNone(ref())
        finally:
            gc.enable()

    def test_processor_reads_its_request(self):
        """A custom logit processor still finds its request under __req__,
        and the reference is weak (fork pin re-aimed at #45923 semantics)."""
        req = _make_req(custom_logit_processor="{}")
        entry = req.sampling_params.custom_params["__req__"]
        self.assertIn(type(entry), weakref.ProxyTypes)
        self.assertIs(entry.__class__, Req)
        self.assertEqual(entry.rid, req.rid)

    def test_req_with_a_processor_is_freed_by_refcount(self):
        """A custom logit processor's "__req__" entry must not keep the request alive."""
        gc.disable()
        try:
            req = _make_req(
                custom_logit_processor=Qwen3ThinkingBudgetLogitProcessor.to_str()
            )
            ref = weakref.ref(req)
            del req
            self.assertIsNone(ref())
        finally:
            gc.enable()

    def test_processor_reads_the_request_through_its_custom_params(self):
        processor = Qwen3ThinkingBudgetLogitProcessor()
        req = _make_req(
            custom_logit_processor=processor.to_str(),
            origin_input_ids=(processor.THINKING_START_TOKEN_ID,),
        )
        # The thinking budget of 8 is spent: the processor forces a newline.
        req.output_ids = [100] * 8
        logits = processor(torch.zeros(1, 200000), [req.sampling_params.custom_params])
        self.assertEqual(logits[0, processor.NEW_LINE_TOKEN_ID].item(), 0.0)
        self.assertTrue(torch.isneginf(logits[0, 100]))


if __name__ == "__main__":
    unittest.main()

# TODO(merge5): fold upstream changes dropped here (entries 306 own this file; batch B2)
