"""Unit tests for the EOS-during-reasoning guard in _check_token_based_finish.

Some reasoning models emit their EOS token (e.g. Qwen im_end) mid-thinking;
finishing on it truncates the reply with an empty answer (issue #24839).
While the reasoning phase is open, only user-provided stop_token_ids may
finish the request; model eos / tokenizer eos / additional_stop_token_ids
are suppressed.
"""

from array import array
from types import SimpleNamespace

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MODEL_EOS = 100  # stands in for im_end
ADDITIONAL_STOP = 101  # stands in for endoftext in additional_stop_token_ids
USER_STOP = 200
THINK_END = [90, 91]


def _make_tokenizer_stub():
    return SimpleNamespace(
        eos_token_id=MODEL_EOS,
        additional_stop_token_ids={ADDITIONAL_STOP},
    )


def _make_req(require_reasoning: bool, **sp_kwargs) -> Req:
    sp = SamplingParams(max_new_tokens=256, **sp_kwargs)
    sp.normalize(None)
    req = Req(
        rid="r0",
        origin_input_text="",
        origin_input_ids=array("q", [1, 2, 3]),
        sampling_params=sp,
        require_reasoning=require_reasoning,
        eos_token_ids={MODEL_EOS},
    )
    req.tokenizer = _make_tokenizer_stub()
    return req


class TestEosReasoningGuard(CustomTestCase):
    def _feed(self, req: Req, tokens) -> None:
        req.output_ids.extend(tokens)
        req.update_finish_state(len(tokens))

    def test_model_eos_suppressed_during_reasoning(self):
        for token in (MODEL_EOS, ADDITIONAL_STOP):
            with self.subTest(token=token):
                req = _make_req(require_reasoning=True)
                self._feed(req, [token])
                self.assertFalse(req.finished())

    def test_eos_run_suppressed_during_reasoning(self):
        # Speculative run of several model-EOS tokens while reasoning open.
        req = _make_req(require_reasoning=True)
        self._feed(req, [5, MODEL_EOS, 6])
        self.assertFalse(req.finished())

    def test_user_stop_token_finishes_during_reasoning(self):
        req = _make_req(require_reasoning=True, stop_token_ids={USER_STOP})
        self._feed(req, [USER_STOP])
        self.assertTrue(req.finished())
        self.assertEqual(req.finished_reason.matched, USER_STOP)

    def test_eos_finishes_after_reasoning_over(self):
        req = _make_req(require_reasoning=True)
        req.update_reasoning_tokens(THINK_END, THINK_END)
        self.assertTrue(req._is_reasoning_over)
        self._feed(req, [MODEL_EOS])
        self.assertTrue(req.finished())
        self.assertEqual(req.finished_reason.matched, MODEL_EOS)

    def test_eos_finishes_without_reasoning(self):
        # Baseline: require_reasoning=False keeps existing behavior untouched.
        req = _make_req(require_reasoning=False)
        self._feed(req, [MODEL_EOS])
        self.assertTrue(req.finished())

        req = _make_req(require_reasoning=False)
        self._feed(req, [ADDITIONAL_STOP])
        self.assertTrue(req.finished())

    def test_acceptance_run_eos_before_think_end_still_finishes(self):
        # Documented edge: under spec decoding one accepted run may contain an
        # EOS and a later think_end; reasoning state is advanced over the whole
        # run before the finish check, so the guard is lifted and EOS wins.
        req = _make_req(require_reasoning=True)
        run = [MODEL_EOS, THINK_END[0], THINK_END[1]]
        req.update_reasoning_tokens(run, THINK_END)
        self.assertTrue(req._is_reasoning_over)
        self._feed(req, run)
        self.assertTrue(req.finished())


if __name__ == "__main__":
    from unittest import main

    main()
