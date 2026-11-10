import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import ForwardMode

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestMixedGrammarSyncDecision(unittest.TestCase):
    """Stack tests ported from test_scheduler_decision_batch_params.py, which
    upstream deleted in 517cfa218c (#41952) without a new home. They cover the
    mixed-decode-tail grammar sync fix (54613d2065, PR #40458), which is not
    in upstream main."""

    @staticmethod
    def _batch(*, prefill_grammar, decode_grammar):
        prefill_req = SimpleNamespace(grammar=prefill_grammar)
        decode_req = SimpleNamespace(grammar=decode_grammar)
        batch = ScheduleBatch(reqs=[prefill_req, decode_req])
        batch.forward_mode = ForwardMode.MIXED
        batch.has_grammar = prefill_grammar is not None or decode_grammar is not None
        batch.decoding_reqs = [decode_req]
        batch.spec_algorithm = SimpleNamespace(
            is_none=lambda: False, supports_grammar_overlap=lambda: True
        )
        return batch

    def test_syncs_when_decode_tail_has_grammar(self):
        batch = self._batch(prefill_grammar=None, decode_grammar=object())

        self.assertTrue(batch.grammar_needs_sync())

    def test_does_not_sync_for_prefill_only_grammar(self):
        batch = self._batch(prefill_grammar=object(), decode_grammar=None)

        self.assertFalse(batch.grammar_needs_sync())

    def test_regular_decode_keeps_worker_grammar_overlap(self):
        batch = self._batch(prefill_grammar=None, decode_grammar=object())
        batch.forward_mode = ForwardMode.DECODE

        self.assertFalse(batch.grammar_needs_sync())


class TestMixedGrammarDecodeTailForcesSync(unittest.TestCase):
    @staticmethod
    def _batch(*, grammar_needs_sync: bool = False):
        return SimpleNamespace(
            is_extend_in_batch=True,
            forward_mode=SimpleNamespace(
                is_extend=lambda: True,
                is_decode=lambda: False,
                is_mixed=lambda: True,
            ),
            spec_algorithm=SimpleNamespace(is_none=lambda: False),
            grammar_needs_sync=lambda: grammar_needs_sync,
        )

    def _scheduler(self, *, require_mlp_sync: bool):
        scheduler = object.__new__(Scheduler)
        scheduler.require_mlp_sync = require_mlp_sync
        scheduler.tree_cache = SimpleNamespace(disable=True)
        scheduler.result_queue = []
        return scheduler

    @patch(
        "sglang.srt.managers.scheduler.envs.SGLANG_DISABLE_OVERLAP_AFTER_PREFILL.get",
        return_value=False,
    )
    def test_mixed_grammar_decode_tail_forces_sync(
        self, _disable_overlap_after_prefill
    ):
        # #47716 folded is_disable_overlap_for_batch's grammar-sync arm into
        # the pre-schedule early-processing decision; the mixed decode tail
        # still forces the sync through the batch predicate.
        mixed = self._batch(grammar_needs_sync=True)
        scheduler = self._scheduler(require_mlp_sync=False)

        self.assertTrue(scheduler.should_process_last_batch_early(mixed))


if __name__ == "__main__":
    unittest.main()
