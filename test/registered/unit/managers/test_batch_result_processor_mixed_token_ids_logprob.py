import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

# Expected host values, derived from the same float32 constructions the
# producer tensors use (tolist widens float32, so literals would not compare).
_FLAT_LOGPROBS = torch.tensor([-0.1, -0.2]).tolist()
_TOP_ROW0 = torch.tensor([-0.1]).tolist()
_TOP_ROW1 = torch.tensor([-0.2]).tolist()
_TOP_IDX0 = torch.tensor([10]).tolist()
_TOP_IDX1 = torch.tensor([11]).tolist()
_TOKEN_IDS_ROW0 = torch.tensor([-15.0]).tolist()


def _make_processor(case) -> SchedulerBatchResultProcessor:
    override = get_context().override_server_args(
        enable_return_hidden_states=True,
        return_hidden_states_mode="full",
    )
    override.install()
    case.addCleanup(override.restore)
    metrics_reporter = Mock()
    metrics_reporter.num_generated_tokens = 0
    metrics_reporter.forward_ct_decode = 0
    return SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=None,
        enable_overlap=False,
        enable_overlap_mlx=False,
        model_config=SimpleNamespace(think_end_ids=None),
        token_to_kv_pool_allocator=Mock(),
        tree_cache=None,
        hisparse_coordinator=None,
        req_to_token_pool=None,
        decode_offload_manager=None,
        metrics_collector=None,
        metrics_reporter=metrics_reporter,
        draft_worker=None,
        model_worker=Mock(),
        logprob_result_processor=None,
        output_streamer=Mock(),
        beam_coordinator=Mock(),
        abort_request=lambda *args, **kwargs: None,
    )


def _mixed_token_ids_logprob_output() -> LogitsProcessorOutput:
    # Producer contract (logprob_processor.get_token_ids_logprobs_raw DECODE,
    # no_copy_to_cpu path): tensor rows for requests that asked for
    # token_ids_logprob, [] placeholder rows for requests that did not.
    return LogitsProcessorOutput(
        next_token_logits=None,
        next_token_logprobs=torch.tensor([-0.1, -0.2]),
        next_token_top_logprobs_val=[torch.tensor([-0.1]), torch.tensor([-0.2])],
        next_token_top_logprobs_idx=[torch.tensor([10]), torch.tensor([11])],
        next_token_token_ids_logprobs_val=[torch.tensor([-15.0]), []],
        next_token_token_ids_logprobs_idx=[[151645], []],
    )


def _make_decode_req(*, token_ids_logprob):
    req = SimpleNamespace(
        return_logprob=True,
        return_sampling_mask=False,
        return_hidden_states=False,
        is_retracted=False,
        output_ids=[],
        time_stats=Mock(),
        grammar=None,
        beam_group=None,
        logprob=SimpleNamespace(
            top_logprobs_num=1,
            token_ids_logprob=token_ids_logprob,
            output_token_logprobs_val=[],
            output_token_logprobs_idx=[],
            output_top_logprobs_val=[],
            output_top_logprobs_idx=[],
            output_token_ids_logprobs_val=[],
            output_token_ids_logprobs_idx=[],
        ),
    )
    req.finished = lambda: False
    req.update_finish_state = Mock()
    return req


def _make_decode_batch(reqs):
    return SimpleNamespace(
        reqs=reqs,
        return_logprob=True,
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
        batch_size=lambda: len(reqs),
    )


class TestNormalizeDecodeOutputsMixedTokenIdsLogprob(CustomTestCase):
    def test_placeholder_rows_survive_tolist_conversion(self):
        processor = _make_processor(self)
        logits_output = _mixed_token_ids_logprob_output()
        placeholder_val_row = logits_output.next_token_token_ids_logprobs_val[1]
        result = GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=torch.tensor([5, 6]),
        )

        next_token_ids, next_token_logprobs = processor._normalize_decode_outputs(
            batch=_make_decode_batch([_make_decode_req(token_ids_logprob=[151645])]),
            result=result,
            logits_output=logits_output,
            next_token_ids=torch.tensor([5, 6]),
        )

        self.assertEqual(next_token_ids, [[5], [6]])
        self.assertEqual(next_token_logprobs, _FLAT_LOGPROBS)
        self.assertEqual(
            logits_output.next_token_top_logprobs_val, [_TOP_ROW0, _TOP_ROW1]
        )
        self.assertEqual(
            logits_output.next_token_top_logprobs_idx, [_TOP_IDX0, _TOP_IDX1]
        )
        # req0's tensor row converted to a list; req1's [] placeholder kept as-is.
        self.assertEqual(
            logits_output.next_token_token_ids_logprobs_val, [_TOKEN_IDS_ROW0, []]
        )
        self.assertIs(
            logits_output.next_token_token_ids_logprobs_val[1], placeholder_val_row
        )

    def test_full_decode_flow_stores_only_requested_token_ids_logprobs(self):
        processor = _make_processor(self)
        req0 = _make_decode_req(token_ids_logprob=[151645])
        req1 = _make_decode_req(token_ids_logprob=None)
        batch = _make_decode_batch([req0, req1])
        logits_output = _mixed_token_ids_logprob_output()
        result = GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=torch.tensor([5, 6]),
        )

        with (
            patch.object(
                SchedulerBatchResultProcessor, "_maybe_update_reasoning_tokens"
            ),
            patch.object(
                SchedulerBatchResultProcessor, "_handle_finish_state_updated_req"
            ),
        ):
            processor.process_batch_result_decode(batch, result)

        # req0: the tensor row landed, converted through the shared helper.
        self.assertEqual(req0.output_ids, [5])
        self.assertEqual(req0.logprob.output_token_logprobs_val, [_FLAT_LOGPROBS[0]])
        self.assertEqual(req0.logprob.output_top_logprobs_val, [_TOP_ROW0])
        self.assertEqual(req0.logprob.output_token_ids_logprobs_val, [_TOKEN_IDS_ROW0])
        self.assertEqual(req0.logprob.output_token_ids_logprobs_idx, [[151645]])
        # req1: guard in _apply_decode_logprobs skips the [] placeholder row.
        self.assertEqual(req1.output_ids, [6])
        self.assertEqual(req1.logprob.output_token_logprobs_val, [_FLAT_LOGPROBS[1]])
        self.assertEqual(req1.logprob.output_top_logprobs_val, [_TOP_ROW1])
        self.assertEqual(req1.logprob.output_token_ids_logprobs_val, [])
        self.assertEqual(req1.logprob.output_token_ids_logprobs_idx, [])
        processor.output_streamer.stream_output.assert_called_once_with(
            [req0, req1], True
        )


class TestMoveLogprobsToCpuMixedRows(CustomTestCase):
    def test_placeholder_rows_survive_tolist_conversion(self):
        processor = _make_processor(self)
        logits_output = _mixed_token_ids_logprob_output()
        placeholder_val_row = logits_output.next_token_token_ids_logprobs_val[1]

        processor.move_logprobs_to_cpu(
            batch=SimpleNamespace(return_logprob=True),
            logits_output=logits_output,
        )

        self.assertEqual(logits_output.next_token_logprobs, _FLAT_LOGPROBS)
        self.assertEqual(
            logits_output.next_token_top_logprobs_val, [_TOP_ROW0, _TOP_ROW1]
        )
        self.assertEqual(
            logits_output.next_token_top_logprobs_idx, [_TOP_IDX0, _TOP_IDX1]
        )
        self.assertEqual(
            logits_output.next_token_token_ids_logprobs_val, [_TOKEN_IDS_ROW0, []]
        )
        self.assertIs(
            logits_output.next_token_token_ids_logprobs_val[1], placeholder_val_row
        )


if __name__ == "__main__":
    unittest.main()
