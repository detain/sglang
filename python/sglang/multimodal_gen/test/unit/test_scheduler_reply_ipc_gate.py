# SPDX-License-Identifier: Apache-2.0
"""Reply-direction CUDA-IPC spill must stay symmetric with the client deref.

The client only materializes ``CudaIpcRef`` handles when the scheduler
endpoint passes ``is_local_endpoint`` (``_materialize_local_cuda_refs`` in
``runtime/scheduler_client.py``). Spilling CUDA tensors into handles for a
non-loopback endpoint (e.g. a server started with ``--host 0.0.0.0``) ships
objects the client can never resolve: they reach the API layer raw, where
responses with live tensors in ``OutputBatch.output`` (``save_output=False``
/ ``return_file_paths_only=False``) crash on the first consumer, e.g.
``len(result.output)``.
"""

import pickle
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import sglang.multimodal_gen.runtime.managers.scheduler as scheduler_module
import torch
from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch


def _make_scheduler(endpoint: str):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.receiver = Mock()
    scheduler.server_args = SimpleNamespace(scheduler_endpoint=endpoint)
    scheduler._record_return_stage = lambda output_batch, name: nullcontext()
    return scheduler


def _reply_output_batch(scheduler, output_batch):
    Scheduler.return_result(scheduler, output_batch, identity=b"req-id")
    sent = scheduler.receiver.send_multipart.call_args.args[0]
    return pickle.loads(sent[2])


def test_nonlocal_endpoint_does_not_spill_cuda_tensors(monkeypatch):
    spill_calls = []

    def _fake_spill(value, *, in_place=False):
        spill_calls.append(value)
        # stand in for the real tree walk: mark tensors as spilled so the
        # reply proves whether the client would receive raw handles
        if isinstance(value, OutputBatch) and isinstance(value.output, torch.Tensor):
            value.output = "SPILLED"
        return value

    monkeypatch.setattr(scheduler_module, "spill_cuda_tensors", _fake_spill)

    scheduler = _make_scheduler("tcp://0.0.0.0:30010")
    output_batch = OutputBatch(output=torch.ones(4))
    reply = _reply_output_batch(scheduler, output_batch)

    assert spill_calls == []
    assert isinstance(reply.output, torch.Tensor)
    assert torch.equal(reply.output, torch.ones(4))


def test_local_endpoint_still_spills_cuda_tensors(monkeypatch):
    spill_calls = []

    def _fake_spill(value, *, in_place=False):
        spill_calls.append(value)
        return value

    monkeypatch.setattr(scheduler_module, "spill_cuda_tensors", _fake_spill)

    scheduler = _make_scheduler("tcp://127.0.0.1:30010")
    output_batch = OutputBatch(output=torch.ones(4))
    _reply_output_batch(scheduler, output_batch)

    assert spill_calls == [output_batch]
