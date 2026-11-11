"""Boot-order regression: router registration must run after IPC channels exist.

The MERGE-6 era tip (b069e6a5ef) crashed every real restart:

    tokenizer_manager.py:667 init_preprocess_executor -> :958
    maybe_init_router_registration -> AttributeError: 'TokenizerManager'
    object has no attribute 'tokenizer_ipc_name'

Upstream (folded in by 4e5057aef9, PR #48119) calls maybe_init_router_registration
as the LAST step of __init__, after init_ipc_channels assigned tokenizer_ipc_name
and send_to_scheduler. The merge resolution moved the call into the fork-only
init_preprocess_executor, which the fork runs BEFORE init_ipc_channels. These
cases drive the real __init__ sequence with only the model/tokenizer loading and
the wire send stubbed, so the attribute-order contract is exercised, not mocked
away.
"""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import tokenizer_manager as tm  # noqa: E402
from sglang.srt.managers.io_struct import (  # noqa: E402
    TokenizerWorkerRegistrationReq,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402
from sglang.srt.runtime_context import get_context  # noqa: E402
from sglang.srt.server_args import PortArgs  # noqa: E402

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def _ipc_addr() -> str:
    fd, path = tempfile.mkstemp(prefix="sglang_m6bootfix_", suffix=".sock")
    os.close(fd)
    os.unlink(path)
    return f"ipc://{path}"


def _fake_init_model_config(self):
    """Skip HF config fetch; the boot path only reads these leaves."""
    self.model_config = SimpleNamespace(
        is_generation=True,
        context_len=2048,
        image_token_id=0,
        hf_config=None,
        dtype=None,
        joint_head_config=None,
        pointer_head=None,
    )
    self.is_generation = True
    self.context_len = 2048
    self.image_token_id = 0
    self.max_req_input_len = None


def _fake_init_tokenizer_and_processor(self):
    self._tokenizer = None
    self.mm_processor = None


class TestRouterRegistrationBootOrder(CustomTestCase):
    def _construct(self, *, tokenizer_worker_num: int):
        override = get_context().override_server_args(
            tokenizer_worker_num=tokenizer_worker_num
        )
        server_args = override.install()
        self.addCleanup(override.restore)
        # PortArgs.init_new reads raw ServerArgs struct slots, which the
        # override only publishes into the config bag; hand-build the record.
        multi = tokenizer_worker_num > 1
        addrs = [_ipc_addr() for _ in range(5 if multi else 4)]
        self.addCleanup(
            lambda: [
                os.unlink(a[len("ipc://") :])
                for a in addrs
                if os.path.exists(a[len("ipc://") :])
            ]
        )
        port_args = PortArgs(
            tokenizer_ipc_name=addrs[0],
            scheduler_input_ipc_name=addrs[1],
            detokenizer_ipc_name=addrs[2],
            nccl_port=0,
            rpc_ipc_name=addrs[3],
            metrics_ipc_name=addrs[3],
            tokenizer_worker_ipc_name=addrs[4] if multi else None,
            decoupled_spec_ipc_config=None,
            instance_id="m6bootfix",
        )
        dispatched = []
        with (
            # Process-entry provenance is not in scope for a unit boot.
            patch.object(tm, "assert_published", lambda sa, *, role: None),
            # Model/tokenizer loading is the heavy external dependency.
            patch.object(
                TokenizerManager, "init_model_config", _fake_init_model_config
            ),
            patch.object(
                TokenizerManager,
                "init_tokenizer_and_processor",
                _fake_init_tokenizer_and_processor,
            ),
            # SHM snapshot reader is an IPC-side detail, not this ordering.
            patch.object(tm, "create_load_snapshot_reader", MagicMock()),
            # Record the wire send instead of writing to a live socket.
            patch.object(tm, "sock_send", lambda sock, obj: dispatched.append(obj)),
        ):
            manager = TokenizerManager(server_args, port_args)
        # The process owns the ipc context in production; in a test, an abandoned
        # context blocks its __del__ term() forever at interpreter teardown.
        self.addCleanup(manager.send_to_scheduler.context.destroy, 0)
        return manager, port_args, dispatched

    def test_single_tokenizer_worker_skips_registration(self):
        """Router OFF: tokenizer_ipc_name is None -> no registration is sent."""
        manager, _port_args, dispatched = self._construct(tokenizer_worker_num=1)
        self.assertIsNone(manager.tokenizer_ipc_name)
        self.assertEqual(dispatched, [])

    def test_multi_tokenizer_worker_registers_after_ipc_init(self):
        """Router ON: registration is dispatched, after the IPC attrs exist."""
        manager, port_args, dispatched = self._construct(tokenizer_worker_num=2)
        self.assertEqual(manager.tokenizer_ipc_name, port_args.tokenizer_ipc_name)
        self.assertEqual(len(dispatched), 1)
        req = dispatched[0]
        self.assertIsInstance(req, TokenizerWorkerRegistrationReq)
        self.assertEqual(req.worker_ipc_name, port_args.tokenizer_ipc_name)


if __name__ == "__main__":
    unittest.main()
