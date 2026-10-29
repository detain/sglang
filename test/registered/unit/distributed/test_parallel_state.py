"""
Test file to verify the correctness of parallel group calculations.

This test validates that the parallel group initialization creates the correct
groups for different parallelism configurations including:
- Tensor parallelism (TP)
- Pipeline parallelism (PP)
- Attention context parallelism (attn_cp)
- Attention data parallelism (attn_dp)
- MoE expert parallelism (EP)
- MoE data parallelism (moe_dp)

These tests call the ACTUAL initialize_model_parallel() function with mocked
distributed backend to verify the group construction logic.

## How These Tests Work

initialize_model_parallel() creates ALL groups for ALL ranks in a single call.
For example, when creating TP groups with tp_size=2 and world_size=8:

    group_ranks = [[0,1], [2,3], [4,5], [6,7]]  # ALL groups created
    _TP = init_model_parallel_group(group_ranks, local_rank, ...)

ALL ranks call this function and get the same complete group structure. Each rank
then figures out which specific group(s) it belongs to.

Our tests:
1. Mock the distributed backend (no real GPUs needed)
2. Mock GroupCoordinator construction to capture the effective group configuration
3. Call the real initialize_model_parallel()
4. Verify the resulting group structure and policies

We only need to simulate rank 0 because we're testing the group creation logic,
not the per-rank group membership logic.
"""

from __future__ import annotations

import inspect
import logging
import sys
from contextlib import nullcontext
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import publish_build_topology

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

# Import the actual parallel_state module
parallel_state = pytest.importorskip("sglang.srt.distributed.parallel_state")


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("inplace_allreduce", [False, True])
@pytest.mark.parametrize("alias_output", [False, True])
def test_deterministic_reduce_scatter_preserves_input_and_selects_rank_shard(
    monkeypatch, rank, inplace_allreduce, alias_output
):
    monkeypatch.setenv("SGLANG_ENABLE_DETERMINISTIC_INFERENCE", "1")
    coordinator = parallel_state.GroupCoordinator.__new__(
        parallel_state.GroupCoordinator
    )
    coordinator.rank_in_group = rank
    coordinator.world_size = 4
    # Flattened input is a valid reduce-scatter layout too.
    input_ = torch.arange(24, dtype=torch.float32)
    original = input_.clone()
    output = input_.view(4, 2, 3)[rank] if alias_output else torch.empty((2, 3))
    reduced = original + 100

    def all_reduce(tensor):
        assert tensor.data_ptr() != input_.data_ptr()
        torch.testing.assert_close(tensor, original)
        if inplace_allreduce:
            tensor.copy_(reduced)
            return tensor
        return reduced

    coordinator.all_reduce = Mock(side_effect=all_reduce)
    coordinator._reduce_scatter_tensor = Mock(side_effect=AssertionError)
    coordinator.reduce_scatter_tensor(output, input_)
    torch.testing.assert_close(output, reduced.view(4, 2, 3)[rank])
    if alias_output:
        original.view(4, 2, 3)[rank].copy_(output)
    torch.testing.assert_close(input_, original)
    coordinator.all_reduce.assert_called_once()


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("sizes", [None, [1, 3, 2, 0]])
def test_deterministic_reduce_scatterv_selects_rank_shard(monkeypatch, rank, sizes):
    monkeypatch.setenv("SGLANG_ENABLE_DETERMINISTIC_INFERENCE", "1")
    coordinator = parallel_state.GroupCoordinator.__new__(
        parallel_state.GroupCoordinator
    )
    coordinator.rank_in_group = rank
    coordinator.world_size = 4
    rows = 8 if sizes is None else sum(sizes)
    input_ = torch.arange(rows * 3, dtype=torch.float32).view(rows, 3)
    original = input_.clone()
    reduced = original + 100

    def all_reduce(tensor):
        assert tensor.data_ptr() != input_.data_ptr()
        torch.testing.assert_close(tensor, original)
        return reduced

    coordinator.all_reduce = Mock(side_effect=all_reduce)
    # pynccl must not be touched on the deterministic path.
    coordinator.pynccl_comm = None
    output = coordinator.reduce_scatterv(input_, sizes=sizes)

    offset = (rows // 4) * rank if sizes is None else sum(sizes[:rank])
    chunk = rows // 4 if sizes is None else sizes[rank]
    torch.testing.assert_close(output, reduced.narrow(0, offset, chunk))
    torch.testing.assert_close(input_, original)
    coordinator.all_reduce.assert_called_once()


def test_nondeterministic_reduce_scatter_keeps_native_path(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_DETERMINISTIC_INFERENCE", "0")
    monkeypatch.setattr(parallel_state, "_is_cpu", True)
    coordinator = parallel_state.GroupCoordinator.__new__(
        parallel_state.GroupCoordinator
    )
    coordinator._reduce_scatter_tensor = Mock()
    coordinator.all_reduce = Mock(side_effect=AssertionError)
    input_ = torch.arange(8, dtype=torch.float32)
    output = torch.empty(2)
    coordinator.reduce_scatter_tensor(output, input_)
    coordinator._reduce_scatter_tensor.assert_called_once_with(output, input_)


def test_custom_allreduce_precedes_symmetric_memory_pynccl():
    coordinator = parallel_state.GroupCoordinator.__new__(
        parallel_state.GroupCoordinator
    )
    coordinator.world_size = 2
    coordinator.unique_name = "test"
    # PR #39605 added these all_reduce() pre-checks; the stub must carry them.
    coordinator.host_staged_comm = None
    coordinator.ll_comm = None
    coordinator.hpu_communicator = None
    coordinator.xpu_communicator = None
    coordinator.qr_comm = None
    coordinator.pymscclpp_comm = None
    coordinator.torch_symm_mem_comm = None
    coordinator._fi_workspace_hint = None
    coordinator.ca_comm = Mock(disabled=False)
    coordinator.ca_comm.should_custom_ar.return_value = True
    coordinator.pynccl_comm = Mock()
    coordinator.pynccl_comm.change_state.return_value = nullcontext()
    coordinator.is_symmetric_memory_enabled = Mock(return_value=True)
    coordinator.debug_check_symmetric_mempool = Mock()
    input_ = Mock(is_cpu=False)
    custom_output = object()

    with (
        patch.object(parallel_state.torch.compiler, "is_compiling", return_value=False),
        patch.object(
            parallel_state, "outplace_all_reduce", return_value=custom_output
        ) as outplace_all_reduce,
    ):
        output = coordinator.all_reduce(input_)

    assert output is custom_output
    outplace_all_reduce.assert_called_once_with(
        input_,
        group_name="test",
        outplace_all_reduce_method="ca",
    )
    coordinator.pynccl_comm.all_reduce.assert_not_called()


@pytest.mark.parametrize("custom_allreduce_enabled", [False, True])
def test_parallel_group_construction_tp8_attn_cp2(custom_allreduce_enabled):
    """
    Test parallel group construction for 8 GPU configuration with:
    - tensor_model_parallel_size = 8
    - attention_context_model_parallel_size = 2

    Expected groups based on docstring example:
        1 tensor model-parallel group:
            [g0, g1, g2, g3, g4, g5, g6, g7]
        4 attention context-parallel groups:
            [g0, g4], [g1, g5], [g2, g6], [g3, g7]

    This test calls the ACTUAL initialize_model_parallel() and verifies the groups.

    Note: We simulate only rank 0 here, but initialize_model_parallel() creates
    ALL groups for ALL ranks in a single call. We capture GroupCoordinator
    construction and verify the complete group structure and effective policy.
    """
    world_size = 8

    # Mock the distributed backend
    # Note: get_rank() returns 0 because we're testing from a single process,
    # but initialize_model_parallel() still creates all groups for all ranks
    with (
        patch.object(parallel_state, "_WORLD", None),
        patch.object(parallel_state, "_TP", None),
        patch.object(parallel_state, "_ATTN_CP", None),
        patch.object(parallel_state, "_ATTN_TP", None),
        patch.object(parallel_state, "_PP", None),
        patch.object(
            parallel_state,
            "_ENABLE_CUSTOM_ALL_REDUCE",
            custom_allreduce_enabled,
        ),
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=world_size),
        patch("torch.distributed.get_rank", return_value=0),
        patch("torch.distributed.get_backend", return_value="nccl"),
    ):
        # Capture the effective configuration after init_model_parallel_group
        # has resolved inherited policies.
        created_groups = {}

        def mock_group_coordinator(
            *, group_ranks, group_name, use_custom_allreduce, **kwargs
        ):
            created_groups[group_name] = group_ranks

            mock_group = Mock()
            mock_group.device_group = Mock()
            mock_group.use_custom_allreduce = use_custom_allreduce
            return mock_group

        with (
            patch.object(
                parallel_state,
                "GroupCoordinator",
                side_effect=mock_group_coordinator,
            ),
            patch.object(parallel_state, "get_world_group") as mock_world_group,
        ):
            # Mock world group
            mock_world = Mock()
            mock_world.device_group = Mock()
            mock_world.local_rank = 0
            mock_world_group.return_value = mock_world

            # Call the actual function
            publish_build_topology(tp_size=8, pp_size=1, attn_cp_size=2)
            parallel_state.initialize_model_parallel()

            # Verify TP groups
            tp_groups = created_groups.get("tp", [])
            assert len(tp_groups) == 1, f"Expected 1 TP group, got {len(tp_groups)}"
            assert tp_groups[0] == [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
            ], f"Wrong TP group: {tp_groups[0]}"

            # Verify ATTN_CP groups
            attn_cp_groups = created_groups.get("attn_cp", [])
            assert len(attn_cp_groups) == 4, (
                f"Expected 4 ATTN_CP groups, got {len(attn_cp_groups)}"
            )
            expected_attn_cp = [
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
            ]
            assert attn_cp_groups == expected_attn_cp, (
                f"Wrong ATTN_CP groups: {attn_cp_groups}"
            )

            # Derived attention-TP groups inherit the effective global policy;
            # the communicator performs the final per-group capability check.
            assert (
                parallel_state.get_attn_tp_group().use_custom_allreduce
                is custom_allreduce_enabled
            )

            print("TP=8, Attn CP=2 group construction verified")

            # Cleanup
            parallel_state.destroy_model_parallel()


def test_parallel_group_construction_tp8_moe_ep4_cp2():
    """
    Test parallel group construction for 8 GPU configuration with:
    - tensor_model_parallel_size = 8
    - expert_model_parallel_size = 4
    - moe_data_model_parallel_size = 2

    Expected groups:
        1 tensor model-parallel group:
            [g0, g1, g2, g3, g4, g5, g6, g7]
        2 MoE expert-parallel groups:
            [g0, g1, g2, g3], [g4, g5, g6, g7]
        4 MoE data-parallel groups:
            [g0, g4], [g1, g5], [g2, g6], [g3, g7]
    """
    world_size = 8

    # Mock the distributed backend
    with (
        patch.object(parallel_state, "_WORLD", None),
        patch.object(parallel_state, "_TP", None),
        patch.object(parallel_state, "_MOE_EP", None),
        patch.object(parallel_state, "_MOE_DP", None),
        patch.object(parallel_state, "_MOE_TP", None),
        patch.object(parallel_state, "_PP", None),
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=world_size),
        patch("torch.distributed.get_rank", return_value=0),
        patch("torch.distributed.get_backend", return_value="nccl"),
    ):
        # Mock init_model_parallel_group to capture the groups being created
        created_groups = {}

        def mock_init_model_parallel_group(group_ranks, local_rank, backend, **kwargs):
            group_name = kwargs.get("group_name", "unknown")
            created_groups[group_name] = group_ranks

            # Create a mock group object
            mock_group = Mock()
            mock_group.device_group = Mock()
            return mock_group

        with (
            patch.object(
                parallel_state,
                "init_model_parallel_group",
                side_effect=mock_init_model_parallel_group,
            ),
            patch.object(parallel_state, "get_world_group") as mock_world_group,
        ):
            # Mock world group
            mock_world = Mock()
            mock_world.device_group = Mock()
            mock_world.local_rank = 0
            mock_world_group.return_value = mock_world

            # Call the actual function
            publish_build_topology(tp_size=8, ep_size=4, pp_size=1, moe_dp_size=2)
            parallel_state.initialize_model_parallel()

            # Verify TP groups
            tp_groups = created_groups.get("tp", [])
            assert len(tp_groups) == 1, f"Expected 1 TP group, got {len(tp_groups)}"
            assert tp_groups[0] == [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
            ], f"Wrong TP group: {tp_groups[0]}"

            # Verify MOE_EP groups
            moe_ep_groups = created_groups.get("moe_ep", [])
            assert len(moe_ep_groups) == 2, (
                f"Expected 2 MOE_EP groups, got {len(moe_ep_groups)}"
            )
            expected_moe_ep = [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
            ]
            assert moe_ep_groups == expected_moe_ep, (
                f"Wrong MOE_EP groups: {moe_ep_groups}"
            )

            # Verify MOE_DP groups
            moe_dp_groups = created_groups.get("moe_dp", [])
            assert len(moe_dp_groups) == 4, (
                f"Expected 4 MOE_DP groups, got {len(moe_dp_groups)}"
            )
            expected_moe_dp = [
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
            ]
            assert moe_dp_groups == expected_moe_dp, (
                f"Wrong MOE_DP groups: {moe_dp_groups}"
            )

            print("TP=8, MoE EP=4, MoE CP=2 group construction verified")

            # Cleanup
            parallel_state.destroy_model_parallel()


def _read_group_descs(group_name):
    """Build a real ``GroupCoordinator`` over a single-rank gloo world and read the
    ``group_desc`` back off the live ProcessGroup objects it created.

    Returns ``(device_group.group_desc, cpu_group.group_desc)``.

    Unlike a mocked ``new_group``, this drives the *actual*
    ``torch.distributed.new_group`` in the normal (NCCL/Gloo) branch, so:

    * it fails if the installed PyTorch does not accept the ``group_desc`` kwarg
      -- the only real risk of this change; and
    * it asserts on the value read back from the constructed ProcessGroup, not on
      the string handed to a patched ``new_group`` (which would just mirror the
      implementation).

    gloo + ``world_size=1`` needs no GPU/NCCL and no extra process, so this stays
    in the CPU suite. The mooncake branch is structurally identical but needs a
    built ``mooncake`` backend that the CPU runner does not have, so it is not
    exercised separately here.
    """
    import torch.distributed as dist

    dist.init_process_group(
        backend="gloo", store=dist.HashStore(), rank=0, world_size=1
    )
    try:
        coord = parallel_state.GroupCoordinator(
            group_ranks=[[0]],
            local_rank=0,
            torch_distributed_backend="gloo",
            use_pynccl=False,
            use_mscclpp=False,
            use_custom_allreduce=False,
            use_torch_symm_mem_all_reduce=False,
            use_hpu_communicator=False,
            use_xpu_communicator=False,
            use_message_queue_broadcaster=False,
            group_name=group_name,
        )
        return coord.device_group.group_desc, coord.cpu_group.group_desc
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("group_name", ["tp", "pp"])
def test_group_desc_propagated_via_real_new_group(group_name):
    """Regression guard: ``GroupCoordinator`` must tag the ProcessGroups it creates
    with ``group_desc=f"{group_name}:{device|cpu}"``.

    This is the metadata NCCL Inspector reads to distinguish TP/PP communicators;
    dropping it in the ``new_group`` calls reintroduces the PP-misclassified-as-TP
    bug. Read back from the real ProcessGroups so a real ``new_group`` actually
    accepts and stores the kwarg.
    """
    assert _read_group_descs(group_name) == (
        f"{group_name}:device",
        f"{group_name}:cpu",
    )


def test_group_desc_none_normalized_to_anonymous():
    """group_name=None keeps the existing "anonymous" normalization, so the
    emitted descs must be anonymous:device / anonymous:cpu (not None:device)."""
    assert _read_group_descs(None) == ("anonymous:device", "anonymous:cpu")


# ---------------------------------------------------------------------------
# LLGUARD: refuse the LL allreduce on a group PCIe-IPC arms. Fork guard over
# the entry-256 port of PR #39605; there is no upstream test to mirror, so
# these follow the stub-coordinator style of this file and the sys.modules
# stand-in style of test_pcie_ipc_ar.py.
# ---------------------------------------------------------------------------

_LL_LOGGER = "sglang.srt.distributed.parallel_state"
_LL_MODULE = "sglang.srt.distributed.device_communicators.ll_allreduce"


def _ll_stub_coordinator(pcie_ipc_eligible: bool, world_size: int = 2):
    """A GroupCoordinator without __init__: the LL gate needs no live groups."""
    coordinator = parallel_state.GroupCoordinator.__new__(
        parallel_state.GroupCoordinator
    )
    coordinator.world_size = world_size
    coordinator.pcie_ipc_eligible = pcie_ipc_eligible
    coordinator.rank_in_group = 1
    coordinator.cpu_group = Mock(name="cpu_group")
    coordinator.device = torch.device("cpu")
    return coordinator


def _ll_module_stub():
    """Stand in for ll_allreduce, whose extension needs CUDA to build."""
    module = Mock(name="ll_allreduce")
    comm = Mock(name="ll_comm")
    module.LLAllReduce.return_value = comm
    return module, comm


def _ll_warnings(caplog):
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == _LL_LOGGER and r.levelno >= logging.WARNING
    ]


def test_ll_allreduce_refused_when_pcie_ipc_armed(monkeypatch, caplog):
    """SGLANG_LL_ALLREDUCE=1 must not build LL on a group PCIe-IPC arms.

    Both claim the <=64 KiB captured-bf16 size class, and prod run scripts
    export SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1 today, so a future LL arming
    would silently double-claim it. The refusal warns naming BOTH envs and
    leaves the LL communicator unbuilt -- not deferred, not disabled later.
    """
    monkeypatch.setenv("SGLANG_LL_ALLREDUCE", "1")
    module, _ = _ll_module_stub()
    coordinator = _ll_stub_coordinator(pcie_ipc_eligible=True)
    with (
        patch.dict(sys.modules, {_LL_MODULE: module}),
        patch.object(parallel_state, "is_cuda_alike", return_value=True),
        caplog.at_level(logging.WARNING, logger=_LL_LOGGER),
    ):
        assert coordinator._build_ll_allreduce("tp") is None
    module.LLAllReduce.assert_not_called()
    text = "\n".join(_ll_warnings(caplog))
    assert "SGLANG_LL_ALLREDUCE" in text
    assert "SGLANG_ENABLE_PCIE_IPC_ALLREDUCE" in text


def test_ll_allreduce_still_builds_when_pcie_ipc_off(monkeypatch, caplog):
    """The guard is inert when PCIe-IPC does not arm the group: entry-256
    behavior -- LL builds on two CUDA-alike ranks -- is unchanged, and the
    collision warning stays silent."""
    monkeypatch.setenv("SGLANG_LL_ALLREDUCE", "1")
    module, comm = _ll_module_stub()
    coordinator = _ll_stub_coordinator(pcie_ipc_eligible=False)
    with (
        patch.dict(sys.modules, {_LL_MODULE: module}),
        patch.object(parallel_state, "is_cuda_alike", return_value=True),
        caplog.at_level(logging.WARNING, logger=_LL_LOGGER),
    ):
        assert coordinator._build_ll_allreduce("tp") is comm
    module.LLAllReduce.assert_called_once_with(
        coordinator.cpu_group, 1, 2, coordinator.device
    )
    assert _ll_warnings(caplog) == []


@pytest.mark.parametrize(
    "env_value,world_size,pcie_eligible",
    [(None, 2, True), ("", 2, True), ("0", 2, True), ("1", 1, False), ("1", 4, True)],
)
def test_guard_inert_when_ll_was_never_requested(
    monkeypatch, caplog, env_value, world_size, pcie_eligible
):
    """No warning and no construction unless the env asked for LL on two
    GPUs: the refusal message is for an operator who armed a collision, not
    for every group boot. The PCIe-armed rows are the adversarial ones --
    the LL gate must fail on its own terms before the collision check."""
    if env_value is None:
        monkeypatch.delenv("SGLANG_LL_ALLREDUCE", raising=False)
    else:
        monkeypatch.setenv("SGLANG_LL_ALLREDUCE", env_value)
    module, _ = _ll_module_stub()
    coordinator = _ll_stub_coordinator(
        pcie_ipc_eligible=pcie_eligible, world_size=world_size
    )
    with (
        patch.dict(sys.modules, {_LL_MODULE: module}),
        patch.object(parallel_state, "is_cuda_alike", return_value=True),
        caplog.at_level(logging.WARNING, logger=_LL_LOGGER),
    ):
        assert coordinator._build_ll_allreduce("tp") is None
    module.LLAllReduce.assert_not_called()
    assert _ll_warnings(caplog) == []


def test_ll_env_acceptance_is_env_bool(monkeypatch):
    """ENVREG widened the gate from the raw == "1" read to the registered
    EnvBool: 1/true/yes/y (case-insensitive) all arm, anything falsy or junk
    does not (junk resolves to the default with a warning). The widening is
    the safer superset -- an operator who wrote SGLANG_LL_ALLREDUCE=true
    meant it on, and the PCIe-IPC collision guard applies to every spelling.
    """
    for value, armed in (
        ("1", True),
        ("true", True),
        ("TRUE", True),
        ("yes", True),
        ("y", True),
        ("0", False),
        ("false", False),
        ("no", False),
    ):
        monkeypatch.setenv("SGLANG_LL_ALLREDUCE", value)
        module, comm = _ll_module_stub()
        coordinator = _ll_stub_coordinator(pcie_ipc_eligible=False)
        with (
            patch.dict(sys.modules, {_LL_MODULE: module}),
            patch.object(parallel_state, "is_cuda_alike", return_value=True),
        ):
            built = coordinator._build_ll_allreduce("tp")
        assert (built is comm) is armed, value


def test_comm_envs_are_registered_not_raw(monkeypatch):
    """The entry-256 fork-convention deviation is closed: the three reads
    live in environ.py, so an environ-only grep finds them, and the
    construction sites read the handles. Mocks cannot catch a reverting to
    raw os.environ.get -- pin the sources.
    """
    from sglang.srt.distributed.device_communicators import ll_allreduce
    from sglang.srt.environ import envs

    assert envs.SGLANG_LL_ALLREDUCE.default is False
    assert envs.SGLANG_HOST_STAGED_ALLREDUCE.default is False
    # The EnvInt default equals the raw fallback str(64 << 10) it replaced.
    assert envs.SGLANG_LL_ALLREDUCE_MAX_BYTES.default == 64 << 10

    gate_src = inspect.getsource(parallel_state.GroupCoordinator._build_ll_allreduce)
    init_src = inspect.getsource(parallel_state.GroupCoordinator.__init__)
    ll_src = inspect.getsource(ll_allreduce.LLAllReduce.__init__)
    for name in ("SGLANG_LL_ALLREDUCE", "SGLANG_HOST_STAGED_ALLREDUCE"):
        assert f'os.environ.get("{name}")' not in gate_src + init_src
    assert "envs.SGLANG_LL_ALLREDUCE_MAX_BYTES.get()" in ll_src
    assert 'os.environ.get("SGLANG_LL_ALLREDUCE_MAX_BYTES"' not in ll_src


def test_host_transport_needle_names_which_transport_won(caplog):
    """Boot-verify: one INFO line per armed group naming the trio's state, so
    operators can read a log line, not the env, for who serves the group."""
    coordinator = _ll_stub_coordinator(pcie_ipc_eligible=True)
    coordinator.host_staged_comm = None
    coordinator.ll_comm = None  # the guard refused it on this armed group
    with caplog.at_level(logging.INFO, logger=_LL_LOGGER):
        coordinator._log_host_transport_state("tp")
    assert (
        "[AR] group tp host-memory transports: "
        "PCIe-IPC=True host-staged=False LL=False" in caplog.text
    )


def test_host_transport_needle_shows_ll_serving_the_group(caplog):
    coordinator = _ll_stub_coordinator(pcie_ipc_eligible=False)
    coordinator.host_staged_comm = None
    coordinator.ll_comm = Mock(name="ll_comm")
    with caplog.at_level(logging.INFO, logger=_LL_LOGGER):
        coordinator._log_host_transport_state("tp")
    assert "PCIe-IPC=False host-staged=False LL=True" in caplog.text


def test_host_transport_needle_absent_when_nothing_is_armed(caplog):
    """Default boots stay quiet: nothing to verify when no opt-in transport
    claims the group."""
    coordinator = _ll_stub_coordinator(pcie_ipc_eligible=False)
    coordinator.host_staged_comm = None
    coordinator.ll_comm = None
    with caplog.at_level(logging.INFO, logger=_LL_LOGGER):
        coordinator._log_host_transport_state("tp")
    assert "host-memory transports" not in caplog.text


def test_ll_construction_and_needle_are_wired_into_init():
    """The stubs above cannot catch __init__ dropping the guard or the needle
    call. The repo's own test_the_deterministic_predicate_still_exists pins a
    predicate's existence for exactly this blind spot; pin both call sites."""
    init_src = inspect.getsource(parallel_state.GroupCoordinator.__init__)
    assert "self._build_ll_allreduce(group_name)" in init_src
    assert "self._log_host_transport_state(group_name)" in init_src


if __name__ == "__main__":
    # Run tests without requiring GPUs
    import sys

    try:
        test_parallel_group_construction_tp8_attn_cp2(False)
        test_parallel_group_construction_tp8_attn_cp2(True)
        test_parallel_group_construction_tp8_moe_ep4_cp2()
        test_group_desc_propagated_via_real_new_group("tp")
        test_group_desc_propagated_via_real_new_group("pp")
        test_group_desc_none_normalized_to_anonymous()

        sys.exit(0)
    except AssertionError as e:
        print(f"\n Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
