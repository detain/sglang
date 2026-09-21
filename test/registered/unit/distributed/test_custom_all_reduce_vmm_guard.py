"""CustomAllreduce (v1) must not engage on a VMM-backed caching allocator.

v1 shares its CUDA graph input buffers with ``cudaIpcGetMemHandle``, which
only accepts cudaMalloc pointers. Under
``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` torch hands out
cuMemCreate/cuMemMap pointers instead and the handle call fails with
"invalid argument" at CUDA graph capture -- a boot crash, not a fallback.

CustomAllReduceV2 has a dedicated VMM path but is admitted only on full
NVLink, so a PCIe-only host with world_size == 2 is exactly the config that
falls through to v1. That combination is what these tests pin.
"""

from unittest.mock import Mock

from sglang.srt.distributed.device_communicators import custom_all_reduce
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _patch_v1(monkeypatch, *, uses_vmm):
    """Drive CustomAllreduce.__init__ up to the VMM guard without a GPU."""
    monkeypatch.setattr(custom_all_reduce, "_is_cuda", True)
    monkeypatch.setattr(custom_all_reduce, "_is_hip", False)
    monkeypatch.setattr(custom_all_reduce.ops, "IS_CUSTOM_AR_AVAILABLE", True)
    monkeypatch.setattr(custom_all_reduce.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(custom_all_reduce.dist, "get_world_size", lambda group: 2)
    # PCIe-only pair: v1 admits world_size == 2 without NVLink, which is what
    # makes this path reachable in the first place.
    monkeypatch.setattr(
        custom_all_reduce,
        "can_use_custom_all_reduce_with_nvlink",
        lambda **kwargs: False,
    )
    monkeypatch.setattr(
        custom_all_reduce,
        "is_vmm_backed_allocator",
        lambda device: uses_vmm,
    )
    created = []
    monkeypatch.setattr(
        custom_all_reduce.CustomAllreduce,
        "create_shared_buffer",
        staticmethod(lambda *a, **kw: created.append(a) or [0, 0]),
    )
    return created


def test_disabled_on_vmm_backed_allocator(monkeypatch):
    """expandable_segments -> stay disabled, allocate nothing, fall back to NCCL."""
    created = _patch_v1(monkeypatch, uses_vmm=True)

    comm = custom_all_reduce.CustomAllreduce(group=Mock(), device="cuda:0")

    assert comm.disabled is True
    assert comm.original_disabled is True
    # Bailing out before create_shared_buffer is the point: those buffers are
    # IPC-shared across ranks and leak if __init__ returns half-built.
    assert created == []
    assert not hasattr(comm, "_ptr")


def test_guard_does_not_fire_on_cuda_malloc_allocator(monkeypatch):
    """Default allocator -> the guard is transparent and v1 proceeds."""
    created = _patch_v1(monkeypatch, uses_vmm=False)

    try:
        custom_all_reduce.CustomAllreduce(group=Mock(), device="cuda:0")
    except Exception:
        # Past the guard the init needs a real GPU and real IPC handles, so it
        # may or may not get all the way through depending on the runner. How
        # far it got is what create_shared_buffer below records.
        pass

    assert created, "guard must not short-circuit a cudaMalloc-backed allocator"
