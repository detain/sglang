"""Detect — and prevent — Triton kernel device-loads after serving starts.

Triton loads each kernel specialization's cubin onto the GPU at its first
launch (``CompiledKernel._init_handles`` -> ``cuModuleLoadData``). That load
needs free device memory *outside* the torch caching allocator. Engines size
their pools to leave little post-init headroom, and the allocator's high-water
mark consumes the rest during early serving — so a specialization first used
mid-serving (e.g. a new adaptive speculative draft length, or a rare batch-size
bucket) can die in ``cuModuleLoadData`` with CUDA OOM, minutes or hours in.

``prewarm_device_loads()`` is the fix: called at engine init, it device-loads
every specialization it can find before the pools have taken the memory.
``install()`` / ``mark_serving_started()`` are the diagnostics for whatever
still slips through — once ``mark_serving_started()`` has been called, this
module warns when an uncached Triton compilation takes at least one second or a
device-load starts with less than ``SGLANG_TRITON_LOAD_WARNING_THRESHOLD_GB``
GiB of free device memory. Set
``SGLANG_CRASH_ON_TRITON_LOAD_AFTER_READY=1`` to raise on every late load
instead — for CI recipes that assert full startup warmup coverage. The hooks
only run for compilation and first-use loads, so steady-state cost is zero.

Note: request-driven warmup (``--warmups``, the server warmup request) runs
*after* ``mark_serving_started()`` and is subject to the same diagnostics;
crash mode is only meant for deployments whose kernels are fully pre-loaded at
engine init.
"""

from __future__ import annotations

import logging
import time
import weakref
from typing import Any, Callable, List, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.utils.common import get_available_gpu_memory

logger = logging.getLogger(__name__)

_serving_started = False
_prev_compile_listener = None
_installed = False
# install() ran and CompiledKernel construction is being recorded (independent
# of _installed: the tracker does not need the knobs hooks to exist).
_tracking_started = False
_prewarm_done = False
# weakref.ref -> CompiledKernel, appended by the __init__ wrap below. Weak on
# purpose: a process can construct thousands of kernels and the recorder must
# not be what keeps their tensors alive.
_kernel_refs: List[weakref.ref[Any]] = []


def install() -> None:
    """Install the diagnostics (idempotent; chains pre-existing hooks)."""
    global _installed, _prev_compile_listener
    if _installed:
        return
    try:
        import triton.knobs as knobs
    except ImportError:
        return

    # Recording kernels is a separate concern from the knobs hooks (and works on
    # builds where those do not exist), so do it before probing for them.
    _instrument_compiled_kernels()

    # Triton 3.4 builds used by ROCm expose triton.knobs without these hooks.
    runtime = getattr(knobs, "runtime", None)
    compilation = getattr(knobs, "compilation", None)
    hook = getattr(runtime, "kernel_load_start_hook", None)
    if hook is None or compilation is None or not hasattr(compilation, "listener"):
        return

    _prev_compile_listener = compilation.listener
    hook.add(_on_kernel_load)
    compilation.listener = _on_compilation
    _installed = True


def mark_serving_started() -> None:
    """Arm diagnostics for subsequent Triton compilations and device-loads."""
    global _serving_started
    _serving_started = True


def _on_compilation(*, src, metadata, metadata_group, times, cache_hit) -> None:
    if _prev_compile_listener is not None:
        _prev_compile_listener(
            src=src,
            metadata=metadata,
            metadata_group=metadata_group,
            times=times,
            cache_hit=cache_hit,
        )
    if not _serving_started or cache_hit:
        return

    compile_time_secs = times.total / 1e6
    if compile_time_secs < envs.SGLANG_TRITON_SLOW_COMPILE_THRESHOLD_SECS.get():
        return

    logger.warning(
        "Triton kernel '%s' took %.2f s to compile after serving started. "
        "Serving-time compilation can stall the engine; pre-compile it during "
        "engine init.",
        src.name,
        compile_time_secs,
    )


def _on_kernel_load(module, function, name, metadata_group, hash) -> None:
    if not _serving_started:
        return

    free_gb = _free_device_gb() if torch.cuda.is_available() else None

    should_crash = envs.SGLANG_CRASH_ON_TRITON_LOAD_AFTER_READY.get()
    if not should_crash and (
        free_gb is None or free_gb >= envs.SGLANG_TRITON_LOAD_WARNING_THRESHOLD_GB.get()
    ):
        return

    free_memory = f"{free_gb:.2f} GiB" if free_gb is not None else "unknown"
    msg = (
        f"Triton kernel '{name}' device-loaded after serving started "
        f"(free device mem: {free_memory}). Pre-load it during engine init "
        f"to avoid CUDA OOM."
    )
    if should_crash:
        raise RuntimeError(msg)
    logger.warning(msg)


def _free_device_gb() -> Optional[float]:
    """Free device memory in GiB, or None when it cannot be queried.

    ``empty_cache=False``: this runs on a hot path (the load hook) and next to
    live pools — flushing the allocator here would be a far bigger disturbance
    than the measurement is worth.
    """
    try:
        return get_available_gpu_memory(
            "cuda", torch.cuda.current_device(), empty_cache=False
        )
    except Exception:
        logger.debug("Unable to query free device memory", exc_info=True)
        return None


def _instrument_compiled_kernels() -> bool:
    """Record every ``CompiledKernel`` this process constructs.

    Triton builds that object lazily, at a specialization's first launch — which
    is exactly the moment a disk-cache hit would device-load it mid-serving. The
    original ``__init__`` is chained, never replaced, and only a weakref is kept.
    Returns False (and changes nothing) if the class or its layout differs.
    """
    global _tracking_started
    try:
        from triton.compiler.compiler import CompiledKernel
    except Exception:
        logger.debug("Triton prewarm: CompiledKernel is unavailable", exc_info=True)
        return False

    if getattr(CompiledKernel, "_sglang_prewarm_tracker", False):
        _tracking_started = True
        return True

    original_init: Callable[..., None] = CompiledKernel.__init__

    def _tracked_init(self, *args, **kwargs) -> None:
        original_init(self, *args, **kwargs)
        try:
            _kernel_refs.append(weakref.ref(self))
        except TypeError:  # pragma: no cover - CompiledKernel is weak-referenceable
            logger.debug(
                "Triton prewarm: kernel is not weak-referenceable", exc_info=True
            )

    try:
        CompiledKernel.__init__ = _tracked_init
        CompiledKernel._sglang_prewarm_tracker = True
    except Exception:
        logger.debug(
            "Triton prewarm: could not wrap CompiledKernel.__init__", exc_info=True
        )
        return False

    _tracking_started = True
    return True


def _jit_registry_kernels():
    """Compiled kernels reachable through Triton's JITFunction registry.

    Catches kernels that were constructed before ``install()`` ran (the wrap only
    sees constructions after it). Iterates ``defaultdict.values()`` — never
    indexes a device key, which would insert an empty cache for that device.
    """
    try:
        from triton.runtime.jit import _triton_jit_function_registry as registry
    except Exception:
        logger.debug(
            "Triton prewarm: JITFunction registry is unavailable", exc_info=True
        )
        return

    for jit_fn in list(registry.values()):
        device_caches = getattr(jit_fn, "device_caches", None)
        if device_caches is None:
            continue
        try:
            binders = list(device_caches.values())
        except Exception:
            continue
        for binder in binders:
            # (kernel_cache, kernel_key_cache, target, backend, binder)
            kernel_cache = binder[0] if isinstance(binder, tuple) else None
            if not isinstance(kernel_cache, dict):
                continue
            try:
                yield from list(kernel_cache.values())
            except Exception:
                continue


def _prune_dead_refs() -> None:
    _kernel_refs[:] = [ref for ref in _kernel_refs if ref() is not None]


def _tracked_kernels() -> List[Any]:
    """Deduplicated CompiledKernel objects this process could device-load."""
    _prune_dead_refs()
    kernels: List[Any] = []
    seen = set()
    for ref in _kernel_refs:
        kernel = ref()
        if kernel is None or id(kernel) in seen:
            continue
        seen.add(id(kernel))
        kernels.append(kernel)
    for kernel in _jit_registry_kernels():
        if id(kernel) in seen:
            continue
        seen.add(id(kernel))
        kernels.append(kernel)
    return kernels


def _is_resident(kernel: Any) -> bool:
    """True when the cubin is already loaded on the device.

    ``CompiledKernel._init_handles`` opens with ``if self.module is not None:
    return``, so a non-None ``module`` is Triton's own residency flag.
    """
    return getattr(kernel, "module", None) is not None


def _load_kernel(kernel: Any) -> Optional[str]:
    """Device-load one kernel. Returns None on success, else the reason."""
    init_handles = getattr(kernel, "_init_handles", None)
    if init_handles is None:
        return "no _init_handles attribute"
    try:
        init_handles()
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return None


# A kernel that will not load is worth shouting about; a hundred of them (an
# arch this GPU cannot run at all) is worth one line.
_MAX_FAILURE_LOGS = 10


def prewarm_device_loads(worker: Any = None) -> None:
    """Device-load Triton's cached cubins while engine init still owns the memory.

    Called once by the scheduler between ``install()`` and
    ``mark_serving_started()``, i.e. after weights/graph capture and before any
    request is served. Two mechanisms, in this order:

    A. Every ``CompiledKernel`` this process has constructed whose cubin is not
       yet resident gets ``_init_handles()`` called on it — the same call the
       first launch would make, just at a moment when losing is cheap.
    B. ``triton_prewarm_ops.prewarm_known_kernels`` dummy-launches the serving
       specializations that init never exercised. Triton has not even *built*
       the CompiledKernel object for those yet, so A cannot see them: in the
       2026-10-18/19 prod logs 92 late device-loads came with zero
       slow-compile warnings, i.e. every one was a disk-cache hit whose object
       was constructed by the serving launch itself.

    A failure here never fails boot: every step is individually guarded. The
    sweep stops (and says how much it skipped) once free device memory drops
    below ``SGLANG_TRITON_PREWARM_MIN_FREE_GB``, because the remaining kernels
    are then better off on the lazy path than crowding out the allocator.
    Idempotent: later calls are a no-op (the point is init-time work, and the
    registry only grows through the wrap, so a re-sweep would just re-walk it).
    """
    global _prewarm_done
    if _prewarm_done:
        return
    if not envs.SGLANG_ENABLE_TRITON_PREWARM_DEVICE_LOADS.get():
        logger.debug("Triton prewarm: disabled by environment")
        return
    if not _tracking_started:
        logger.debug("Triton prewarm: install() did not run; nothing to prewarm")
        return
    if not torch.cuda.is_available():
        # CPU CI, and the MLX/XPU paths where this module's CUDA assumptions do
        # not hold. Triton there uses a different driver entirely.
        return
    if getattr(torch.version, "hip", None) is not None:
        # ROCm: cuModuleLoadData is not what runs here, and the late-load
        # evidence is CUDA-only. Left on the lazy path deliberately.
        return

    _prewarm_done = True
    try:
        _run_prewarm(worker)
    except Exception as e:
        # Nothing in a prewarm may be fatal: the lazy path (with its warnings) is
        # the fallback for every failure mode, including a bug in this function.
        logger.warning(
            "Triton prewarm: sweep aborted (%s: %s); the remaining kernels stay "
            "on the lazy device-load path.",
            type(e).__name__,
            e,
        )


def _run_prewarm(worker: Any) -> None:
    started_at = time.perf_counter()
    min_free_gb = envs.SGLANG_TRITON_PREWARM_MIN_FREE_GB.get()
    free_start = _free_device_gb()

    loaded = resident = failed = skipped = 0
    kernels = _tracked_kernels()
    for index, kernel in enumerate(kernels):
        if _is_resident(kernel):
            resident += 1
            continue
        free_gb = _free_device_gb()
        if free_gb is not None and free_gb < min_free_gb:
            skipped = len(kernels) - index
            logger.warning(
                "Triton prewarm: stopping sweep at %d/%d kernels — only %.2f GiB "
                "free (floor %.2f GiB), %d kernel(s) left on the lazy device-load "
                "path.",
                index,
                len(kernels),
                free_gb,
                min_free_gb,
                skipped,
            )
            break
        error = _load_kernel(kernel)
        if error is None:
            loaded += 1
        else:
            failed += 1
            if failed <= _MAX_FAILURE_LOGS:
                logger.warning(
                    "Triton prewarm: could not device-load kernel '%s' (%s); it "
                    "stays on the lazy path.",
                    getattr(kernel, "name", "<unknown>"),
                    error,
                )

    elapsed = time.perf_counter() - started_at
    free_end = _free_device_gb()
    logger.info(
        "Triton prewarm: loaded %s cubins (%s already resident, %s skipped, free "
        "device mem %s -> %s GiB, %s s)",
        loaded,
        resident,
        skipped,
        _fmt_gb(free_start),
        _fmt_gb(free_end),
        f"{elapsed:.2f}",
    )
    if failed:
        detail = (
            ""
            if failed <= _MAX_FAILURE_LOGS
            else f" (first {_MAX_FAILURE_LOGS} logged)"
        )
        logger.warning(
            "Triton prewarm: %d kernel(s) failed to device-load%s.", failed, detail
        )

    # Mechanism B. Kept after A so the dummy launches inherit any cubin A already
    # resident-ed, and after the same headroom discipline: a sweep that hit the
    # floor is a sweep that must not start allocating more.
    free_gb = _free_device_gb()
    if free_gb is not None and free_gb < min_free_gb:
        logger.warning(
            "Triton prewarm: skipping known-kernel recipes (%.2f GiB free, floor %.2f GiB)",
            free_gb,
            min_free_gb,
        )
        return
    try:
        from sglang.srt.utils import triton_prewarm_ops

        triton_prewarm_ops.prewarm_known_kernels(
            torch.device("cuda", torch.cuda.current_device()), worker=worker
        )
    except Exception:
        logger.debug("Triton prewarm: known-kernel recipes failed", exc_info=True)


def _fmt_gb(value: Optional[float]) -> str:
    return f"{value:.2f}" if value is not None else "unknown"
