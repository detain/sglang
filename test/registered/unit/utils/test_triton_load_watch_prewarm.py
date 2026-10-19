"""Unit tests for the Triton device-load prewarm in srt/utils/triton_load_watch.

CPU-only: the sweep is driven with fake ``CompiledKernel`` look-alikes and a
faked free-memory query, and the CUDA gates are patched, so no GPU and no real
Triton kernel is needed.
"""

import gc
import re
import unittest
import weakref
from collections import defaultdict
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import triton_load_watch as watch
from sglang.srt.utils import triton_prewarm_ops
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_LOGGER = "sglang.srt.utils.triton_load_watch"

_STATE = (
    "_serving_started",
    "_prev_compile_listener",
    "_installed",
    "_tracking_started",
    "_prewarm_done",
)


class _FakeKernel:
    """Stand-in for triton's CompiledKernel: same residency contract."""

    def __init__(self, name, *, resident=False, raises=False):
        self.name = name
        self.module = "resident" if resident else None
        self.raises = raises
        self.init_calls = 0

    def _init_handles(self):
        self.init_calls += 1
        if self.raises:
            raise RuntimeError("cuModuleLoadData OOM")
        self.module = "loaded"


class _FakeHook:
    def __init__(self):
        self.added = []

    def add(self, fn):
        self.added.append(fn)


class TritonPrewarmTestBase(CustomTestCase):
    def setUp(self):
        self._saved = {name: getattr(watch, name) for name in _STATE}
        self._saved_refs = list(watch._kernel_refs)
        for name in _STATE:
            setattr(watch, name, None if name == "_prev_compile_listener" else False)
        watch._kernel_refs.clear()

        # Mechanism B is a separate module; never launch real kernels here.
        recipes = patch.object(
            triton_prewarm_ops, "prewarm_known_kernels", return_value=[]
        )
        self.recipes = recipes.start()
        self.addCleanup(recipes.stop)
        # Keep the sweep deterministic: only the fakes under test are walked.
        # Tests of the scan itself use self.real_jit_scan.
        self.real_jit_scan = watch._jit_registry_kernels
        jit = patch.object(watch, "_jit_registry_kernels", return_value=iter(()))
        jit.start()
        self.addCleanup(jit.stop)

    def tearDown(self):
        for name, value in self._saved.items():
            setattr(watch, name, value)
        watch._kernel_refs[:] = self._saved_refs

    def add_kernel(self, kernel):
        """Record a kernel the way the CompiledKernel wrap does."""
        watch._kernel_refs.append(weakref.ref(kernel))
        return kernel

    def arm_cuda(self, memory_query=None):
        """Make the module's CUDA gates pass and drive the memory query."""
        cuda = patch.object(watch.torch.cuda, "is_available", return_value=True)
        cuda.start()
        self.addCleanup(cuda.stop)
        device = patch.object(watch.torch.cuda, "current_device", return_value=0)
        device.start()
        self.addCleanup(device.stop)
        if memory_query is not None:
            query = patch.object(
                watch, "get_available_gpu_memory", side_effect=memory_query
            )
            query.start()
            self.addCleanup(query.stop)
        watch._tracking_started = True


class TestPrewarmSweep(TritonPrewarmTestBase):
    def test_loads_unloaded_and_skips_resident(self):
        self.arm_cuda([8.0] * 32)
        unloaded = self.add_kernel(_FakeKernel("unloaded"))
        resident = self.add_kernel(_FakeKernel("resident", resident=True))

        watch.prewarm_device_loads()

        self.assertEqual(unloaded.init_calls, 1)
        self.assertIsNotNone(unloaded.module)
        self.assertEqual(resident.init_calls, 0, "a resident cubin must be left alone")
        self.recipes.assert_called_once()

    def test_survives_raising_init_handles(self):
        self.arm_cuda([8.0] * 32)
        bad = self.add_kernel(_FakeKernel("bad", raises=True))
        good = self.add_kernel(_FakeKernel("good"))

        with self.assertLogs(_LOGGER, level="WARNING") as logs:
            watch.prewarm_device_loads()

        self.assertIsNotNone(good.module, "one bad kernel must not abort the sweep")
        self.assertEqual(bad.module, None)
        output = "\n".join(logs.output)
        self.assertIn("could not device-load kernel 'bad'", output)
        self.assertIn("cuModuleLoadData OOM", output, "the reason must be in the log")
        self.assertIn("1 kernel(s) failed", output)

    def test_respects_min_free_floor(self):
        # free_start, k0 check, the floor trip, free_end, the recipe gate.
        self.arm_cuda([8.0, 8.0, 0.5, 0.5, 0.5])
        first = self.add_kernel(_FakeKernel("first"))
        second = self.add_kernel(_FakeKernel("second"))
        third = self.add_kernel(_FakeKernel("third"))

        with self.assertLogs(_LOGGER, level="WARNING") as logs:
            watch.prewarm_device_loads()

        self.assertIsNotNone(first.module)
        self.assertIsNone(second.module, "the sweep must stop once below the floor")
        self.assertIsNone(third.module)
        output = "\n".join(logs.output)
        self.assertIn("stopping sweep", output)
        self.assertIn(
            "2 kernel(s) left", output, "the log must name how many were skipped"
        )
        self.recipes.assert_not_called()

    def test_floor_is_configurable(self):
        self.arm_cuda([8.0, 0.5, 0.5, 0.5])
        kernel = self.add_kernel(_FakeKernel("k"))
        with envs.SGLANG_TRITON_PREWARM_MIN_FREE_GB.override(0.1):
            watch.prewarm_device_loads()
        self.assertIsNotNone(kernel.module, "0.5 GiB free is above a 0.1 GiB floor")

    def test_summary_line_is_always_logged(self):
        self.arm_cuda([8.0] * 32)
        loaded_kernel = self.add_kernel(_FakeKernel("a"))
        resident_kernel = self.add_kernel(_FakeKernel("b", resident=True))

        with self.assertLogs(_LOGGER, level="INFO") as logs:
            watch.prewarm_device_loads()

        pattern = re.compile(
            r"Triton prewarm: loaded (\d+) cubins \((\d+) already resident, "
            r"(\d+) skipped, free device mem (\d+\.\d+) -> (\d+\.\d+) GiB, "
            r"(\d+\.\d+) s\)"
        )
        summary = next(
            (m for m in (pattern.search(line) for line in logs.output) if m), None
        )
        self.assertIsNotNone(summary, logs.output)
        self.assertEqual(summary.groups()[:3], ("1", "1", "0"))
        self.assertIsNotNone(loaded_kernel.module)
        self.assertEqual(resident_kernel.init_calls, 0)

    def test_summary_line_logged_with_no_kernels(self):
        self.arm_cuda([8.0] * 8)
        with self.assertLogs(_LOGGER, level="INFO") as logs:
            watch.prewarm_device_loads()
        self.assertIn("Triton prewarm: loaded 0 cubins", "\n".join(logs.output))

    def test_internal_error_never_escapes(self):
        # A bug in the sweep itself must not fail boot.
        self.arm_cuda([8.0] * 32)
        self.add_kernel(_FakeKernel("k"))
        with patch.object(watch, "_tracked_kernels", side_effect=RuntimeError("boom")):
            with self.assertLogs(_LOGGER, level="WARNING") as logs:
                watch.prewarm_device_loads()
        self.assertIn("sweep aborted", "\n".join(logs.output))
        self.assertTrue(watch._prewarm_done)

    def test_unknown_free_memory_still_loads(self):
        # A failing memory query must not block the prewarm: it is best-effort,
        # and the alternative is losing the whole protection.
        self.arm_cuda()
        kernel = self.add_kernel(_FakeKernel("k"))
        with patch.object(watch, "_free_device_gb", return_value=None):
            watch.prewarm_device_loads()
        self.assertIsNotNone(kernel.module)

    def test_kill_switch_off_is_noop(self):
        self.arm_cuda([8.0] * 32)
        kernel = self.add_kernel(_FakeKernel("k"))
        with envs.SGLANG_ENABLE_TRITON_PREWARM_DEVICE_LOADS.override(False):
            watch.prewarm_device_loads()
        self.assertEqual(kernel.init_calls, 0)
        self.assertFalse(
            watch._prewarm_done, "a gated-off call is not a completed sweep"
        )

    def test_no_cuda_is_noop(self):
        with patch.object(watch.torch.cuda, "is_available", return_value=False):
            watch._tracking_started = True
            kernel = self.add_kernel(_FakeKernel("k"))
            watch.prewarm_device_loads()
        self.assertEqual(kernel.init_calls, 0)
        self.recipes.assert_not_called()

    def test_rocm_is_noop(self):
        self.arm_cuda([8.0] * 32)
        kernel = self.add_kernel(_FakeKernel("k"))
        with patch.object(watch.torch.version, "hip", "6.2.0", create=True):
            watch.prewarm_device_loads()
        self.assertEqual(kernel.init_calls, 0)
        self.recipes.assert_not_called()

    def test_without_install_is_noop(self):
        # _tracking_started stays False: install() never ran.
        with patch.object(watch.torch.cuda, "is_available", return_value=True):
            kernel = self.add_kernel(_FakeKernel("k"))
            watch.prewarm_device_loads()
        self.assertEqual(kernel.init_calls, 0)

    def test_idempotent(self):
        self.arm_cuda([8.0] * 32)
        kernel = self.add_kernel(_FakeKernel("k"))
        watch.prewarm_device_loads()
        self.assertEqual(kernel.init_calls, 1)

        late = self.add_kernel(_FakeKernel("late"))
        watch.prewarm_device_loads()
        self.assertEqual(late.init_calls, 0, "later calls are documented to be a no-op")

    def test_dead_refs_are_pruned_and_kernels_not_owned(self):
        self.arm_cuda([8.0] * 32)
        doomed = _FakeKernel("doomed")
        ref = weakref.ref(doomed)
        self.add_kernel(doomed)
        del doomed
        gc.collect()
        self.assertIsNone(ref(), "the recorder must not keep its kernels alive")

        survivor = self.add_kernel(_FakeKernel("survivor"))
        watch.prewarm_device_loads()
        self.assertEqual(len(watch._kernel_refs), 1, "the dead entry must be pruned")
        self.assertIsNotNone(survivor.module)


class TestCompiledKernelInstrumentation(TritonPrewarmTestBase):
    def _fake_triton_compiler(self, attrs):
        """Shadow the triton.compiler.compiler module with a module-like object."""
        return patch.dict(
            "sys.modules", {"triton.compiler.compiler": type("compiler", (), attrs)}
        )

    def test_wrap_chains_original_init_and_records(self):
        class Kernel:
            original_calls = 0

            def __init__(self, src, metadata_group, hash):
                Kernel.original_calls += 1
                self.src = src
                self.module = None
                self.name = "wrapped"

        with self._fake_triton_compiler({"CompiledKernel": Kernel}):
            self.assertTrue(watch._instrument_compiled_kernels())
            kernel = Kernel("src", {}, "hash")

        self.assertEqual(
            Kernel.original_calls, 1, "the original __init__ must still run"
        )
        self.assertTrue(watch._tracking_started)
        self.assertEqual(len(watch._kernel_refs), 1)
        self.assertIs(watch._kernel_refs[0](), kernel)

        # Idempotent: a second install() must not double-wrap.
        with self._fake_triton_compiler({"CompiledKernel": Kernel}):
            self.assertTrue(watch._instrument_compiled_kernels())
        self.assertEqual(len(watch._kernel_refs), 1)
        self.assertEqual(Kernel.original_calls, 1)

    def test_missing_class_is_a_clean_noop(self):
        with self._fake_triton_compiler({}):
            self.assertFalse(watch._instrument_compiled_kernels())
        self.assertFalse(watch._tracking_started)

    def test_unwrappable_class_is_a_clean_noop(self):
        class _FrozenMeta(type):
            def __setattr__(cls, name, value):
                raise AttributeError("read-only type")

        class Kernel(metaclass=_FrozenMeta):
            def __init__(self):
                pass

        with self._fake_triton_compiler({"CompiledKernel": Kernel}):
            self.assertFalse(watch._instrument_compiled_kernels())
        self.assertFalse(watch._tracking_started)


class TestJitRegistryScan(TritonPrewarmTestBase):
    def test_scans_device_caches_without_inserting_keys(self):
        kernel = _FakeKernel("cached")
        seen = _FakeKernel("also_cached")
        caches = defaultdict(lambda: ({}, {}, None, None, None))
        caches[0] = ({"k": kernel}, {}, None, None, None)
        caches[1] = ({"k": seen, "dup": kernel}, {}, None, None, None)
        registry = {
            "a": type("JIT", (), {"device_caches": caches})(),
            "b": type("JIT", (), {"device_caches": None})(),
        }
        fake_jit = type("jit", (), {"_triton_jit_function_registry": registry})

        with patch.dict("sys.modules", {"triton.runtime.jit": fake_jit}):
            found = list(self.real_jit_scan())

        self.assertEqual({id(k) for k in found}, {id(kernel), id(seen)})
        # Iterating a defaultdict must not have created device entries.
        self.assertEqual(sorted(caches.keys()), [0, 1])

        # The two sources deduplicate against each other.
        self.add_kernel(kernel)
        self.add_kernel(seen)
        self.assertEqual(len(watch._tracked_kernels()), 2)

    def test_missing_registry_is_empty(self):
        with patch.dict("sys.modules", {"triton.runtime.jit": None}):
            self.assertEqual(list(self.real_jit_scan()), [])


class TestInstallChaining(TritonPrewarmTestBase):
    def setUp(self):
        super().setUp()
        import triton.knobs as knobs

        self.knobs = knobs

    @staticmethod
    def _instrument():
        def instrument():
            # Stand in for the real wrap, which would touch global Triton state.
            watch._tracking_started = True
            return True

        return instrument

    def test_install_chains_pre_existing_compile_listener(self):
        seen = []

        def prev(**kwargs):
            seen.append(kwargs)

        hook = _FakeHook()
        with (
            patch.object(self.knobs.runtime, "kernel_load_start_hook", hook),
            patch.object(self.knobs.compilation, "listener", prev),
            patch.object(watch, "_instrument_compiled_kernels", new=self._instrument()),
        ):
            watch.install()
            self.assertTrue(watch._installed)
            self.assertEqual(hook.added, [watch._on_kernel_load])
            self.assertIs(self.knobs.compilation.listener, watch._on_compilation)
            self.assertIs(watch._prev_compile_listener, prev)

            # The chain must still hand every argument to the previous listener.
            times = type("T", (), {"total": 5e6})()
            src = type("S", (), {"name": "k"})()
            watch.mark_serving_started()
            with envs.SGLANG_TRITON_SLOW_COMPILE_THRESHOLD_SECS.override(0.001):
                watch._on_compilation(
                    src=src,
                    metadata=None,
                    metadata_group=None,
                    times=times,
                    cache_hit=False,
                )

        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0]["src"], src)
        self.assertIs(seen[0]["times"], times)

    def test_install_survives_missing_hooks(self):
        with (
            patch.object(
                self.knobs.runtime, "kernel_load_start_hook", None, create=True
            ),
            patch.object(watch, "_instrument_compiled_kernels", new=self._instrument()),
        ):
            watch.install()
        self.assertFalse(watch._installed)
        self.assertIsNone(watch._prev_compile_listener)
        # Tracking is independent of the knobs hooks, so prewarm stays available.
        self.assertTrue(watch._tracking_started)


class TestLateLoadWarning(TritonPrewarmTestBase):
    """The pre-existing diagnostic the prewarm exists to make rare."""

    def test_warns_below_threshold_after_serving_started(self):
        watch.mark_serving_started()
        with (
            patch.object(watch.torch.cuda, "is_available", return_value=True),
            patch.object(watch, "_free_device_gb", return_value=0.7),
        ):
            with self.assertLogs(_LOGGER, level="WARNING") as logs:
                watch._on_kernel_load("m", "f", "late_kernel", {}, "hash")
        output = "\n".join(logs.output)
        self.assertIn("late_kernel", output)
        self.assertIn("0.70 GiB", output)

    def test_silent_above_threshold(self):
        watch.mark_serving_started()
        with (
            patch.object(watch.torch.cuda, "is_available", return_value=True),
            patch.object(watch, "_free_device_gb", return_value=4.0),
        ):
            with self.assertNoLogs(_LOGGER, level="INFO"):
                watch._on_kernel_load("m", "f", "late_kernel", {}, "hash")

    def test_crash_mode_raises(self):
        watch.mark_serving_started()
        with patch.object(watch, "_free_device_gb", return_value=0.7):
            with envs.SGLANG_CRASH_ON_TRITON_LOAD_AFTER_READY.override(True):
                with self.assertRaises(RuntimeError):
                    watch._on_kernel_load("m", "f", "late_kernel", {}, "hash")


class TestEnvDefaults(CustomTestCase):
    def test_defaults(self):
        self.assertIs(envs.SGLANG_ENABLE_TRITON_PREWARM_DEVICE_LOADS.get(), True)
        self.assertEqual(envs.SGLANG_TRITON_PREWARM_MIN_FREE_GB.get(), 1.0)


class TestKnownKernelRecipes(CustomTestCase):
    def test_cpu_noop(self):
        self.assertEqual(
            triton_prewarm_ops.prewarm_known_kernels(torch.device("cpu")), []
        )

    def test_geometry_collection_without_worker(self):
        ctx = triton_prewarm_ops._collect_ctx(torch.device("cpu"), None)
        self.assertEqual(ctx.page_size, 1)
        self.assertIsNone(ctx.qsa_pool)
        self.assertIsNone(ctx.max_cache_len)
        self.assertIsNone(ctx.req_to_token)

    def test_recipe_failure_does_not_stop_the_others(self):
        def boom(ctx):
            raise RuntimeError("signature moved")

        ok = []

        def fine(ctx):
            ok.append(ctx)

        with patch.object(
            triton_prewarm_ops.torch.cuda, "is_available", return_value=True
        ):
            with patch.object(
                triton_prewarm_ops, "_RECIPES", [("bad", boom), ("fine", fine)]
            ):
                ran = triton_prewarm_ops.prewarm_known_kernels(torch.device("cpu"))

        self.assertEqual(ran, ["fine"])
        self.assertEqual(len(ok), 1)

    def test_recipes_shape(self):
        names = [name for name, _ in triton_prewarm_ops._RECIPES]
        self.assertEqual(len(names), len(set(names)), "recipe names must be unique")
        for name, fn in triton_prewarm_ops._RECIPES:
            self.assertTrue(callable(fn), name)
            self.assertTrue(
                fn.__doc__, f"{name} must document its call-site provenance"
            )

    def test_spec_worker_wrapper_is_unwrapped(self):
        runner = object()
        spec = type(
            "SpecWorker",
            (),
            {"target_worker": type("T", (), {"model_runner": runner})()},
        )()
        self.assertIs(triton_prewarm_ops._resolve_model_runner(spec), runner)
        direct = type("TpWorker", (), {"model_runner": runner})()
        self.assertIs(triton_prewarm_ops._resolve_model_runner(direct), runner)
        self.assertIsNone(triton_prewarm_ops._resolve_model_runner(None))
        self.assertIsNone(triton_prewarm_ops._resolve_model_runner(object()))


if __name__ == "__main__":
    unittest.main()
