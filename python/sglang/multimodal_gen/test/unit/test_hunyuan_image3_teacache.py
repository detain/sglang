# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the HunyuanImage-3 TeaCache skip path.

The skip hook must resolve its window on the current step-unit API
(``TeaCacheParams.is_step_boundary`` / ``get_skip_step_range``, upstream
#42422). It previously called the removed ``get_skip_boundaries``, which
raised AttributeError mid-serve on every TeaCache request. These tests drive
the model's real hook with a stub host so a future API drift fails here
instead of in a served request.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.multimodal_gen.configs.sample.teacache import TeaCacheParams
from sglang.multimodal_gen.runtime.cache.teacache import TeaCacheMixin
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.hunyuan_image3 import (
    HunyuanImage3ForCausalMM,
)
from sglang.test.test_utils import CustomTestCase

NUM_STEPS = 5


class _HunyuanTeaCacheHost(TeaCacheMixin):
    """Minimal `self` for the unbound HunyuanImage3 hook.

    Mirrors what the real model provides at forward time: mixin state (the
    real prefix does not select the CFG cache split, matching the packed
    cond/uncond forward) and ``time_embed`` producing the modulated input.
    """

    prefix = "hunyuan_image3"

    def __init__(self):
        self._init_teacache_state()
        self.time_embed = nn.Identity()


def _batch(params):
    return SimpleNamespace(
        enable_teacache=True,
        teacache_params=params,
        num_inference_steps=NUM_STEPS,
        do_classifier_free_guidance=False,  # cond/uncond run as one packed row
        is_cfg_negative=False,
    )


class TestHunyuanImage3TeaCache(CustomTestCase):
    def setUp(self):
        # start_skipping=1 / end_skipping=-1 -> window [1, 4): steps 1-3 may
        # skip, first and last steps always compute.
        self.params = TeaCacheParams(
            teacache_thresh=0.5, coefficients=[1.0, 0.0], start_skipping=1
        )
        server_args_patch = patch(
            "sglang.multimodal_gen.runtime.server_args.get_global_server_args",
            return_value=SimpleNamespace(enable_cfg_parallel=False),
        )
        server_args_patch.start()
        self.addCleanup(server_args_patch.stop)

    def _forward_step(self, host, step):
        """Run one denoising step's TeaCache gate through the model's hook."""
        timestep = torch.tensor([1.0 + step * 0.01])
        batch = _batch(self.params)
        with set_forward_context(step, None, batch):
            return HunyuanImage3ForCausalMM.should_skip_forward_for_cached_states(
                host, timestep=timestep
            )

    def test_skip_window_walk_runs_without_attribute_error(self):
        """Every step's gate must resolve on the step-unit API (#42422).

        On the stale ``get_skip_boundaries`` call site this raises
        AttributeError at step 0, where the boundary window is first read.
        """
        host = _HunyuanTeaCacheHost()
        # Near-identical modulated inputs: inside the window every step
        # accumulates well under the 0.5 threshold and skips.
        for step, expected_calc in enumerate((True, False, False, False, True)):
            should_skip = self._forward_step(host, step)
            self.assertEqual(should_skip, not expected_calc, f"step {step}")
            self.assertTrue(host.enable_teacache)
            host.cnt += 1

    def test_cached_residual_roundtrip_on_packed_batch(self):
        """forward_block's gate/cache/retrieve contract on one tensor shape."""
        host = _HunyuanTeaCacheHost()
        hidden = torch.zeros(3, 2)
        # Boundary step 0 must compute and arm the residual cache.
        self.assertFalse(self._forward_step(host, 0))
        output = hidden + torch.ones(3, 2)
        HunyuanImage3ForCausalMM.maybe_cache_states(host, output, hidden)
        self.assertTrue(host.enable_teacache)  # consumed by maybe_cache_states
        # Interior step 1 with an unchanged input must skip and reuse it.
        self.assertTrue(self._forward_step(host, 1))
        torch.testing.assert_close(
            HunyuanImage3ForCausalMM.retrieve_cached_states(host, hidden),
            output,
        )


if __name__ == "__main__":
    unittest.main()
