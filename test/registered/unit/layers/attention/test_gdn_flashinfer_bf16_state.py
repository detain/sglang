from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=6, stage="base-b", runner_config="1-gpu-small")

import unittest

import torch

from sglang.srt.layers.attention.linear.utils import (
    flashinfer_gdn_uses_state_pool,
    has_flashinfer_gdn_bf16_state_kernel,
)

_KV = 128


def _pooled_bf16_state_supported():
    return (
        torch.cuda.is_available()
        and flashinfer_gdn_uses_state_pool(torch.cuda.get_device_capability())
        and has_flashinfer_gdn_bf16_state_kernel()
    )


@unittest.skipUnless(
    _pooled_bf16_state_supported(),
    "requires a capability on the pooled-bf16 policy with flashinfer's "
    "bf16-state kernel installed",
)
class TestGdnFlashInferBf16State(unittest.TestCase):
    """SM120 pooled-bf16 contract: the relaxed FlashInferGDNKernel must agree
    with the dtype-generic Triton reference on the CuTe bf16-state decode
    kernels (the capability probe that lifts the SM120 unpooled-fp32 gate),
    including across a sequential decode loop (state drift) and the bf16 MTP
    kernel vs. the same number of sequential Triton steps."""

    B, H, HV, K, V, P = 2, 2, 4, _KV, _KV, 4

    def setUp(self):
        free = torch.cuda.mem_get_info()[0]
        if free < 1 << 30:
            self.skipTest(f"needs ~1GB free GPU memory, have {free >> 20}MB")
        from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
            FlashInferGDNKernel,
        )
        from sglang.srt.layers.attention.linear.kernels.gdn_triton import (
            TritonGDNKernel,
        )

        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.fi = FlashInferGDNKernel()
        self.tr = TritonGDNKernel()
        self.idx = torch.tensor([1, 3], dtype=torch.int32, device=self.device)
        self.qsl = torch.arange(self.B + 1, dtype=torch.int32, device=self.device)
        # The relaxation under test: SM120 must now be on the pooled policy.
        self.assertTrue(self.fi.use_state_pool)

    def _inputs(self, seed):
        g = torch.Generator(device=self.device).manual_seed(seed)

        def r(*shape, dt=torch.float32, sc=1.0):
            return (
                torch.randn(*shape, generator=g, device=self.device, dtype=dt) * sc
            ).to(dt)

        B, H, HV, K, V = self.B, self.H, self.HV, self.K, self.V
        return (
            r(1, B, H, K, sc=0.5).to(self.dtype),
            r(1, B, H, K, sc=0.5).to(self.dtype),
            r(1, B, HV, V, sc=0.5).to(self.dtype),
            r(1, B, HV, sc=1.0).to(self.dtype),
            r(1, B, HV, sc=1.0).to(self.dtype),
            r(HV, sc=0.5),  # A_log fp32 (asserted by the bf16 kernel)
            r(HV, sc=0.1).to(self.dtype),  # dt_bias bf16 (bf16-or-fp32 allowed)
        )

    def _pool(self, seed=1000):
        g = torch.Generator(device=self.device).manual_seed(seed)
        shape = (self.P, self.HV, self.V, self.K)
        return (torch.randn(*shape, generator=g, device=self.device) * 0.05).to(
            self.dtype
        )

    def _decode(self, kernel, pool, inputs):
        q, k, v, a, b, A_log, dt_bias = inputs
        return kernel.decode(
            q,
            k,
            v,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_states=pool,
            cache_indices=self.idx,
            query_start_loc=self.qsl,
        )

    @staticmethod
    def _rel_l2(a, b):
        a = a.float().reshape(-1)
        b = b.float().reshape(-1)
        return ((a - b).norm() / (b.norm() + 1e-12)).item()

    def test_pooled_bf16_decode_matches_triton(self):
        inputs = self._inputs(0)
        out_fi = self._decode(self.fi, self._pool(), inputs)
        out_tr = self._decode(self.tr, self._pool(), inputs)
        pool_fi, pool_tr = self._pool(), self._pool()
        self._decode(self.fi, pool_fi, inputs)
        self._decode(self.tr, pool_tr, inputs)
        self.assertLess(self._rel_l2(out_fi, out_tr), 5e-2)
        self.assertLess(self._rel_l2(pool_fi, pool_tr), 5e-2)

    def test_sequential_decode_no_state_drift(self):
        pool_fi, pool_tr = self._pool(99), self._pool(99)
        worst_out = worst_state = 0.0
        for step in range(8):
            inputs = self._inputs(10 + step)
            o_fi = self._decode(self.fi, pool_fi, inputs)
            o_tr = self._decode(self.tr, pool_tr, inputs)
            worst_out = max(worst_out, self._rel_l2(o_fi, o_tr))
            worst_state = max(worst_state, self._rel_l2(pool_fi, pool_tr))
        self.assertLess(worst_out, 1e-1)
        self.assertLess(worst_state, 1e-1)
        self.assertFalse(pool_fi.isnan().any().item())
        self.assertFalse(pool_tr.isnan().any().item())

    def test_bf16_mtp_matches_sequential_triton(self):
        from flashinfer.gdn_kernels.gdn_decode_bf16_state import (
            gated_delta_rule_mtp as bf16_mtp,
        )

        T = 4
        g = torch.Generator(device=self.device).manual_seed(77)

        def r(*shape, dt=torch.float32, sc=1.0):
            return (
                torch.randn(*shape, generator=g, device=self.device, dtype=dt) * sc
            ).to(dt)

        B, H, HV, K, V = self.B, self.H, self.HV, self.K, self.V
        q4 = r(B, T, H, K, sc=0.5).to(self.dtype)
        k4 = r(B, T, H, K, sc=0.5).to(self.dtype)
        v4 = r(B, T, HV, V, sc=0.5).to(self.dtype)
        a4 = r(B, T, HV, sc=1.0).to(self.dtype)
        b4 = r(B, T, HV, sc=1.0).to(self.dtype)
        A_log = r(HV, sc=0.5)
        dt_bias = r(HV, sc=0.1).to(self.dtype)

        pool_mtp, pool_ref = self._pool(55), self._pool(55)
        out_mtp = bf16_mtp(
            A_log=A_log,
            a=a4,
            dt_bias=dt_bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=q4,
            k=k4,
            v=v4,
            b=b4,
            initial_state_source=pool_mtp,
            initial_state_indices=self.idx,
            use_qk_l2norm_in_kernel=True,
        )
        steps = []
        for t in range(T):
            steps.append(
                self._decode(
                    self.tr,
                    pool_ref,
                    (
                        q4[:, t].unsqueeze(0).contiguous(),
                        k4[:, t].unsqueeze(0).contiguous(),
                        v4[:, t].unsqueeze(0).contiguous(),
                        a4[:, t].unsqueeze(0).contiguous(),
                        b4[:, t].unsqueeze(0).contiguous(),
                        A_log,
                        dt_bias,
                    ),
                )[0]
            )
        out_seq = torch.stack(steps, dim=1)
        # MTP keeps h in fp32 registers across steps while sequential decode
        # re-reads the bf16 pool every step, so allow a rounding-wider bound.
        self.assertLess(self._rel_l2(out_mtp, out_seq), 5e-2)
        self.assertLess(self._rel_l2(pool_mtp, pool_ref), 5e-2)


if __name__ == "__main__":
    unittest.main()
