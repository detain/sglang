"""Fork-carried fused-MoE tuning-config coverage.

Upstream's own #45672 CPU prune deleted the original file; the two cases
here are FORK artifacts that prune did not (and could not) cover: the
entry-3-era int4 tuner filename contract and the entry-235-wave RTX PRO
6000 (Blackwell Max-Q) tuned-config fleet. Re-homed by MERGE-5 B2 so a
future triton-version or tuner-path drift on THIS stack's kernels fails
loudly on the CPU suite.
"""

import json
import sys
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe_triton_config

BENCHMARK_DIR = Path(__file__).parents[5] / "benchmark" / "kernels" / "fused_moe_triton"
sys.path.insert(0, str(BENCHMARK_DIR))

import common_utils  # noqa: E402


def test_int4_tuner_filename_uses_runtime_down_projection_dimension(monkeypatch):
    monkeypatch.setattr(
        common_utils,
        "get_config_file_name",
        lambda E, N, *_args: f"E={E},N={N}.json",
    )

    filename = common_utils.get_config_filename(
        num_experts=256,
        shard_intermediate_size=512,
        hidden_size=1024,
        topk=8,
        dtype=None,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=True,
        per_channel_quant=False,
        block_shape=[128, 128],
    )

    assert filename == "E=256,N=256.json"


def test_rtx_pro_6000_server_edition_configs_resolve_for_current_triton_runtime():
    config_root = (
        Path(fused_moe_triton_config.__file__).parent / "configs" / "triton_3_7_1"
    )
    for experts in (64, 128, 256):
        config_path = config_root / (
            f"E={experts},N=640,"
            "device_name=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition,"
            "dtype=fp8_w8a8,block_shape=[128, 128].json"
        )
        assert config_path.is_file(), config_path.name
        config = json.loads(config_path.read_text())
        assert config, config_path.name
        for tile in config.values():
            assert tile["BLOCK_SIZE_M"] > 0
            assert tile["BLOCK_SIZE_N"] > 0


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
