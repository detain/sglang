"""The `get_rope` memo key must describe the object it builds, not the call.

`_ROPE_DICT` is process-wide and the module it hands out is attached to every
model that asks for it, so a key that over-specifies duplicates a cos/sin cache
per model while a key that under-specifies shares one object whose buffers are
on the wrong device or whose class differs.  The over-specified half bit
Qwen3.8-Flash-Next serving: `ModelConfig._derive_context_length` rewrites the
*draft's* `max_position_embeddings` to the target's `--context-length` while the
target keeps the checkpoint value, and both resolve to a YaRN rope whose cache
length comes from `rope_scaling["original_max_position_embeddings"]` alone.  Two
bit-identical 256 MiB caches per rank.
"""

import math
import sys

import pytest
import torch

from sglang.srt.layers.rotary_embedding import base, factory, rope_variant
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

# The live override (--json-model-override-args) with every length divided by
# 1024, so the caches these tests build are KBs rather than hundreds of MB.
# head_dim 256 x partial_rotary_factor 0.25 = rotary_dim 64 is kept verbatim:
# it is the cache width, and it is what makes the duplicated bytes large.
YARN_MROPE_SCALING = {
    "mrope_interleaved": True,
    "mrope_section": [11, 11, 10],
    "rope_type": "yarn",
    "rope_theta": 10000000,
    "partial_rotary_factor": 0.25,
    "factor": 4.0,
    "original_max_position_embeddings": 256,
}
# Verbatim from nvidia/Qwen3.8-Flash-Next-NVFP4 + the run script's override.  Used
# only for the resolved-row arithmetic below, never built here.
LIVE_YARN_MROPE_SCALING = {
    **YARN_MROPE_SCALING,
    "original_max_position_embeddings": 262144,
}
LIVE_TARGET_MAX_POSITION = 262144
LIVE_DRAFT_MAX_POSITION = 1000000

PLAIN = dict(
    head_size=64,
    rotary_dim=64,
    base=10000,
    is_neox_style=True,
    dtype=torch.float32,
)


@pytest.fixture(autouse=True)
def isolate_rope(monkeypatch):
    # CPU-only construction: `RotaryEmbedding` picks its kernel path from these,
    # and `DeepseekScalingRotaryEmbedding` places its cache on get_device().
    monkeypatch.setattr(base, "_is_cpu", True)
    monkeypatch.setattr(base, "publish_role", lambda: None)
    monkeypatch.setattr(rope_variant, "get_device", lambda: "cpu")
    monkeypatch.setattr(factory, "_ROPE_DICT", {})


def _yarn(**overrides):
    return {**YARN_MROPE_SCALING, **overrides}


def _get(max_position, *, rope_scaling=YARN_MROPE_SCALING, head=256, rot=256, prf=0.25):
    return factory.get_rope(
        head_size=head,
        rotary_dim=rot,
        max_position=max_position,
        base=10000000,
        rope_scaling=rope_scaling,
        partial_rotary_factor=prf,
        is_neox_style=True,
        dtype=torch.float32,
    )


# ---------------------------------------------------------------------------
# The win: one entry for target + draft.
# ---------------------------------------------------------------------------


def test_target_and_draft_share_one_yarn_entry():
    target = _get(256)
    draft = _get(1024)  # what _derive_context_length writes for --context-length

    assert type(target).__name__ == "YaRNScalingMRotaryEmbedding"
    assert draft is target
    assert len(factory._ROPE_DICT) == 1
    # The cache is sized by the scaling, not by either caller's context length:
    # 256 * 4.0 rows, wide enough for the 1024-position draft.
    assert target.max_position_embeddings == 256
    assert target.cos_sin_cache.shape == (1024, 64)


@pytest.mark.parametrize(
    "scaling,expected_cls",
    [
        (_yarn(), "YaRNScalingMRotaryEmbedding"),
        (
            {k: v for k, v in _yarn().items() if k != "mrope_section"},
            "YaRNScalingRotaryEmbedding",
        ),
        (
            {
                "rope_type": "deepseek_yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 256,
                "mscale": 1.0,
                "mscale_all_dim": 0.0,
            },
            "DeepseekScalingRotaryEmbedding",
        ),
    ],
)
def test_every_max_position_free_branch_shares(scaling, expected_cls):
    target = _get(256, rope_scaling=scaling)
    draft = _get(999999, rope_scaling=scaling)

    assert type(target).__name__ == expected_cls
    assert draft is target
    assert len(factory._ROPE_DICT) == 1
    assert target.max_position_embeddings == 256
    assert target.cos_sin_cache.shape[0] == 1024


def test_live_geometry_duplicated_256_mib_per_rank():
    """The measured number the fix removes, kept as arithmetic on the real config.

    rows = 262144 * 4.0, width = rotary_dim 64 (head 256 x partial_rotary_factor
    0.25), fp32 on CUDA (base.py keeps the cache fp32 for numerical stability):
    one cache is 268435456 B, and the duplicated second copy was the waste.
    """
    cls, ctor_max_position, rows = factory._rope_geometry(
        64, LIVE_DRAFT_MAX_POSITION, LIVE_YARN_MROPE_SCALING, None
    )
    assert (cls, ctor_max_position, rows) == (
        "YaRNScalingMRotaryEmbedding",
        LIVE_TARGET_MAX_POSITION,
        1048576,
    )
    bytes_per_cache = rows * 64 * torch.finfo(torch.float32).bits // 8
    assert bytes_per_cache == 268_435_456  # 256 MiB saved per rank
    # And the target resolves to the same triple, which is why they share.
    assert factory._rope_geometry(
        64, LIVE_TARGET_MAX_POSITION, LIVE_YARN_MROPE_SCALING, None
    ) == (cls, ctor_max_position, rows)


def test_shared_entry_is_bit_identical_to_a_freshly_built_one():
    """Sharing is legal here because the two builds were never different."""
    target = _get(256)
    built_for_target = target.cos_sin_cache.clone()

    factory._ROPE_DICT.clear()  # force the draft's own construction
    draft = _get(1024)

    assert draft.cos_sin_cache.shape == built_for_target.shape
    assert torch.equal(draft.cos_sin_cache, built_for_target)


# ---------------------------------------------------------------------------
# The safety property: a longer cache is never borrowed as a shorter one.
# ---------------------------------------------------------------------------


def test_plain_rope_draft_needing_a_longer_cache_gets_its_own_entry():
    target = factory.get_rope(max_position=4096, **PLAIN)
    draft = factory.get_rope(max_position=8192, **PLAIN)

    assert draft is not target
    assert len(factory._ROPE_DICT) == 2
    # The whole point: the draft must not be handed the target's too-small cache.
    assert target.cos_sin_cache.shape == (4096, 64)
    assert draft.cos_sin_cache.shape == (8192, 64)
    assert draft.max_position_embeddings == 8192


def test_yarn_without_an_original_in_the_dict_stays_unshared():
    """`original_max_position_embeddings` defaults to the caller's max_position,
    so the YaRN cache does depend on it there and must not be shared."""
    scaling = {
        k: v
        for k, v in YARN_MROPE_SCALING.items()
        if k != "original_max_position_embeddings"
    }
    target = _get(256, rope_scaling=scaling)
    draft = _get(512, rope_scaling=scaling)

    assert target.cos_sin_cache.shape[0] == 1024
    assert draft.cos_sin_cache.shape[0] == 2048
    assert draft is not target
    assert len(factory._ROPE_DICT) == 2


def test_yarn_cache_length_is_the_arange_length_of_a_fractional_product():
    """rows = len(torch.arange(original * factor)) = ceil(product).

    Getting this wrong in _rope_geometry would not corrupt a shared entry (the
    class and the constructor positional already pin the length), but it is the
    field that proves a draft's positions fit, so it has to be the real formula.
    """
    scaling = _yarn(factor=1.5, original_max_position_embeddings=101)
    rope = _get(64, rope_scaling=scaling)

    assert factory._rope_geometry(64, 64, scaling, None) == (
        "YaRNScalingMRotaryEmbedding",
        101,
        math.ceil(101 * 1.5),
    )
    assert rope.cos_sin_cache.shape[0] == math.ceil(101 * 1.5) == 152


def test_different_scaling_factor_is_not_shared():
    short = _get(256, rope_scaling=_yarn(factor=4.0))
    long_ = _get(256, rope_scaling=_yarn(factor=8.0))

    assert long_ is not short
    assert long_.cos_sin_cache.shape[0] == 2048
    assert len(factory._ROPE_DICT) == 2


def test_equal_cache_rows_from_different_classes_are_not_shared():
    """Keying on the resolved length alone would collide here; the class and the
    scaling dict are what keep these two apart."""
    plain = factory.get_rope(max_position=1024, rope_scaling=None, **PLAIN)
    yarn = _get(256)  # 256 * 4.0 = the same 1024 rows
    assert plain.cos_sin_cache.shape[0] == yarn.cos_sin_cache.shape[0] == 1024

    assert plain is not yarn
    assert len(factory._ROPE_DICT) == 2


def test_mrope_section_difference_is_not_shared():
    interleaved = _get(256, rope_scaling=_yarn(mrope_interleaved=False))
    plain_section = _get(256, rope_scaling=_yarn(mrope_section=[10, 12, 10]))

    assert interleaved is not plain_section
    assert len(factory._ROPE_DICT) == 2


# ---------------------------------------------------------------------------
# Staleness: what else the key has to carry.
# ---------------------------------------------------------------------------


def test_rope_scaling_insertion_order_does_not_split_the_entry():
    reordered = {
        "factor": 4.0,
        "rope_type": "yarn",
        "mrope_section": [11, 11, 10],
        "rope_theta": 10000000,
        "mrope_interleaved": True,
        "partial_rotary_factor": 0.25,
        "original_max_position_embeddings": 256,
    }
    assert set(reordered) == set(YARN_MROPE_SCALING)
    assert list(reordered) != list(YARN_MROPE_SCALING)

    first = _get(256)
    second = _get(256, rope_scaling=reordered)

    assert second is first
    assert len(factory._ROPE_DICT) == 1


def test_meta_device_entry_is_not_handed_to_a_real_build():
    """The presharding signature pass builds the whole model under
    `torch.device("meta")`; a real model must not inherit that module.  The
    dead-buffer discard used to be the only thing standing between them, and it
    is a content check -- the device belongs in the key."""
    with torch.device("meta"):
        meta = _get(256)
        assert meta.cos_sin_cache.device.type == "meta"
        assert _get(256) is meta  # the meta pass keeps sharing one entry

    real = _get(256)
    assert real is not meta
    assert real.cos_sin_cache.device.type == "cpu"
    assert real.cos_sin_cache.shape == (1024, 64)


def test_get_rope_cpu_entry_is_disjoint_from_the_device_build():
    """`get_rope_cpu` reaches the same class with an explicit device argument, so
    the objects are not interchangeable even under an identical config."""
    scaling = {
        "rope_type": "deepseek_yarn",
        "factor": 2.0,
        "original_max_position_embeddings": 128,
    }
    kwargs = dict(
        head_size=64,
        rotary_dim=64,
        max_position=256,
        base=10000,
        is_neox_style=True,
        rope_scaling=scaling,
        dtype=torch.float32,
    )
    on_device = factory.get_rope(**kwargs)
    on_cpu = factory.get_rope_cpu(**kwargs, device="cpu")

    assert type(on_cpu).__name__ == "DeepseekScalingRotaryEmbedding"
    assert on_cpu is not on_device
    assert len(factory._ROPE_DICT) == 2


def test_dead_entry_is_still_discarded_under_the_new_key():
    rope = _get(256)
    assert _get(1024) is rope

    rope.cos_sin_cache.untyped_storage().resize_(0)
    rebuilt = _get(1024)

    assert rebuilt is not rope
    assert len(factory._ROPE_DICT) == 1


# ---------------------------------------------------------------------------
# Drift guard: _rope_geometry is a hand-written mirror of the dispatch.
# ---------------------------------------------------------------------------

# (name, kwargs passed to get_rope, expected class).  Every entry is also checked
# against the geometry helper: the class it names, the max_position_embeddings the
# constructor was handed, and the cache length it materialises.
BRANCH_CASES = [
    ("no-scaling", dict(rope_scaling=None), "RotaryEmbedding"),
    (
        "mrope-default",
        dict(rope_scaling={"rope_type": "default", "mrope_section": [16, 8, 8]}),
        "MRotaryEmbedding",
    ),
    (
        "llama3",
        dict(
            rope_scaling={
                "rope_type": "llama3",
                "factor": 2.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 512,
            }
        ),
        "Llama3RotaryEmbedding",
    ),
    (
        "linear",
        dict(rope_scaling={"rope_type": "linear", "factor": 2.0}),
        "LinearScalingRotaryEmbedding",
    ),
    (
        "dynamic",
        dict(rope_scaling={"rope_type": "dynamic", "factor": 2.0}),
        "DynamicNTKScalingRotaryEmbedding",
    ),
    (
        "dynamic-alpha",
        dict(rope_scaling={"rope_type": "dynamic", "alpha": 2.0}),
        "DynamicNTKAlphaRotaryEmbedding",
    ),
    (
        "yarn",
        dict(
            rope_scaling={
                "rope_type": "yarn",
                "factor": 2.0,
                "original_max_position_embeddings": 256,
            }
        ),
        "YaRNScalingRotaryEmbedding",
    ),
    (
        "yarn-mrope",
        dict(rope_scaling=_yarn()),
        "YaRNScalingMRotaryEmbedding",
    ),
    (
        "deepseek-yarn",
        dict(
            rope_scaling={
                "rope_type": "deepseek_yarn",
                "factor": 2.0,
                "original_max_position_embeddings": 256,
                "mscale": 1.0,
                "mscale_all_dim": 0.0,
            }
        ),
        "DeepseekScalingRotaryEmbedding",
    ),
    (
        "longrope",
        dict(
            rope_scaling={
                "rope_type": "longrope",
                "short_factor": [1.0] * 32,
                "long_factor": [1.0] * 32,
                "original_max_position_embeddings": 512,
            }
        ),
        "Phi3LongRoPEScaledRotaryEmbedding",
    ),
    (
        "proportional",
        dict(rope_scaling={"rope_type": "proportional"}),
        "Gemma4RotaryEmbedding",
    ),
    (
        "bailing-video",
        dict(
            rope_scaling={
                "rope_type": "default",
                "mrope_section": [16, 8, 8],
                "video_rope": True,
                "factor": 2.0,
                "original_max_position_embeddings": 256,
            }
        ),
        "BailingMRotaryEmbedding",
    ),
]


@pytest.mark.parametrize(
    "name,extra,expected_cls", BRANCH_CASES, ids=[c[0] for c in BRANCH_CASES]
)
def test_geometry_describes_the_object_that_gets_built(name, extra, expected_cls):
    max_position = 1024
    rope = factory.get_rope(
        head_size=64,
        rotary_dim=64,
        max_position=max_position,
        base=10000,
        is_neox_style=True,
        rope_scaling=extra["rope_scaling"],
        dtype=torch.float32,
    )
    cls, ctor_max_position, rows = factory._rope_geometry(
        64, max_position, extra["rope_scaling"], None
    )

    assert type(rope).__name__ == cls == expected_cls
    # Every branch that is not shareable keeps the caller's context length.
    if cls not in {
        "YaRNScalingRotaryEmbedding",
        "YaRNScalingMRotaryEmbedding",
        "DeepseekScalingRotaryEmbedding",
    }:
        assert ctor_max_position == max_position
    # rows is a redundant field (the class plus the ctor positional already pin
    # it), so a wrong formula can only cost sharing -- but it must not be wrong.
    if rows is not None:
        assert rope.cos_sin_cache.shape[0] == rows


def test_unanalyzed_scaling_type_is_conservative():
    scaling = {"mrope_section": [16, 8, 8]}  # no rope_type / type at all
    assert factory._rope_geometry(64, 4096, scaling, None) == ("", 4096, None)
    with pytest.raises(ValueError):
        factory.get_rope(max_position=4096, rope_scaling=scaling, **PLAIN)
    assert factory._ROPE_DICT == {}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
