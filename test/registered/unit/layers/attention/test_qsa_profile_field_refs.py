"""Static guard: QSA code may only read fields QSAProfile actually defines.

``QSAProfile`` is a frozen msgspec struct, so a stale attribute (e.g. the
``variant`` field dropped with the tokenwise implementation) raises
AttributeError at the first forward pass instead of at import. The QSA
prefill paths that read the profile need a GPU, so nothing in the CPU suite
exercises them; this walks the AST instead.
"""

import ast
import inspect
from pathlib import Path

from sglang.srt.layers.attention import qwen_sparse_attn_backend
from sglang.srt.layers.attention.qsa import config as qsa_config
from sglang.srt.layers.attention.qsa.config import QSAProfile
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

# Modules that hold a parsed profile and read fields off it.
_MODULES = (qwen_sparse_attn_backend, qsa_config)
# Locals/attributes that hold a QSAProfile in those modules.
_PROFILE_HOLDERS = ("qsa_profile", "profile")


def _profile_attribute_names(module) -> set:
    """Names read off any QSAProfile-holding expression in module."""

    tree = ast.parse(Path(inspect.getfile(module)).read_text())
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        holder = node.value
        if isinstance(holder, ast.Attribute):
            holder_name = holder.attr
        elif isinstance(holder, ast.Name):
            holder_name = holder.id
        else:
            continue
        if holder_name in _PROFILE_HOLDERS:
            names.add(node.attr)
    return names


def _profile_members() -> set:
    members = set(QSAProfile.__struct_fields__)
    members.update(
        name
        for name, value in vars(QSAProfile).items()
        if isinstance(value, property) or not name.startswith("_")
    )
    return members


def test_qsa_profile_reads_resolve_to_real_members():
    members = _profile_members()
    stale = {}
    for module in _MODULES:
        offenders = sorted(_profile_attribute_names(module) - members)
        if offenders:
            stale[module.__name__] = offenders

    assert not stale, (
        "QSA code reads attributes that QSAProfile does not define: "
        f"{stale}; fields are {sorted(QSAProfile.__struct_fields__)}"
    )


def test_guard_detects_a_removed_field():
    # The pre-fix backend line was
    #   self.qsa_profile.variant == QSA_VARIANT_COMPRESSED
    # which this guard must flag.
    tree = ast.parse("x = self.qsa_profile.variant\n")
    read = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr in _PROFILE_HOLDERS
    }

    assert read == {"variant"}
    assert read - _profile_members() == {"variant"}
