# SPDX-License-Identifier: Apache-2.0
"""Quantization method names and their deprecated aliases.

Stack note: upstream renamed ``kitchen_int8`` to ``convrot_int8`` when the
ConvRot W8A8 backends landed (sgl-project/sglang#38040), which is after this
stack's upstream merge point.  This stack still ships ``kitchen_int8`` as the
canonical name, so the alias map is empty here; when the stack re-merges main
past the ConvRot work, restore upstream's
``{"kitchen_int8": "convrot_int8"}`` entry and drop the kitchen registry keys in
favor of ``convrot_int8``.
"""

from __future__ import annotations

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Deprecated public name -> current name.
QUANTIZATION_METHOD_ALIASES: dict[str, str] = {}


def canonical_quantization_method(name: str) -> str:
    """Map a deprecated method name to its current name, warning once per alias."""
    canonical = QUANTIZATION_METHOD_ALIASES.get(name)
    if canonical is None:
        return name
    logger.warning_once(
        f"quantization method {name!r} is a deprecated alias of {canonical!r}"
    )
    return canonical
