"""Low-level utilities shared by model layers and CUDA graph runners.

Keep this package initializer free of imports: model layers import lightweight
execution state from forward_batch and prefill_graph while their own modules
are still initializing. Loading runner dependencies here (especially DeepEP)
can cycle back through attention, quantization, and disaggregation modules.

Import utilities directly from their defining submodules. The legacy runner
exports remain in model_executor.runner for speculative draft runners.
"""

# TODO(merge5): re-plant entry 176 semantics (batch B4); fork hunks saved /tmp/opencode/merge5-dossier/replant/python__sglang__srt__model_executor__runner_utils____init__.py.fork-hunks.patch
