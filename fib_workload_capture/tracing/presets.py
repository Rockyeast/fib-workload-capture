# Vendored from FlashInfer-Bench flashinfer_bench/tracing/presets.py
# (https://github.com/flashinfer-ai/flashinfer-bench), as added by PR #428.
# Copyright (c) 2025 by FlashInfer-Bench Contributors, Apache-2.0. See NOTICE.
"""Input dump policy used by SGLang module capture, registered with flashinfer-bench."""

from typing import Any, Dict, List

import torch

from flashinfer_bench.tracing.policy import PolicyRegistry
from flashinfer_bench.utils import is_dtype_integer


@PolicyRegistry.register_input_dump_policy("dump_structural")
class DumpStructuralPolicy:
    """Dump integer tensors and numeric scalar arguments, but randomize float activations."""

    def dump(self, inputs: Dict[str, Any]) -> List[str]:
        """Return names whose exact values affect workload structure."""
        return [
            name
            for name, value in inputs.items()
            if isinstance(value, (int, float, bool))
            or (isinstance(value, torch.Tensor) and is_dtype_integer(value.dtype))
        ]
