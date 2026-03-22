from typing import Dict, Tuple

import torch
import torch.nn as nn


class VFAHook:
    """Placeholder hook holder for future layer-level VFA insertion."""

    def __init__(self, layer: int = 16):
        self.layer = layer
        self._handles = []

    def register(self, model: nn.Module) -> None:
        # Placeholder: no-op until VFA logic is implemented.
        del model

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


def compute_vfa_loss(*args, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
    # Placeholder API required by the plan.
    del args, kwargs
    return torch.tensor(0.0), {}
