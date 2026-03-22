from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def resize_patch_grid(tokens: torch.Tensor, out_hw: Tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
    # tokens: [B, T, H, W, C]
    b, t, h, w, c = tokens.shape
    flat = tokens.view(b * t, h, w, c).permute(0, 3, 1, 2).contiguous()
    resized = F.interpolate(flat, size=out_hw, mode=mode, align_corners=False if mode in {"bilinear", "bicubic"} else None)
    out = resized.permute(0, 2, 3, 1).contiguous()
    return out.view(b, t, out_hw[0], out_hw[1], c)


class TargetProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, H, W, C] -> [B, T, C_out, H, W]
        y = self.proj(x)
        return y.permute(0, 1, 4, 2, 3).contiguous()
