from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model


def remove_cls_token(tokens: torch.Tensor) -> torch.Tensor:
    return tokens[:, 1:, :] if tokens.shape[1] > 1 else tokens


class DinoV2GTEncoder(nn.Module):
    def __init__(self, model_name: str = "facebook/dinov2-base"):
        super().__init__()
        self.model = Dinov2Model.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.image_size = int(getattr(self.model.config, "image_size", 224))
        self.hidden_size = int(self.model.config.hidden_size)
        self.patch = int(getattr(self.model.config, "patch_size", 14))
        self.grid = self.image_size // self.patch

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # Align device and dtype with inputs dynamically to circumvent DeepSpeed ZeRO-3 offloading/CPU placement quirks
        if self.model.device != frames.device or self.model.dtype != frames.dtype:
            self.model.to(device=frames.device, dtype=frames.dtype)

        # frames: [B,T,3,H,W] or [N,3,H,W]
        restore_bt = False
        if frames.ndim == 5:
            B, T = int(frames.shape[0]), int(frames.shape[1])
            flat = frames.reshape(B * T, *frames.shape[2:])
            restore_bt = True
        else:
            B, T = int(frames.shape[0]), 1
            flat = frames

        flat = flat.float()
        if flat.max() > 1.0:
            flat = flat / 255.0

        if flat.shape[-1] != self.image_size or flat.shape[-2] != self.image_size:
            flat = F.interpolate(flat, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

        mean = torch.tensor([0.485, 0.456, 0.406], device=flat.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=flat.device).view(1, 3, 1, 1)
        pixel_values = (flat - mean) / std
        
        # Cast back to the model's expected dtype before forwarding
        pixel_values = pixel_values.to(dtype=self.model.dtype)

        out = self.model(pixel_values=pixel_values, return_dict=True)
        patch_tokens = remove_cls_token(out.last_hidden_state)
        grid = patch_tokens.view(patch_tokens.shape[0], self.grid, self.grid, self.hidden_size)

        if restore_bt:
            return grid.view(B, T, self.grid, self.grid, self.hidden_size)
        return grid.view(B, T, self.grid, self.grid, self.hidden_size)
