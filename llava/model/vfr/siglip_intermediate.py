import math

import torch


def extract_siglip_intermediate_features(vision_tower, frames: torch.Tensor, select_layer: int = -2) -> torch.Tensor:
    # Diagnostic-only side path: call vision encoder with output_hidden_states=True.
    core = getattr(vision_tower, "vision_tower", vision_tower)

    if frames.ndim == 5:
        B, T = int(frames.shape[0]), int(frames.shape[1])
        flat = frames.reshape(B * T, *frames.shape[2:])
    else:
        B, T = int(frames.shape[0]), 1
        flat = frames

    p = next(core.parameters())
    out = core(flat.to(device=p.device, dtype=p.dtype), output_hidden_states=True, return_dict=True)
    hs = out.hidden_states[select_layer]

    side = int(math.sqrt(hs.shape[1]))
    if side * side + 1 == hs.shape[1]:
        hs = hs[:, 1:, :]
        side = int(math.sqrt(hs.shape[1]))

    grid = hs.view(hs.shape[0], side, side, hs.shape[-1])
    return grid.view(B, T, side, side, hs.shape[-1])
