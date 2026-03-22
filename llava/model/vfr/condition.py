from typing import List

import torch
import torch.nn as nn

from .span import SampleSpans


def extract_last_layer_hidden(outputs) -> torch.Tensor:
    if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        return outputs.hidden_states[-1]
    if isinstance(outputs, dict) and "hidden_states" in outputs and outputs["hidden_states"] is not None:
        return outputs["hidden_states"][-1]
    raise ValueError("Model outputs do not contain hidden_states; enable output_hidden_states=True")


def strip_newline_tokens(tokens: torch.Tensor, H: int, W: int, mm_newline_position: str) -> torch.Tensor:
    # tokens: [N, T, C] -> [N, H, W, C]
    if mm_newline_position == "grid":
        tokens = tokens[:, : H * (W + 1), :].reshape(tokens.shape[0], H, W + 1, tokens.shape[-1])
        return tokens[:, :, :W, :]

    tokens = tokens[:, : H * W, :]
    return tokens.reshape(tokens.shape[0], H, W, tokens.shape[-1])


class ConditionProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, H, W, C] -> [B, T, C_out, H, W]
        y = self.proj(x)
        return y.permute(0, 1, 4, 2, 3).contiguous()


def extract_llm_visual_condition(
    last_hidden_state: torch.Tensor,
    spans: List[SampleSpans],
    model_config,
    vfr_config,
) -> torch.Tensor:
    if isinstance(model_config, dict):
        mm_newline_position = str(model_config.get("mm_newline_position", "grid"))
        if str(model_config.get("mm_patch_merge_type", "flat")) == "flat":
            mm_newline_position = "no_token"
    else:
        mm_newline_position = str(getattr(model_config, "mm_newline_position", "grid"))
        if str(getattr(model_config, "mm_patch_merge_type", "flat")) == "flat":
            mm_newline_position = "no_token"

    device = last_hidden_state.device
    dtype = last_hidden_state.dtype

    per_sample = []
    max_frames = max(max(s.num_frames, 1) for s in spans) if spans else 1

    for b, sample in enumerate(spans):
        H = max(sample.frame_h, 1)
        W = max(sample.frame_w, 1)
        frames = []
        for frame_span in sample.frame_spans:
            toks = last_hidden_state[b, frame_span.start : frame_span.end, :].unsqueeze(0)
            grid = strip_newline_tokens(toks, H, W, mm_newline_position)
            frames.append(grid.squeeze(0))

        if not frames:
            frames = [torch.zeros(H, W, last_hidden_state.shape[-1], device=device, dtype=dtype)]

        stacked = torch.stack(frames, dim=0)  # [T, H, W, C]
        if stacked.shape[0] < max_frames:
            pad = torch.zeros(max_frames - stacked.shape[0], H, W, stacked.shape[-1], device=device, dtype=dtype)
            stacked = torch.cat([stacked, pad], dim=0)
        per_sample.append(stacked)

    cond = torch.stack(per_sample, dim=0) if per_sample else torch.zeros(1, 1, 1, 1, last_hidden_state.shape[-1], device=device, dtype=dtype)
    if bool(getattr(vfr_config, "detach_cond", False)):
        cond = cond.detach()
    return cond
