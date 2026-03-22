from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .condition import ConditionProjector, extract_last_layer_hidden, extract_llm_visual_condition
from .diffusion.denoiser_dit import RossDenoiser
from .gt_dinov2 import DinoV2GTEncoder
from .span import compute_visual_spans, compute_tokens_per_frame
from .target import TargetProjector, resize_patch_grid


def _to_batched_frames(batch_images, batch_size: int, device) -> torch.Tensor:
    # Convert list[tensor(3,H,W) or tensor(T,3,H,W)] to [B,T,3,H,W].
    out = []
    max_t = 1
    for b in range(batch_size):
        x = batch_images[b] if isinstance(batch_images, list) and b < len(batch_images) else None
        if x is None:
            frame = torch.zeros(1, 3, 224, 224, device=device)
        elif x.ndim == 3:
            frame = x.unsqueeze(0).to(device)
        elif x.ndim == 4:
            frame = x.to(device)
        else:
            frame = torch.zeros(1, 3, 224, 224, device=device)
        max_t = max(max_t, int(frame.shape[0]))
        out.append(frame)

    padded = []
    for x in out:
        if x.shape[0] < max_t:
            pad = torch.zeros(max_t - x.shape[0], *x.shape[1:], device=device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=0)
        padded.append(x)
    return torch.stack(padded, dim=0)


class VFRLossComputer(nn.Module):
    def __init__(self, vfr_config, model_config):
        super().__init__()
        self.vfr_config = vfr_config
        self.model_config = model_config

        self.cond_projector = None
        self.target_projector = None
        self.denoiser = None
        self.gt_encoder = None

    def _build_modules_if_needed(self, model, cond_grid: torch.Tensor, gt_grid: torch.Tensor) -> None:
        if self.cond_projector is None:
            self.cond_projector = ConditionProjector(cond_grid.shape[-1], self.vfr_config.cond_dim).to(cond_grid.device)

        if self.target_projector is None:
            self.target_projector = TargetProjector(gt_grid.shape[-1], self.vfr_config.target_dim).to(gt_grid.device)

        if self.denoiser is None:
            h, w = int(cond_grid.shape[2]), int(cond_grid.shape[3])
            self.denoiser = RossDenoiser(
                x_channel=self.vfr_config.target_dim,
                z_channel=self.vfr_config.cond_dim,
                embed_dim=self.vfr_config.cond_dim,
                depth=3,
                learn_sigma=False,
                timesteps=str(self.vfr_config.diffusion_steps),
                n_patches=h * w,
            ).to(cond_grid.device)

        if self.gt_encoder is None:
            self.gt_encoder = DinoV2GTEncoder(model_name=self.vfr_config.gt_model_name).to(gt_grid.device)

    def _zero_vfr_loss(self, device: torch.device) -> torch.Tensor:
        """Return a zero scalar that is still connected to VFR trainable params.

        This keeps gradient synchronization behavior consistent across ranks when
        some batches are skipped by VFR guards.
        """
        loss = torch.zeros((), device=device)
        for module in (self.cond_projector, self.target_projector, self.denoiser):
            if module is None:
                continue
            for p in module.parameters():
                if p.requires_grad:
                    loss = loss + p.reshape(-1)[0] * 0.0
        return loss

    def forward(self, batch: Dict, outputs, model) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = outputs.logits.device

        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        modalities = batch.get("modalities", ["image"] * int(input_ids.shape[0]))
        images = batch.get("images", None)

        if images is None:
            return self._zero_vfr_loss(device), {"vfr/skipped": 1.0}

        spans = compute_visual_spans(input_ids, attention_mask, modalities, images, self.model_config)
        last_hidden = extract_last_layer_hidden(outputs)
        cond_grid = extract_llm_visual_condition(last_hidden, spans, self.model_config, self.vfr_config)

        cond_h, cond_w = int(cond_grid.shape[2]), int(cond_grid.shape[3])

        # =======================================================================================================
        if cond_h <= 1 or cond_w <= 1:
            return self._zero_vfr_loss(device), {"vfr/skipped": 1.0, "vfr/skipped_invalid_grid": 1.0}

        if self.denoiser is not None:
            expected_hw = self.denoiser.net.x_embedder.img_size
            if isinstance(expected_hw, tuple):
                exp_h, exp_w = int(expected_hw[0]), int(expected_hw[1])
            else:
                exp_h = exp_w = int(expected_hw)
            if cond_h != exp_h or cond_w != exp_w:
                return self._zero_vfr_loss(device), {"vfr/skipped": 1.0, "vfr/skipped_mismatched_grid": 1.0}
        # =======================================================================================================

        frames = _to_batched_frames(images, int(input_ids.shape[0]), device)
        # gt_encoder is part of self, will be handled properly if we prebuilt it
        gt = self.gt_encoder(frames)  # [B,T,Hd,Wd,Cd]

        gt = resize_patch_grid(gt, out_hw=(cond_h, cond_w), mode="bilinear")

        # We don't need to rebuild it if we prebuilt it! But just in case:
        self._build_modules_if_needed(model, cond_grid, gt)

        cond = self.cond_projector(cond_grid)
        target = self.target_projector(gt)

        losses: List[torch.Tensor] = []
        for t in range(cond.shape[1]):
            losses.append(self.denoiser(cond[:, t], target[:, t]).mean())

        if not losses:
            return self._zero_vfr_loss(device), {"vfr/skipped": 1.0}

        loss_vfr = torch.stack(losses).mean()
        info = compute_tokens_per_frame(self.model_config)
        metrics = {
            "vfr/loss": float(loss_vfr.detach().item()),
            "vfr/tokens_per_frame": float(info["tokens_per_frame_with_newline"]),
            "vfr/frames": float(cond.shape[1]),
        }
        return loss_vfr, metrics
