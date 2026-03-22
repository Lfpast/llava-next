from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist

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

    def _sync_skip_flag(self, local_skip: bool, device: torch.device) -> bool:
        """Synchronize skip decision across ranks.

        If any rank decides to skip VFR for this step, all ranks skip it.
        This avoids distributed hangs caused by rank-dependent branches.
        """
        if not (dist.is_available() and dist.is_initialized()):
            return local_skip

        skip_tensor = torch.tensor(1 if local_skip else 0, device=device, dtype=torch.int32)
        dist.all_reduce(skip_tensor, op=dist.ReduceOp.MAX)
        return bool(skip_tensor.item() > 0)

    def _get_denoiser_hw(self) -> Tuple[int, int]:
        if self.denoiser is None:
            return -1, -1
        expected_hw = self.denoiser.net.x_embedder.img_size
        if isinstance(expected_hw, tuple):
            return int(expected_hw[0]), int(expected_hw[1])
        return int(expected_hw), int(expected_hw)

    def forward(self, batch: Dict, outputs, model) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = outputs.logits.device

        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        batch_size = int(input_ids.shape[0])
        modalities = batch.get("modalities", ["image"] * batch_size)
        images = batch.get("images", None)

        local_skip = False
        local_reason = None
        if images is None:
            local_skip = True
            local_reason = "images"

        if local_skip and self._sync_skip_flag(local_skip, device):
            return torch.zeros((), device=device), {"vfr/skipped": 1.0, "vfr/skipped_images": 1.0}

        if not isinstance(modalities, list):
            modalities = ["image"] * batch_size

        mode = str(getattr(self.vfr_config, "tokens_mode", "both")).lower()
        if mode not in {"image", "video", "both"}:
            mode = "both"

        def _allow_modality(modality: str) -> bool:
            m = str(modality).lower()
            if mode == "both":
                return True
            return m == mode

        selected = [i for i, m in enumerate(modalities) if _allow_modality(m)]
        local_skip = len(selected) == 0
        local_reason = "tokens_mode" if local_skip else None
        if self._sync_skip_flag(local_skip, device):
            metrics = {"vfr/skipped": 1.0}
            if local_reason is None:
                metrics["vfr/skipped_other_rank"] = 1.0
            else:
                metrics["vfr/skipped_tokens_mode"] = 1.0
            return torch.zeros((), device=device), metrics

        input_ids = input_ids[selected]
        attention_mask = attention_mask[selected] if attention_mask is not None else None
        modalities = [modalities[i] for i in selected]
        if isinstance(images, list):
            images = [images[i] if i < len(images) else None for i in selected]
        last_hidden = extract_last_layer_hidden(outputs)[selected]

        spans = compute_visual_spans(input_ids, attention_mask, modalities, images, self.model_config)
        seq_len = int(last_hidden.shape[1])
        valid_span_idx = []
        for i, s in enumerate(spans):
            if s.visual_span is None or not s.frame_spans:
                continue
            if s.visual_span.end <= s.visual_span.start:
                continue
            if s.visual_span.start >= seq_len:
                continue
            if any(fs.end <= fs.start for fs in s.frame_spans):
                continue
            if any(fs.start >= seq_len for fs in s.frame_spans):
                continue
            valid_span_idx.append(i)

        local_skip = len(valid_span_idx) == 0
        local_reason = "no_valid_span" if local_skip else None
        if self._sync_skip_flag(local_skip, device):
            metrics = {"vfr/skipped": 1.0}
            if local_reason is None:
                metrics["vfr/skipped_other_rank"] = 1.0
            else:
                metrics["vfr/skipped_no_valid_span"] = 1.0
            return torch.zeros((), device=device), metrics

        input_ids = input_ids[valid_span_idx]
        if attention_mask is not None:
            attention_mask = attention_mask[valid_span_idx]
        modalities = [modalities[i] for i in valid_span_idx]
        if isinstance(images, list):
            images = [images[i] for i in valid_span_idx]
        last_hidden = last_hidden[valid_span_idx]
        spans = [spans[i] for i in valid_span_idx]

        cond_grid = extract_llm_visual_condition(last_hidden, spans, self.model_config, self.vfr_config)

        cond_h, cond_w = int(cond_grid.shape[2]), int(cond_grid.shape[3])
        desired_h, desired_w = cond_h, cond_w
        cond_resized = False

        # =======================================================================================================
        if cond_h <= 1 or cond_w <= 1:
            local_skip = True
            local_reason = "invalid_grid"

        if self.denoiser is not None:
            exp_h, exp_w = self._get_denoiser_hw()
            if cond_h != exp_h or cond_w != exp_w:
                # Keep DiT's fixed grid and align cond/target to it, instead of
                # skipping VFR for the whole step.
                desired_h, desired_w = exp_h, exp_w
                cond_resized = True

        if self._sync_skip_flag(local_skip, device):
            metrics = {"vfr/skipped": 1.0}
            if local_reason is None:
                metrics["vfr/skipped_other_rank"] = 1.0
            elif local_reason == "invalid_grid":
                metrics["vfr/skipped_invalid_grid"] = 1.0
            elif local_reason == "mismatched_grid":
                metrics["vfr/skipped_mismatched_grid"] = 1.0
            return torch.zeros((), device=device), metrics
        # =======================================================================================================

        if cond_resized:
            cond_grid = resize_patch_grid(cond_grid, out_hw=(desired_h, desired_w), mode="bilinear")
            cond_h, cond_w = desired_h, desired_w

        frames = _to_batched_frames(images, int(input_ids.shape[0]), device)
        if self.gt_encoder is None:
            self.gt_encoder = DinoV2GTEncoder(model_name=self.vfr_config.gt_model_name).to(device)
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
            return torch.zeros((), device=device), {"vfr/skipped": 1.0, "vfr/skipped_empty_losses": 1.0}

        loss_vfr = torch.stack(losses).mean()
        info = compute_tokens_per_frame(self.model_config)
        metrics = {
            "vfr/loss": float(loss_vfr.detach().item()),
            "vfr/tokens_per_frame": float(info["tokens_per_frame_with_newline"]),
            "vfr/frames": float(cond.shape[1]),
            "vfr/active_samples": float(cond.shape[0]),
            "vfr/valid_span_samples": float(len(valid_span_idx)),
            "vfr/cond_resized_to_denoiser": 1.0 if cond_resized else 0.0,
            "vfr/cond_grid_h": float(cond_h),
            "vfr/cond_grid_w": float(cond_w),
        }
        return loss_vfr, metrics
