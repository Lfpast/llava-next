import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from llava.constants import IMAGE_TOKEN_INDEX


@dataclass
class Span:
    start: int
    end: int


@dataclass
class SampleSpans:
    modality: str
    visual_span: Optional[Span] = None
    frame_spans: List[Span] = field(default_factory=list)
    num_frames: int = 0
    frame_h: int = 0
    frame_w: int = 0


def compute_tokens_per_frame(model_config) -> Dict[str, int]:
    if isinstance(model_config, dict):
        patch = int(model_config.get("mm_spatial_pool_stride", 1) or 1)
        merge_type = str(model_config.get("mm_patch_merge_type", "flat"))
        side = 27
        if "vision_config" in model_config and hasattr(model_config["vision_config"], "image_size") and hasattr(model_config["vision_config"], "patch_size"):
            side = int(model_config["vision_config"].image_size) // int(model_config["vision_config"].patch_size)
    else:
        patch = int(getattr(model_config, "mm_spatial_pool_stride", 1) or 1)
        merge_type = str(getattr(model_config, "mm_patch_merge_type", "flat"))
        side = 27
        if hasattr(model_config, "vision_config") and hasattr(model_config.vision_config, "image_size") and hasattr(model_config.vision_config, "patch_size"):
            side = int(model_config.vision_config.image_size) // int(model_config.vision_config.patch_size)

    # Align with llava_arch behavior:
    # - video features are spatially pooled by mm_spatial_pool_stride
    # - image features keep raw patch grid for common single-image path
    image_h = int(side)
    image_w = int(side)
    pooled_h = int(math.ceil(side / patch))
    pooled_w = int(math.ceil(side / patch))

    if isinstance(model_config, dict):
        newline_mode = str(model_config.get("mm_newline_position", "grid"))
    else:
        newline_mode = str(getattr(model_config, "mm_newline_position", "grid"))
    video_no_newline = pooled_h * pooled_w
    if newline_mode == "grid":
        video_with_newline = pooled_h * (pooled_w + 1)
    elif newline_mode in {"frame", "one_token"}:
        video_with_newline = video_no_newline + 1
    else:
        video_with_newline = video_no_newline

    image_no_newline = image_h * image_w
    image_with_newline = image_no_newline + (1 if "unpad" in merge_type else 0)

    info = {
        "H": pooled_h,
        "W": pooled_w,
        "image_H": image_h,
        "image_W": image_w,
        "tokens_per_frame_with_newline": video_with_newline,
        "tokens_per_frame_no_newline": video_no_newline,
        "image_tokens_no_newline": image_no_newline,
        "image_tokens_with_newline": image_with_newline,
    }
    return info


def split_video_span_into_frames(span: Span, tokens_per_frame_with_newline: int, num_frames: int) -> List[Span]:
    out = []
    cursor = span.start
    for _ in range(num_frames):
        out.append(Span(start=cursor, end=cursor + tokens_per_frame_with_newline))
        cursor += tokens_per_frame_with_newline
    return out


def _visual_tokens_for_sample(modality: str, image_tensor, model_config, info: Dict[str, int]) -> int:
    if isinstance(model_config, dict):
        merge_type = str(model_config.get("mm_patch_merge_type", "flat"))
    else:
        merge_type = str(getattr(model_config, "mm_patch_merge_type", "flat"))
        
    if modality != "video":
        if "unpad" in merge_type:
            return int(info["image_tokens_with_newline"])
        return int(info["image_tokens_no_newline"])

    if image_tensor is None:
        frames = 1
    elif image_tensor.ndim == 4:
        frames = int(image_tensor.shape[0])
    else:
        frames = 1

    per_frame = int(info["tokens_per_frame_no_newline"])
    if merge_type == "flat":
        return frames * per_frame

    if isinstance(model_config, dict):
        newline_mode = str(model_config.get("mm_newline_position", "grid"))
    else:
        newline_mode = str(getattr(model_config, "mm_newline_position", "grid"))
        
    if newline_mode == "grid":
        per_frame = int(info["tokens_per_frame_with_newline"])
    elif newline_mode in {"frame", "one_token"}:
        per_frame = int(info["tokens_per_frame_no_newline"] + 1)

    return frames * per_frame


def compute_visual_spans(
    batch_input_ids: torch.Tensor,
    batch_attention_mask: Optional[torch.Tensor],
    batch_modalities,
    batch_images,
    model_config,
) -> List[SampleSpans]:
    info = compute_tokens_per_frame(model_config)
    batch_size = int(batch_input_ids.shape[0])
    out: List[SampleSpans] = []

    for b in range(batch_size):
        modality = "image"
        if isinstance(batch_modalities, list) and b < len(batch_modalities):
            modality = str(batch_modalities[b])

        image_tensor = None
        if isinstance(batch_images, list) and b < len(batch_images):
            image_tensor = batch_images[b]

        attn = batch_attention_mask[b].bool() if batch_attention_mask is not None else torch.ones_like(batch_input_ids[b], dtype=torch.bool)
        trimmed = batch_input_ids[b][attn]
        image_pos = (trimmed == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False).flatten().tolist()

        if len(image_pos) == 0:
            sample = SampleSpans(modality=modality)
            if modality == "video":
                sample.frame_h = int(info["H"])
                sample.frame_w = int(info["W"])
            else:
                sample.frame_h = int(info["image_H"])
                sample.frame_w = int(info["image_W"])
            out.append(sample)
            continue

        vis_tokens = _visual_tokens_for_sample(modality, image_tensor, model_config, info)

        boundaries = [-1] + image_pos + [trimmed.shape[0]]
        expanded_cursor = 0
        vis_start = 0
        vis_end = 0
        for i in range(len(boundaries) - 1):
            seg_len = boundaries[i + 1] - boundaries[i] - 1
            expanded_cursor += seg_len
            if i < len(image_pos):
                vis_start = expanded_cursor
                expanded_cursor += vis_tokens
                vis_end = expanded_cursor

        sample = SampleSpans(modality=modality, visual_span=Span(vis_start, vis_end))
        if modality == "video":
            num_frames = int(image_tensor.shape[0]) if image_tensor is not None and image_tensor.ndim == 4 else 1
            pf = info["tokens_per_frame_with_newline"]
            
            if isinstance(model_config, dict):
                merge_type = str(model_config.get("mm_patch_merge_type", "flat"))
            else:
                merge_type = str(getattr(model_config, "mm_patch_merge_type", "flat"))
                
            if merge_type == "flat":
                pf = info["tokens_per_frame_no_newline"]
            sample.frame_spans = split_video_span_into_frames(sample.visual_span, pf, num_frames)
            sample.num_frames = num_frames
        else:
            sample.frame_spans = [sample.visual_span]
            sample.num_frames = 1

        if modality == "video":
            sample.frame_h = int(info["H"])
            sample.frame_w = int(info["W"])
        else:
            sample.frame_h = int(info["image_H"])
            sample.frame_w = int(info["image_W"])
        out.append(sample)

    return out
