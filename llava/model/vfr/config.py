from dataclasses import dataclass


@dataclass
class VFRConfig:
    enabled: bool = False
    weight: float = 0.0
    gt_encoder: str = "dinov2"
    gt_model_name: str = "facebook/dinov2-base"
    use_dinov2_transformers: bool = True
    cond_dim: int = 1024
    target_dim: int = 1024
    diffusion_steps: int = 1000
    timestep_respacing: str = ""
    tokens_mode: str = "both"
    detach_cond: bool = False
    log_every: int = 50
    use_siglip_intermediate: bool = False
    siglip_select_layer: int = -2

    @classmethod
    def from_training_args(cls, args):
        return cls(
            enabled=bool(getattr(args, "vfr_enabled", False)),
            weight=float(getattr(args, "vfr_weight", 0.0)),
            gt_encoder=str(getattr(args, "vfr_gt_encoder", "dinov2")),
            gt_model_name=str(getattr(args, "vfr_gt_model_name", "facebook/dinov2-base")),
            use_dinov2_transformers=bool(getattr(args, "vfr_use_dinov2_transformers", True)),
            cond_dim=int(getattr(args, "vfr_cond_dim", 1024)),
            target_dim=int(getattr(args, "vfr_target_dim", 1024)),
            diffusion_steps=int(getattr(args, "vfr_diffusion_steps", 1000)),
            timestep_respacing=str(getattr(args, "vfr_timestep_respacing", "")),
            tokens_mode=str(getattr(args, "vfr_tokens_mode", "both")),
            detach_cond=bool(getattr(args, "vfr_detach_cond", False)),
            log_every=int(getattr(args, "vfr_log_every", 50)),
            use_siglip_intermediate=bool(getattr(args, "vfr_use_siglip_intermediate", False)),
            siglip_select_layer=int(getattr(args, "vfr_siglip_select_layer", -2)),
        )
