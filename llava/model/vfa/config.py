from dataclasses import dataclass


@dataclass
class VFAConfig:
    enabled: bool = False
    weight: float = 0.0
    layer: int = 16

    @classmethod
    def from_training_args(cls, args):
        return cls(
            enabled=bool(getattr(args, "vfa_enabled", False)),
            weight=float(getattr(args, "vfa_weight", 0.0)),
            layer=int(getattr(args, "vfa_layer", 16)),
        )
