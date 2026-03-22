from typing import Dict, Optional, Tuple

import torch

from llava.model.vfa.api import compute_vfa_loss
from llava.model.vfa.config import VFAConfig
from llava.model.vfr.config import VFRConfig
from llava.model.vfr.loss import VFRLossComputer
from llava.train.llava_trainer import LLaVATrainer


class LLaVATrainerVFR(LLaVATrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vfr_config = VFRConfig.from_training_args(self.args)
        self.vfa_config = VFAConfig.from_training_args(self.args)
        
        # We attached it to the model in train.py so it gets correctly partitioned by ZeRO-3
        # Need to handle case where model is a DDP or DeepSpeed wrapper
        underlying_model = self.model
        if hasattr(underlying_model, "module"):
            underlying_model = underlying_model.module
            
        if hasattr(underlying_model, "vfr_loss_computer"):
            self.vfr_loss_computer = underlying_model.vfr_loss_computer
        else:
            self.vfr_loss_computer = VFRLossComputer(self.vfr_config, underlying_model.config)

    def compute_loss(self, model, inputs, return_outputs: bool = False):
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        lm_loss = outputs.loss

        vfr_loss = torch.tensor(0.0, device=lm_loss.device)
        metrics: Dict[str, float] = {}
        if self.vfr_config.enabled:
            vfr_loss, metrics = self.vfr_loss_computer(inputs, outputs, model)

        vfa_loss = torch.tensor(0.0, device=lm_loss.device)
        if self.vfa_config.enabled:
            vfa_loss, _ = compute_vfa_loss(model=model, inputs=inputs, outputs=outputs)

        total_loss = lm_loss + self.args.vfr_weight * vfr_loss + self.args.vfa_weight * vfa_loss

        if self.state.global_step % max(1, int(self.args.vfr_log_every)) == 0:
            log_data = {
                "train/lm_loss": float(lm_loss.detach().item()),
                "train/vfr_loss": float(vfr_loss.detach().item()),
                "train/total_loss": float(total_loss.detach().item()),
            }
            log_data.update(metrics)
            self.log(log_data)

        return (total_loss, outputs) if return_outputs else total_loss
