import pytorch_lightning as pl
import sys, gc
import random
import torch
import torchaudio
import typing as tp
import wandb

from ema_pytorch import EMA
from einops import rearrange
from safetensors.torch import save_file
from torch import optim
from torch.nn import functional as F
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from ..inference.sampling import get_alphas_sigmas, sample, sample_discrete_euler
from ..models.diffusion import ConditionedDiffusionModelWrapper
from .losses import MSELoss, MultiLoss
from .utils import create_optimizer_from_config, create_scheduler_from_config

from time import time



class DiffusionCondTrainingWrapper(pl.LightningModule):
    '''
    Wrapper for training a conditional audio diffusion model.
    '''
    def __init__(
            self,
            model: ConditionedDiffusionModelWrapper,
            lr: float = None,
            mask_padding: bool = False,
            mask_padding_dropout: float = 0.0,
            use_ema: bool = False,
            log_loss_info: bool = False,
            optimizer_configs: dict = None,
            pre_encoded: bool = False,
            cfg_dropout_prob = 0.1,
            timestep_sampler: tp.Literal["uniform", "logit_normal"] = "uniform",
    ):
        super().__init__()

        self.diffusion = model

        if use_ema:
            self.diffusion_ema = EMA(
                self.diffusion.model,
                beta=0.9999,
                power=3/4,
                update_every=1,
                update_after_step=1,
                include_online_model=False
            )
        else:
            self.diffusion_ema = None

        self.mask_padding = mask_padding
        self.mask_padding_dropout = mask_padding_dropout

        self.cfg_dropout_prob = cfg_dropout_prob

        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

        self.timestep_sampler = timestep_sampler

        self.diffusion_objective = model.diffusion_objective

        self.loss_modules = [
            MSELoss("output", 
                   "targets", 
                   weight=1.0, 
                   mask_key="padding_mask" if self.mask_padding else None, 
                   name="mse_loss"
            )
        ]

        self.losses = MultiLoss(self.loss_modules)

        self.train_epoch_losses_info = []

        self.log_loss_info = log_loss_info

        assert lr is not None or optimizer_configs is not None, "Must specify either lr or optimizer_configs in training config"

        if optimizer_configs is None:
            optimizer_configs = {
                "diffusion": {
                    "optimizer": {
                        "type": "Adam",
                        "config": {
                            "lr": lr
                        }
                    }
                }
            }
        else:
            if lr is not None:
                print(f"WARNING: learning_rate and optimizer_configs both specified in config. Ignoring learning_rate and using optimizer_configs.")

        self.optimizer_configs = optimizer_configs
        self.pre_encoded = pre_encoded


    def configure_optimizers(self):
        diffusion_opt_config = self.optimizer_configs['diffusion']
        opt_diff = create_optimizer_from_config(diffusion_opt_config['optimizer'], self.diffusion.parameters())

        if "scheduler" in diffusion_opt_config:
            sched_diff = create_scheduler_from_config(diffusion_opt_config['scheduler'], opt_diff)
            sched_diff_config = {
                "scheduler": sched_diff,
                "interval": "step"
            }
            return [opt_diff], [sched_diff_config]

        return [opt_diff]


    def training_step(self, batch, batch_idx):
        reals, metadata = batch
        loss_info = {}

        diffusion_input = reals  # [batchsize, hidden_dim(VAE), seq]

        with torch.cuda.amp.autocast():
            conditioning = self.diffusion.conditioner(metadata, self.device) # dict{"condition_name":(condition,  mask)}   ([batchsize, seq, output_dim], [batchsize, seq]) 


        if self.diffusion.pretransform is not None:
            self.diffusion.pretransform.to(self.device)

            if not self.pre_encoded:
                with torch.cuda.amp.autocast() and torch.set_grad_enabled(self.diffusion.pretransform.enable_grad):
                    diffusion_input = self.diffusion.pretransform.encode(diffusion_input)
            else:            
                # Apply scale to pre-encoded latents if needed, as the pretransform encode function will not be run
                if hasattr(self.diffusion.pretransform, "scale") and self.diffusion.pretransform.scale != 1.0:
                    diffusion_input = diffusion_input / self.diffusion.pretransform.scale

        if self.timestep_sampler == "uniform":   
            # Draw uniformly distributed continuous timesteps
            t = self.rng.draw(reals.shape[0])[:, 0].to(self.device) # [batchsize]
        elif self.timestep_sampler == "logit_normal":
            t = torch.sigmoid(torch.randn(reals.shape[0], device=self.device))
            

        # Calculate the noise schedule parameters for those timesteps
        if self.diffusion_objective == "v":
            alphas, sigmas = get_alphas_sigmas(t)
        elif self.diffusion_objective == "rectified_flow":
            alphas, sigmas = 1-t, t


        # Combine the ground truth data and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(diffusion_input)
        noised_inputs = diffusion_input * alphas + noise * sigmas


        if self.diffusion_objective == "v":
            targets = noise * alphas - diffusion_input * sigmas
        elif self.diffusion_objective == "rectified_flow":
            targets = noise - diffusion_input


        extra_args = {}
        with torch.cuda.amp.autocast():
            output = self.diffusion(noised_inputs, t, cond=conditioning, cfg_dropout_prob = self.cfg_dropout_prob, **extra_args)
            loss_info.update({
                "output": output,
                "targets": targets,
                "padding_mask":  None,
            })
            loss, losses = self.losses(loss_info)


        log_dict = {
            'step': batch_idx,
            'train_step/loss': loss.detach(),
            'train_step/lr': self.trainer.optimizers[0].param_groups[0]['lr']
            # 'train/std_data': diffusion_input.std(),
        }
        for loss_name, loss_value in losses.items():
            log_dict[f"train/{loss_name}"] = loss_value.detach()
        self.train_epoch_losses_info.append(log_dict)
        self.log_dict(log_dict)
        
        return loss

    def on_train_epoch_end(self):
        train_epoch_losses_info = self.all_gather(self.train_epoch_losses_info)
        train_epoch_loss = torch.stack([loss['train_step/loss'] for loss in train_epoch_losses_info])
        train_epoch_lr = torch.stack([loss['train_step/lr'] for loss in train_epoch_losses_info])

        log_dict = {
            'epoch': self.current_epoch,
            'train_epoch/loss':train_epoch_loss.mean(),
            'train_epoch/lr':train_epoch_lr.mean()
            }
        self.log_dict(log_dict)
        self.epoch_losses_info = []


    def test_step(self, batch, batch_idx):
        pass
    def on_test_epoch_end(self):
        pass

    def export_model(self, path, use_safetensors=True):
        if self.diffusion_ema is not None:
            self.diffusion.model = self.diffusion_ema.ema_model
        
        if use_safetensors:
            save_file(self.diffusion.state_dict(), path)
        else:
            torch.save({"state_dict": self.diffusion.state_dict()}, path)
