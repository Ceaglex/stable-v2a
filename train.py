import json
import os
import pytorch_lightning as pl
from einops import rearrange
from collections import OrderedDict
from torch.utils.data import DataLoader
from safetensors.torch import load_file
from datetime import datetime

from stable_audio_tools import create_model_from_config
from stable_audio_tools.data.dataset import VideoFeatDataset, collation_fn
from stable_audio_tools.training.training_wrapper import DiffusionCondTrainingWrapper
from stable_audio_tools.training.diffusion import PrintCallback



def main():

    model_config_file = './stable_audio_tools/configs/model_config.json'
    with open(model_config_file) as f:
        model_config = json.load(f)
        sample_rate = model_config["sample_rate"]
    # state_dict = torch.load('./lightning_logs/version_2/checkpoints/epoch=9-step=12800.ckpt')['state_dict']
    # state_dict = OrderedDict([(".".join(key.split('.')[1:]), value)  for key, value in state_dict.items()])
    model = create_model_from_config(model_config)
    model.load_state_dict(load_file('./weight/StableAudio/lightning_logs/version_3/checkpoints/epoch=9-step=14620.safetensors'), strict=True)


    info_dirs = [
        './dataset/feature/train/AudioSet/10', 
        './dataset/feature/train/VGGSound/10'
        ]
    audio_dirs = [
        '/home/chengxin/chengxin/AudioSet/generated_audios/train/10', 
        '/home/chengxin/chengxin/VGGSound/generated_audios/train/10'
        ]
    ds_config = {
        'info_dirs' : info_dirs,
        'audio_dirs' : audio_dirs,
        'sample_rate':sample_rate, 
        'exts':'wav',
        'force_channels':"stereo"
    }
    dl_config = {
        'batch_size':25, 
        'shuffle':True,
        # 'num_workers':1, 
        # 'persistent_workers':True, 
        'pin_memory':True, 
        'drop_last':False, 
    }
    dataset = VideoFeatDataset(**ds_config)
    dataloader = DataLoader(dataset=dataset,  collate_fn=collation_fn, **dl_config)


    training_config = model_config.get('training', None)
    training_wrapper = DiffusionCondTrainingWrapper(
                model=model, 
                lr=training_config.get("learning_rate", 1e-6),
                mask_padding=training_config.get("mask_padding", False),
                mask_padding_dropout=training_config.get("mask_padding_dropout", 0.0),
                use_ema = training_config.get("use_ema", True),
                log_loss_info=training_config.get("log_loss_info", False),
                optimizer_configs=training_config.get("optimizer_configs", None),
                pre_encoded=training_config.get("pre_encoded", False),
                cfg_dropout_prob = training_config.get("cfg_dropout_prob", 0.1),
                timestep_sampler = training_config.get("timestep_sampler", "uniform"),
            )


    devices = [0,1,2,3,4,5,6,7] 
    run_name = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    wandb_logger = pl.loggers.WandbLogger(project="stable-v2a", name = run_name, save_dir="./weight/StableAudio")
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=50, dirpath=f'./weight/StableAudio/{run_name}', filename = 'epoch{epoch}-step{step}', save_top_k=-1)
    
    strategy = 'ddp_find_unused_parameters_true' if len(devices) > 1 else "auto" 
    trainer = pl.Trainer(
        devices = devices, 
        accelerator="gpu",
        num_nodes = 1,
        max_epochs=100,
        strategy = strategy,
        callbacks=[ckpt_callback],
        logger = wandb_logger,
        default_root_dir = "./weight/StableAudio"
    )
    trainer.fit(training_wrapper, dataloader)


if __name__ == '__main__':
    main()