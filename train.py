import json
import pytorch_lightning as pl
from einops import rearrange
from torch.utils.data import DataLoader
from safetensors.torch import load_file
from datetime import datetime

from stable_audio_tools import create_model_from_config
from stable_audio_tools.data.dataset import VideoFeatDataset, VideoFeatDataset_VL, collation_fn
from stable_audio_tools.training.training_wrapper import DiffusionCondTrainingWrapper, DiffusionCondDemoCallback
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# 注意修改 pl.callbacks.ModelCheckpoint-every_n_train_steps every_n_epochs以及 DiffusionCondDemoCallback-every_n_epochs
# 注意修改 devices 和 CUCUDA_VISIBLE_DEVICES
def main():
    # model_config_file = './stable_audio_tools/configs/model_config.json'
    # model_config_file = './stable_audio_tools/configs/model_config_gc16000.json'
    # model_config_file = './stable_audio_tools/configs/model_config_gc16000_vl30.json'
    model_config_file = './stable_audio_tools/configs/model_config_sr16000.json'

    with open(model_config_file) as f:
        model_config = json.load(f)
        sample_rate = model_config["sample_rate"]
        sample_size = model_config["sample_size"]
        variable_length = model_config["variable_length"]
        fps = model_config["fps"]
        force_channels = "stereo" if model_config["audio_channels"] == 2 else "mono"
        # sample_size = None
        # variable_length = None
    print(sample_size, sample_rate, sample_size/sample_rate, variable_length, fps)
    model = create_model_from_config(model_config)


    # model.load_state_dict(load_file('./weight/StableAudio/lightning_logs/version_3/checkpoints/epoch=9-step=14620.safetensors'), strict=True)   # 第一版有成果的v2a模型
    # model.load_state_dict(load_file('./weight/StableAudio/2024-07-02 19:49:04/epoch=2-step=427.safetensors'), strict=False)   # 一个还不错的v2a模型, 可以用作训练的base
    # model.load_state_dict(load_file('./weight/StableAudio/2024-08-01 09:36:20/epoch=29-step=2818.safetensors'), strict=True)  # x和cond上加入了pos_ebd, 利用VGG进行训练  
    # model.load_state_dict(load_file('./weight/StableAudio/2024-08-04 02:52:24/epoch=36-step=2818.safetensors'), strict=True)  # x和cond上加入了pos_ebd, 利用VGG进行训练 BEST MODEL
    # model.load_state_dict(load_file('./weight/StableAudio/2024-08-04 02:52:24/epoch=60-step=2818.safetensors'), strict=True)  # x和cond上加入了pos_ebd, 利用VGG进行训练 不错 可以接着训练试试
    # model.load_state_dict(load_file('./weight/StableAudio/2024-08-19 11:11:56/epoch=60-step=2818.safetensors'), strict=True)  # model_config_gc16000.json 不错 可以接着训练试试
    model.load_state_dict(load_file('./weight/StableAudio/2024-08-04 02:52:24/epoch=60-step=2818.safetensors'), strict=True)    


    info_dirs = [
        './dataset/feature/train/AudioSet/10', 
        './dataset/feature/train/VGGSound/10',
        # './dataset/feature/train/unav100/10',
        ]
    audio_dirs = [
        '/home/chengxin/chengxin/AudioSet/generated_audios/train/10', 
        '/home/chengxin/chengxin/VGGSound/generated_audios/train/10',
        # '/home/chengxin/chengxin/unav100/generated_audios/train/10',
        ]
    ds_config = {
        'info_dirs' : info_dirs,
        'audio_dirs' : audio_dirs,
        'exts':'wav',
        'sample_rate':sample_rate, 
        'sample_size':sample_size,
        'variable_length':variable_length,
        'fps':fps,
        'force_channels':force_channels,
        # 'limit_num':7000
    }

    dl_config = {
        'batch_size':16, 
        'shuffle':True,
        'num_workers':4, 
        'persistent_workers':True, 
        'pin_memory':True, 
        'drop_last':False, 
    }
    train_dataset = VideoFeatDataset_VL(**ds_config)
    train_dataloader = DataLoader(dataset=train_dataset,  collate_fn=collation_fn, **dl_config)



    info_dirs = [
        './dataset/feature/test/VGGSound/10',
    ]
    audio_dirs = [
        '/home/chengxin/chengxin/VGGSound/generated_audios/test/10',
    ]
    ds_config = {
        'info_dirs' : info_dirs,
        'audio_dirs' : audio_dirs,
        'exts':'wav',
        'sample_rate':sample_rate, 
        'sample_size':sample_size,
        # 'variable_length':variable_length,
        'fps':fps,
        # 'force_channels':"mono",
        'limit_num':32
    }
    dl_config = {
        'batch_size':32, 
        'shuffle':False,
        'num_workers':4, 
        'persistent_workers':True, 
        'pin_memory':True, 
        'drop_last':False, 
    }
    test_dataset = VideoFeatDataset_VL(**ds_config)
    test_dataloader = DataLoader(dataset=test_dataset,  collate_fn=collation_fn, **dl_config)


    training_config = model_config.get('training', None)
    training_wrapper = DiffusionCondTrainingWrapper(
                model=model, 
                lr=training_config.get("learning_rate", None),
                optimizer_configs=training_config.get("optimizer_configs", None),
                pre_encoded=training_config.get("pre_encoded", False),
                cfg_dropout_prob = training_config.get("cfg_dropout_prob", 0.1),
                timestep_sampler = training_config.get("timestep_sampler", "uniform"),
                # duration_mask=None,
                latent_per_sec = sample_rate // 2000
            )


    run_name = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    wandb_logger = pl.loggers.WandbLogger(project="stable-v2a", name = run_name, save_dir="./weight/StableAudio")
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_epochs=5, dirpath=f'./weight/StableAudio/{run_name}', filename = '{epoch}-{step}', save_top_k=-1)
    demo_callback = DiffusionCondDemoCallback(test_dataloader=test_dataloader, sample_size = sample_size, every_n_epochs=5, sample_rate=sample_rate)

    
    devices = [1] 


    strategy = 'ddp_find_unused_parameters_true' if len(devices) > 1 else "auto" 
    trainer = pl.Trainer(
        devices = devices, 
        accelerator="gpu",
        num_nodes = 1,
        max_epochs=100,
        strategy = strategy,
        callbacks=[demo_callback, ckpt_callback],
        logger = wandb_logger,
        default_root_dir = "./weight/StableAudio"
    )
    trainer.fit(training_wrapper, train_dataloader)


if __name__ == '__main__':
    main()