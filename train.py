from prefigure.prefigure import get_all_args, push_wandb_config
import json
import os
import torch
import pytorch_lightning as pl
import random

from stable_audio_tools.data.dataset import VideoFeatDataset, collation_fn
from torch.utils.data import DataLoader


def main():

    info_dirs = ['/home/chengxin/chengxin/stable-v2a/dataset/video_feature/test/AudioSet']
    audio_dirs = ['/home/chengxin/chengxin/AudioSet/generated_audios/test/10']

    ds_config = {
        'info_dirs' : ['/home/chengxin/chengxin/stable-v2a/dataset/video_feature/test/AudioSet'],
        'audio_dirs' : ['/home/chengxin/chengxin/AudioSet/generated_audios/test/10'],
        'exts':'wav',
        'sample_rate':44100, 
        'force_channels':"stereo"
    }
    dl_config = {
        'batch_size':16, 
        'shuffle':True,
        'num_workers':4, 
        'persistent_workers':True, 
        'pin_memory':True, 
        'drop_last':False, 
    }
    dataset = VideoFeatDataset(**ds_config)
    dataloader = DataLoader(dataset=dataset,  collate_fn=collation_fn, **dl_config)




    for x, metadata in dataloader:
        print(x.shape, metadata['fps'].shape, metadata['duration'].shape, metadata['frame_num'].shape, metadata['feature'].shape)

if __name__ == '__main__':
    main()