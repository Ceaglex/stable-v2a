import json
import math
from accelerate import Accelerator
from tqdm import tqdm
import torch
import copy
from torch.utils.data import DataLoader
from safetensors.torch import load_file

from stable_audio_tools import create_model_from_config, replace_audio, save_audio
from stable_audio_tools.data.dataset import VideoFeatDataset, collation_fn
from stable_audio_tools.training.training_wrapper import DiffusionCondTrainingWrapper
from stable_audio_tools.inference.generation import generate_diffusion_cond, generate_diffusion_cond_from_path


def main():
    accelerator = Accelerator()
    device = accelerator.device

    dataset = "unav100"
    train_test = 'test'
    info_dirs = [f'./dataset/feature/{train_test}/{dataset}/10']
    output_dir = f"/home/chengxin/chengxin/{dataset}/generated_audios/stablev2a_/10"


    model_config_file = './stable_audio_tools/configs/model_config.json'
    with open(model_config_file) as f:
        model_config = json.load(f)
        sample_size = model_config['sample_size']
        sample_size = None
        sample_rate = model_config['sample_rate']
        fps = model_config['fps']

    model = create_model_from_config(model_config)
    # model.load_state_dict(load_file('./weight/StableAudio/2024-07-29 10:26:20/epoch=55-step=87.safetensors'), strict=True)
    # model.load_state_dict(load_file('./weight/StableAudio/2024-07-24 23:06:33/epoch=70-step=304.safetensors'), strict=True)
    # model.load_state_dict(load_file('./weight/StableAudio/2024-08-01 09:36:20/epoch=29-step=2818.safetensors'), strict=True)  # BEST model_config.json
    # model.load_state_dict(load_file('./weight/StableAudio/2024-08-01 09:36:20/epoch=45-step=2818.safetensors'), strict=True)    # Second BEST model_config.json
    model.load_state_dict(load_file('./weight/StableAudio/2024-08-04 02:52:24/epoch=27-step=2818.safetensors'), strict=True)

    ds_config = {
        'info_dirs' : info_dirs,
        # 'audio_dirs' : audio_dirs,
        'exts':'wav',
        'sample_rate':sample_rate, 
        'sample_size':sample_size,
        'fps':fps,
        'output_dir':output_dir,
        # 'force_channels':"mono",
        # 'limit_num':50
    }
    dl_config = {
        'batch_size':1, 
        'shuffle':False,
        'num_workers':8, 
        'persistent_workers':True, 
        'pin_memory':True, 
        'drop_last':False, 
    }
    dataset = VideoFeatDataset(**ds_config)
    dataloader = DataLoader(dataset=dataset,  collate_fn=collation_fn, **dl_config)


    model, dataloader = accelerator.prepare(model, dataloader)
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    
    split_duration = 10
    for conditioning in tqdm(dataloader, desc = f'GPU-{device}   '):
        seconds_total = conditioning['feature'].shape[1]/max(conditioning['fps'])
        output_list = []
        
        for i in range(math.ceil(seconds_total/split_duration)):
            conditioning_ = copy.deepcopy(conditioning)
            start = i*split_duration*max(conditioning['fps'])
            end = (i+1)*split_duration*max(conditioning['fps'])
            conditioning_['feature'] = conditioning['feature'][:,start:end,:]
            # print(conditioning_['feature'].shape, start, end)

            output = generate_diffusion_cond(
                model = model.to(device),
                steps=150,
                cfg_scale=7,
                conditioning=conditioning_,
                sample_size=int(sample_rate*split_duration),
                batch_size=len(conditioning_['feature']),
                sigma_min=0.3,
                sigma_max=500,
                sampler_type="dpmpp-3m-sde", # k-dpm-fast
                device=device
            )
            output_list.append(output)
        output = torch.concat(output_list, dim=-1)
        

        for idx in range(len(conditioning['feature'])):
            # Save generated audio
            video_path = conditioning['video_path'][idx].replace('../../', './')
            audio_path = f"{output_dir}/{video_path.split('/')[-1].replace('.mp4', '.wav')}"
            
            waveform = output[idx:1+idx,...,:int(conditioning['seconds_total'][idx]*sample_rate)]
            print(waveform.shape, waveform.shape[-1]/sample_rate, conditioning['seconds_total'][idx])
            save_audio(waveform, audio_path, sample_rate)
            
            # Replace the audio of original video to generated one
            # moved_video_path = f"{output_dir}/{video_path.split('/')[-1]}"
            # shutil.copy(video_path, moved_video_path)
            # generated_video_path = moved_video_path.replace(".mp4","_GEN.mp4")
            # replace_audio(moved_video_path, audio_path, generated_video_path)

if __name__ == "__main__":
    main()