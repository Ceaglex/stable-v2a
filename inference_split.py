import json
import math
from accelerate import Accelerator
from tqdm import tqdm
import torch
import copy
from torch.utils.data import DataLoader
from safetensors.torch import load_file

from stable_audio_tools import create_model_from_config, replace_audio, save_audio
from stable_audio_tools.data.dataset import VideoFeatDataset, VideoFeatDataset_VL, collation_fn
from stable_audio_tools.training.training_wrapper import DiffusionCondTrainingWrapper
from stable_audio_tools.inference.generation import generate_diffusion_cond, generate_diffusion_cond_from_path
import random

def main():
    accelerator = Accelerator()
    # seed = random.randint(0, 9999999)
    seed = 555
    batchsize = 1
    device = accelerator.device


    dataset = "unav100"
    name = 'stablev2a_'
    train_test = 'test'
    cfg_scale = 5
    param = 'model8_ss60_noabs_cfg5'
    info_dirs = [f'./dataset/feature/{train_test}/{dataset}/10']
    output_dir = f"/home/chengxin/chengxin/{dataset}/generated_audios/{name}/{param}"


    # model_config_file = './stable_audio_tools/configs/model_config.json'
    # model_config_file = './stable_audio_tools/configs/model_config_gc16000.json'
    # model_config_file = './stable_audio_tools/configs/model_config_vl30.json'
    # model_config_file = './stable_audio_tools/configs/model_config_ss30.json'
    model_config_file = './stable_audio_tools/configs/model_config_vl30_noabs.json'


    with open(model_config_file) as f:
        model_config = json.load(f)
        sample_size = model_config['sample_size']
        sample_rate = model_config['sample_rate']
        variable_length = model_config['variable_length']
        fps = model_config['fps']
        force_channels = "stereo" if model_config["audio_channels"] == 2 else "mono"
        sample_size = sample_rate*60
        variable_length = None
    print(sample_size, sample_rate, sample_size/sample_rate, variable_length, fps, force_channels, cfg_scale)

    model = create_model_from_config(model_config)
    # model.load_state_dict(load_file('./weight/StableAudio/2024-07-29 10:26:20/epoch=55-step=87.safetensors'), strict=True)
    # model.load_state_dict(load_file('./weight/StableAudio/2024-07-24 23:06:33/epoch=70-step=304.safetensors'), strict=True)
    # model.load_state_dict(load_file('./weight/StableAudio/2024-08-01 09:36:20/epoch=29-step=2818.safetensors'), strict=True)  # SECOND BEST model_config.json
    # model.load_state_dict(load_file('./weight/StableAudio/2024-08-01 09:36:20/epoch=45-step=2818.safetensors'), strict=True)    # BEST model_config.json
    # model.load_state_dict(load_file('./weight/StableAudio/2024-08-04 02:52:24/epoch=36-step=2818.safetensors'), strict=True)    # BEST model_config.json
    model.load_state_dict(load_file('./weight/StableAudio/2024-08-04 02:52:24/epoch=60-step=2818.safetensors'), strict=False)    # BEST model_config.json
    # model.load_state_dict(load_file('./weight/StableAudio/2024-08-19 11:11:56/epoch=60-step=2818.safetensors'), strict=True)    # BEST model_config_gc16000.json
    # model.load_state_dict(load_file('./weight/StableAudio/2024-08-24 20:27:19/epoch=0-step=1409.safetensors'), strict=True)       # BEST model_config_vl30.json
    # model.load_state_dict(load_file('./weight/StableAudio/2024-09-07 00:11:02/epoch=0-step=101.safetensors'), strict=True)       # model_config_ss30.json

    ds_config = {
        'info_dirs' : info_dirs,
        # 'audio_dirs' : audio_dirs,
        'exts':'wav',
        'sample_rate':sample_rate, 
        'sample_size':sample_size,
        'variable_length':variable_length,
        'fps':fps,
        'output_dir':output_dir,
        'force_channels':force_channels,
        # 'limit_num':50
    }
    dl_config = {
        'batch_size':batchsize, 
        'shuffle':False,
        'num_workers':8, 
        'persistent_workers':True, 
        'pin_memory':True, 
        'drop_last':False, 
    }
    # dataset = VideoFeatDataset(**ds_config)
    dataset = VideoFeatDataset_VL(**ds_config)
    dataloader = DataLoader(dataset=dataset,  collate_fn=collation_fn, **dl_config)


    model, dataloader = accelerator.prepare(model, dataloader)
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    split_duration = int(sample_size/sample_rate)
    print(f"split duration: {split_duration}")

    for conditioning in tqdm(dataloader, desc = f'GPU-{device}   '):

        seconds_total = conditioning['feature'].shape[1]/fps
        output_list = []
        
        for i in range(math.ceil(seconds_total/split_duration)):    
            split_duration_ = split_duration
            if i == math.ceil(seconds_total/split_duration)-1:
                split_duration_ = max(10,int(seconds_total-i*split_duration))
            print(f"split duration: {split_duration_}")

            conditioning_ = copy.deepcopy(conditioning)
            start = i*split_duration*fps
            end = start + split_duration_*fps

            print(split_duration_, (end-start)/fps, int(seconds_total), i)
            # conditioning_['seconds_total'] = torch.tensor([[10]])
            conditioning_['seconds_total'] = torch.tensor([[split_duration_ for _ in range(batchsize)]])
            conditioning_['feature'] = conditioning['feature'][:,start:end,:]
            # print(conditioning_['feature'].shape, conditioning['feature'].shape, start, end, conditioning_['feature'][0])
        
        # break
            output = generate_diffusion_cond(
                model = model.to(device),
                steps=150,
                cfg_scale=cfg_scale,
                conditioning=conditioning_,
                sample_size=int(sample_rate*split_duration_),
                batch_size=len(conditioning_['feature']),
                sigma_min=0.3,
                sigma_max=500,
                sampler_type="dpmpp-3m-sde", # k-dpm-fast
                device=device,
                seed = seed
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