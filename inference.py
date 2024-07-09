import json
from accelerate import Accelerator
from tqdm import tqdm
import torch
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
    output_dir = f"/home/chengxin/chengxin/{dataset}/generated_audios/stablev2a"


    model_config_file = './stable_audio_tools/configs/model_config.json'
    with open(model_config_file) as f:
        model_config = json.load(f)
    sample_rate = model_config["sample_rate"]
    model = create_model_from_config(model_config)
    model.load_state_dict(load_file('./weight/StableAudio/2024-07-06 21:34:56/epoch=3-step=1023.safetensors'), strict=True)


    ds_config = {
        'info_dirs' : info_dirs,
        # 'audio_dirs' : audio_dirs,
        'exts':'wav',
        'sample_rate':sample_rate, 
        'force_channels':"stereo",
        # 'limit_num':50
    }
    dl_config = {
        'batch_size':1, 
        'shuffle':False,
        'num_workers':4, 
        'persistent_workers':True, 
        'pin_memory':True, 
        'drop_last':False, 
    }
    dataset = VideoFeatDataset(**ds_config)
    dataloader = DataLoader(dataset=dataset,  collate_fn=collation_fn, **dl_config)


    model, dataloader = accelerator.prepare(model, dataloader)
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    for conditioning in tqdm(dataloader, desc = f'GPU-{device}   '):
        output = generate_diffusion_cond(
            model = model.to(device),
            steps=150,
            cfg_scale=7,
            conditioning=conditioning,
            sample_size=int(sample_rate*conditioning['duration'][0]),
            batch_size=len(conditioning['feature']),
            sigma_min=0.3,
            sigma_max=500,
            sampler_type="dpmpp-3m-sde",
            device=device
        )
        for idx in range(len(conditioning['feature'])):
            # Save generated audio
            video_path = conditioning['video_path'][idx].replace('../../', './')
            audio_path = f"{output_dir}/{video_path.split('/')[-1].replace('.mp4', '.wav')}"
            save_audio(output[idx:1+idx], audio_path, sample_rate)
            
            # Replace the audio of original video to generated one
            # moved_video_path = f"{output_dir}/{video_path.split('/')[-1]}"
            # shutil.copy(video_path, moved_video_path)
            # generated_video_path = moved_video_path.replace(".mp4","_GEN.mp4")
            # replace_audio(moved_video_path, audio_path, generated_video_path)

if __name__ == "__main__":
    main()