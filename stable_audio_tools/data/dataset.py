import importlib
import numpy as np
import io
import os
import posixpath
import random
import re
import subprocess
import time
import copy
import torch
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm
import webdataset as wds
import glob
import pickle

from aeiou.core import is_silence
from os import path
from pedalboard.io import AudioFile
from torchaudio import transforms as T
from typing import Optional, Callable, List

from .utils import Stereo, Mono, PhaseFlipper, PadCrop_Normalized_T


def collation_fn(samples):
    # stack_keys = ['fps','duration', 'frame_num']
    stack_keys = ['fps','seconds_start', 'seconds_total']
    pad_keys = ['feature']
    # list_keys = ['video_path', 'time_cond']
    list_keys = ['video_path']

    if type(samples[0]) == tuple:
        batched = list(zip(*samples))
        # batch(tuple(tensor), tuple(dict) )
        # length of tuple == batchsize
        result = []
        for b in batched:
            if isinstance(b[0], torch.Tensor):
                max_length = max([audio.shape[1] for audio in b])
                b = torch.stack([
                            F.pad(audio, (0, max_length - audio.shape[-1]), mode='constant', value=0) 
                            for audio in b
                        ])
                result.append(b)

            if isinstance(b[0], dict):
                combined_dict = {}

                for key in stack_keys:
                    combined_dict[key] = torch.tensor([[metadata[key]] for metadata in b])
                for key in pad_keys:
                    max_length = max([metadata[key].shape[0] for metadata in b])
                    combined_dict[key] = torch.stack([
                        F.pad(metadata[key], (0, 0, 0, max_length - metadata[key].shape[0]), mode='constant', value=0) 
                        for metadata in b
                    ])
                for key in list_keys:
                    combined_dict[key] = [metadata[key] for metadata in b]
                result.append(combined_dict)
        return result
    

    elif type(samples[0]) == dict:
        b = samples
        combined_dict = {}

        for key in stack_keys:
            combined_dict[key] = torch.tensor([[metadata[key]] for metadata in b])
        for key in pad_keys:
            max_length = max([metadata[key].shape[0] for metadata in b])
            combined_dict[key] = torch.stack([
                F.pad(metadata[key], (0, 0, 0, max_length - metadata[key].shape[0]), mode='constant', value=0) 
                for metadata in b
            ])
        for key in list_keys:
            combined_dict[key] = [metadata[key] for metadata in b]
        return combined_dict
        


class VideoFeatDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        info_dirs,
        audio_dirs = None,
        exts='wav',
        sample_rate=44100, 
        fps=10,
        sample_size=None, 
        random_crop=True,
        force_channels="stereo",
        limit_num = None,
        output_dir = None,
    ):
        super().__init__()
        self.audio_dirs = audio_dirs
        self.fps = fps
        self.sample_size = sample_size
        self.output_dir = output_dir
        self.all_file_info = self.get_audio_info(info_dirs, audio_dirs, exts, limit_num) 
        # dict("video_path", "fps", "duration", "frame_num", "feature", "audio_path")


        # self.pad_crop = PadCrop_Normalized_T(sample_size, sample_rate, randomize=random_crop)
        self.force_channels = force_channels
        self.encoding = torch.nn.Sequential(
            Stereo() if self.force_channels == "stereo" else torch.nn.Identity(),
            Mono(2) if self.force_channels == "mono" else torch.nn.Identity(),
        )

        self.sr = sample_rate
        
        print(f'Found {len(self.all_file_info)} files')



    def load_file(self, filename):
        ext = filename.split(".")[-1]

        if ext == "mp3":
            with AudioFile(filename) as f:
                audio = f.read(f.frames)
                audio = torch.from_numpy(audio)
                in_sr = f.samplerate
        else:
            audio, in_sr = torchaudio.load(filename, format=ext)
        if in_sr != self.sr:
            resample_tf = T.Resample(in_sr, self.sr)
            audio = resample_tf(audio)

        return audio



    def  get_audio_info(self,
                        info_dirs,  # directories to store pickle info
                        audio_dirs,   # directories to store audio files
                        exts = 'wav',  # extention for each audio dir
                        limit_num = None,
                        ):
        file_info = []

        if audio_dirs != None:
            if type(info_dirs) is str:
                info_dirs = [info_dirs]
            if type(audio_dirs) is str:
                audio_dirs = [audio_dirs]
            if type(exts) is str:
                exts = [exts for _ in range(len(audio_dirs))]

            assert len(audio_dirs) == len(info_dirs), "Infomation Directories list should match the Audio File Directories list !!"

            for i in range(len(info_dirs)):
                info_dir = info_dirs[i]
                audio_dir = audio_dirs[i]
                ext = f".{exts[i]}"

                pickle_paths = glob.glob(os.path.join(info_dir, "*.pickle"))
                if limit_num:
                    pickle_paths = pickle_paths[:limit_num]
                    
                for pickle_path in pickle_paths:  #########
                    audio_name = os.path.basename(pickle_path).replace('.pickle', ext)
                    audio_path = os.path.join(audio_dir, audio_name)
                    if not os.path.exists(audio_path):
                        continue
                    with open(pickle_path, 'rb') as file:
                        info = pickle.load(file)
                        info.update({"audio_path":audio_path})
                        for key, item in info.items():
                            if isinstance(info[key], torch.Tensor):
                                info[key] = item.cpu().detach()
                        if self.output_dir!=None and os.path.exists(f"{self.output_dir}/{audio_name}"):
                            continue
                        file_info.append(info)


        else:
            if type(info_dirs) is str:
                info_dirs = [info_dirs]
            if type(exts) is str:
                exts = [exts for _ in range(len(info_dirs))]

            for i in range(len(info_dirs)):
                info_dir = info_dirs[i]
                ext = f".{exts[i]}"

                pickle_paths = glob.glob(os.path.join(info_dir, "*.pickle"))
                if limit_num:
                    pickle_paths = pickle_paths[:limit_num]
                    
                for pickle_path in pickle_paths:  #########
                    audio_name = os.path.basename(pickle_path).replace('.pickle', ext)

                    with open(pickle_path, 'rb') as file:
                        info = pickle.load(file)
                        for key, item in info.items():
                            if isinstance(info[key], torch.Tensor):
                                info[key] = item.cpu().detach()
                        if self.output_dir!=None and os.path.exists(f"{self.output_dir}/{audio_name}"):
                            continue
                        file_info.append(info)
        return file_info



    def __len__(self):
        return len(self.all_file_info)
    


    def __getitem__(self, idx):
        info_dict = copy.deepcopy(self.all_file_info[idx])
        try:
            # process
            info_dict['feature'] = info_dict['feature'][np.linspace(0, info_dict['feature'].shape[0], int(info_dict['feature'].shape[0]/info_dict['fps']*self.fps), endpoint=False, dtype=int)]
            info_dict['seconds_total'] = info_dict['duration']
            info_dict['seconds_start'] = 0
            # del info_dict['fps']
            del info_dict['frame_num']
            del info_dict['duration']

            if self.audio_dirs == None:
                return info_dict
            
            # TODO: Add load audio
            audio_file = info_dict['audio_path']
            audio = self.load_file(audio_file)
            if self.encoding is not None:
                audio = self.encoding(audio)
            info_dict['seconds_total'] = round(audio.shape[1]/self.sr, 3)
            audio = audio.clamp(-1, 1)
            if self.sample_size:
                audio = torch.concat([audio, torch.zeros([2, self.sample_size - audio.shape[1]])], dim=1)
            
            
            # Encode the file to assist in prediction

    
            return (audio, info_dict)
        
        except Exception as e:
            print(f'Couldn\'t load file {audio_file}: {e}')
            return self[random.randrange(len(self))]









