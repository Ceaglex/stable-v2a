import importlib
import numpy as np
import io
import os
import posixpath
import random
import re
import subprocess
import time
import torch
import torchaudio
import torch.nn.functional as F

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
                stack_keys = ['fps','duration', 'frame_num']
                pad_keys = ['feature']
                list_keys = ['video_path']

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


class VideoFeatDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        info_dirs,
        audio_dirs,
        exts='wav',
        sample_rate=44100, 
        # sample_size=2097152, 
        # random_crop=True,
        force_channels="stereo"
    ):
        super().__init__()

        self.all_file_info = self.get_audio_info(info_dirs, audio_dirs, exts) 
        # dict("video_path", "fps", "duration", "frame_num", "feature", "audio_path")


        # self.pad_crop = PadCrop_Normalized_T(sample_size, sample_rate, randomize=random_crop)
        self.force_channels = force_channels
        self.encoding = torch.nn.Sequential(
            Stereo() if self.force_channels == "stereo" else torch.nn.Identity(),
            Mono() if self.force_channels == "mono" else torch.nn.Identity(),
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
                        exts = 'wav'          # extention for each audio dir
                        ):

        file_info = []

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

            for pickle_path in glob.glob(os.path.join(info_dir, "*.pickle"))[:10]:
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
                    file_info.append(info)
        return file_info



    def __len__(self):
        return len(self.all_file_info)
    


    def __getitem__(self, idx):
        info_dict = self.all_file_info[idx]
        audio_file = info_dict['audio_path']
        try:
            # TODO: Add load audio
            audio = self.load_file(audio_file)
            info_dict['duration'] = round(audio.shape[1]/self.sr, 3)

            # TODO: Add pad_crop and augs
            # audio, t_start, t_end, seconds_start, seconds_total, padding_mask = self.pad_crop(audio)
            # if self.augs is not None:
            #     audio = self.augs(audio)

            audio = audio.clamp(-1, 1)

            # Encode the file to assist in prediction
            if self.encoding is not None:
                audio = self.encoding(audio)

            return (audio, info_dict)
        
        except Exception as e:
            print(f'Couldn\'t load file {audio_file}: {e}')
            return self[random.randrange(len(self))]









