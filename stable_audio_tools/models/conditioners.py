#Heavily influenced by https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/modules/conditioners.py

import torch
import typing as tp

from .adp import NumberEmbedder

from torch import nn
import clip
from moviepy.editor import VideoFileClip
from PIL import Image
import torch.nn.functional as F


class Conditioner(nn.Module):
    def __init__(
            self
        ):
        
        super().__init__()

    def forward(self, x: tp.Any) -> tp.Any:
        raise NotImplementedError()
    


class NumberConditioner(Conditioner):
    '''
        Conditioner that takes a list of floats, normalizes them for a given range, and returns a list of embeddings
    '''
    def __init__(self, 
                output_dim: int,
                min_val: float=0,
                max_val: float=1
                ):
        super().__init__()

        self.min_val = min_val
        self.max_val = max_val

        self.embedder = NumberEmbedder(features=output_dim)

    def forward(self, floats: torch.Tensor, device=None) -> tp.Any:
    
            # Cast the inputs to floats
            floats = floats.to(torch.float).to(device)

            floats = floats.clamp(self.min_val, self.max_val)
    
            normalized_floats = (floats - self.min_val) / (self.max_val - self.min_val)

            # Cast floats to same type as embedder
            embedder_dtype = next(self.embedder.parameters()).dtype
            normalized_floats = normalized_floats.to(embedder_dtype)
            
            # [batchsize, 1, output_dim]
            float_embeds = self.embedder(normalized_floats) 

            return [float_embeds, torch.ones(float_embeds.shape[0], 1).to(device)]



class CLIPFeatConditioner(Conditioner):
    def __init__(self,
                 fps :int = 22,
                 clip_name :str = None,
                 download_root :str = None,
                 **kargs):
        '''
            Get the video feature from restored feature or videos
        '''
        super().__init__()
        self.fps = fps
        self.clip_name = clip_name
        self.download_root = download_root
        # self.clip_model = None
        # self.preprocess = None

    def forward(self, data, device=None) -> tp.Any:
        if isinstance(data[0], str):
            with torch.no_grad():
                # if self.clip_model == None:
                clip_model,  preprocess = clip.load(self.clip_name, device=device, download_root=self.download_root)
                # elif next(self.clip_model.parameters()).device.index != device:
                clip_model.to(device)

                images_features = []
                for mp4_file in data:
                    video = VideoFileClip(mp4_file)
                    video = video.set_fps(self.fps)
                    print(f"Extracting features from video:{mp4_file}")

                    frames = []
                    image_features = []
                    for idx, frame in enumerate(video.iter_frames()):
                        frame = Image.fromarray(frame)
                        frame = preprocess(frame).unsqueeze(0)
                        frames.append(frame)

                        # do it in batch to avoid OOM
                        if idx % 240 == 239  or  idx == int(video.duration*video.fps)-1:
                            frames = torch.cat(frames).to(device)
                            frame_batchsize = 20
                            for i in range(len(frames)//frame_batchsize + 1):
                                start_id = i*frame_batchsize
                                end_id = (i+1)*frame_batchsize
                                if start_id >= len(frames):
                                    break
                                image_features.append(clip_model.encode_image(frames[start_id:end_id]))
                            # del frames
                            # torch.cuda.empty_cache()
                            frames = []

                    image_features = torch.concat(image_features).to(device)
                    images_features.append(image_features)                    

                max_length = max([image_features.shape[0] for image_features in images_features])
                images_features = torch.stack([
                    F.pad(image_features, (0, 0, 0, max_length - image_features.shape[0]), mode='constant', value=0) 
                    for image_features in images_features
                ])

                return [images_features.to(device),               torch.ones(images_features.shape[0], images_features.shape[1]).to(device)]
        
        elif isinstance(data[0], torch.Tensor):
            # TODO: attention MASK ?????
            # [batchsize, frame_num, output_dim]   [batchsize, frame_num]
            return [data.to(device),               torch.ones(data.shape[0], data.shape[1]).to(device)]  
        
        else:
            raise ValueError("data type for CLIPConditioner can only be str(directory path) or tensor(restored feature)")


class MultiConditioner(nn.Module):
    """
    A module that applies multiple conditioners to an input dictionary based on the keys

    Args:
        conditioners: a dictionary of conditioners with keys corresponding to the keys of the conditioning input dictionary (e.g. "prompt")
        default_keys: a dictionary of default keys to use if the key is emmpty 
                      (e.g. {"feature": "video_path"}, When the dict['feature'] is empty, use dict['video_path'] as the input to the conditioner)
    """
    def __init__(self, 
                 conditioners: tp.Dict[str, Conditioner], 
                 default_keys: tp.Dict[str, str] = {},
                 ):
        super().__init__()

        self.conditioners = nn.ModuleDict(conditioners)
        self.default_keys = default_keys

    def forward(self, batch_metadata: tp.Dict[str, tp.Any], device: tp.Union[torch.device, str]) -> tp.Dict[str, tp.Any]:
        output = {}
        for key, conditioner in self.conditioners.items():
            if key in batch_metadata and len(batch_metadata[key]) != 0:
                output[key] = conditioner(batch_metadata[key], device)
            elif key in self.default_keys:
                key_ = self.default_keys[key]
                output[key] = conditioner(batch_metadata[key_], device)
            else:
                continue
            
        
        if len(output) == 0:
            print("No conditioned is specified !!")
            raise    
        return output
    
    
def create_multi_conditioner_from_conditioning_config(config: tp.Dict[str, tp.Any]) -> MultiConditioner:
    """
    Create a MultiConditioner from a conditioning config dictionary

    Args:
        config: the conditioning config dictionary
        device: the device to put the conditioners on
    """
    conditioners = {}
    cond_dim = config["cond_dim"]
    default_keys = config.get("default_keys", {})

    for conditioner_info in config["configs"]:
        id = conditioner_info["id"]
        conditioner_type = conditioner_info["type"]
        conditioner_config = {"output_dim": cond_dim}
        conditioner_config.update(conditioner_info["config"])

        if conditioner_type == "number":
            conditioners[id] = NumberConditioner(**conditioner_config)
        elif conditioner_type == "video_feature":
            conditioners[id] = CLIPFeatConditioner(**conditioner_config)

    return MultiConditioner(conditioners, default_keys=default_keys)