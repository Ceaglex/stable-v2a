#Heavily influenced by https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/modules/conditioners.py

import torch
import typing as tp

from .adp import NumberEmbedder

from torch import nn

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
                 **kargs):
        '''
            Get the video feature from restored feature or videos
        '''
        super().__init__()

    def forward(self, data, device=None) -> tp.Any:
        if isinstance(data[0], str):
            pass
        elif isinstance(data[0], torch.Tensor):
            # TODO: attention MASK ?????
            # [batchsize, frame_num, output_dim]
            return [data.to(device), torch.ones(data.shape[0], data.shape[1]).to(device)]  
        else:
            raise ValueError("data type for CLIPConditioner can only be str(directory path) or tensor(restored feature)")


class MultiConditioner(nn.Module):
    """
    A module that applies multiple conditioners to an input dictionary based on the keys

    Args:
        conditioners: a dictionary of conditioners with keys corresponding to the keys of the conditioning input dictionary (e.g. "prompt")
        default_keys: a dictionary of default keys to use if the key is not in the input dictionary (e.g. {"prompt_t5": "prompt"})
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
                pass
            elif key in self.default_keys:
                key = self.default_keys[key]
            else:
                continue
            output[key] = conditioner(batch_metadata[key], device)
        
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