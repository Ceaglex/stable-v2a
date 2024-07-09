from moviepy.editor import VideoFileClip
from PIL import Image
import glob
import os
import torch
import pickle
from PIL import Image
import clip
from tqdm import tqdm
from accelerate import Accelerator


import torch
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self,
                 mp4_dir, 
                 pickle_dir,
                 recalculate = True):
        self.pickle_dir = pickle_dir
        self.recalculate = recalculate
        self.mp4_files = glob.glob(os.path.join(mp4_dir, '*.mp4'))    ######

    def __len__(self):
        return len(self.mp4_files)

    def __getitem__(self, idx):
        mp4_file = self.mp4_files[idx]
        pickle_file = os.path.join(self.pickle_dir, os.path.basename(mp4_file)[:-4] + ".pickle")

        if self.recalculate or not os.path.exists(pickle_file):
            return mp4_file, pickle_file
        else:
            return None, None

    



def main(mp4_dir,
        pickle_dir,
        frame_size = 20,
        fps = 22, 
        device = 0,
        recalculate = True):
    
    accelerator = Accelerator()
    device = accelerator.device

    with torch.no_grad():
        model, preprocess = clip.load("ViT-L/14", device=device, download_root='../../weight/CLIP')
        dataset = AudioDataset(mp4_dir=mp4_dir, pickle_dir=pickle_dir,  recalculate = recalculate)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle=False)

        model, dataloader = accelerator.prepare(model, dataloader)
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        
        for mp4_file, pickle_file in tqdm(dataloader, desc = f'GPU {device}'):
            if mp4_file == None:
                continue
            mp4_file = mp4_file[0]
            pickle_file = pickle_file[0]
            video = VideoFileClip(mp4_file)
            video = video.set_fps(fps)

            frames = []
            image_features = []
            feature_dict = {}

            if video.duration > 12:
                print(f"{video.duration} secs, {mp4_file} TOO LONG FOR VGG!!")
                continue
            for idx, frame in enumerate(video.iter_frames()):
                if idx % frame_size == frame_size-1 or idx == int(video.duration*video.fps)-1:
                    frames = torch.cat(frames).to(device)
                    image_features.append(model.encode_image(frames))
                    frames = []
                frame = Image.fromarray(frame)
                frame = preprocess(frame).unsqueeze(0)
                frames.append(frame)



            image_features = torch.concat(image_features).cpu()
            print(image_features.shape, video.duration)
            feature_dict['feature'] = image_features
            feature_dict['video_path'] = mp4_file
            feature_dict['pickle_path'] = pickle_file
            feature_dict['fps'] = fps
            feature_dict['duration'] = video.duration
            feature_dict['frame_num'] = image_features.shape[0]

            with open(feature_dict['pickle_path'], 'wb') as handle:
                pickle.dump(feature_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            
            

                        



if __name__ == "__main__":
    train_test = 'train'
    dataset = 'VGGSound'
    duration = 10
    # mp4_dir = f'../../../{dataset}/dataset/{train_test}'
    mp4_dir = f'../../../{dataset}/dataset/{train_test}/{duration}'
    pickle_dir = f'../../dataset/feature/{train_test}/{dataset}/{duration}'

    frame_size = 20
    main(mp4_dir = mp4_dir, pickle_dir = pickle_dir, frame_size = frame_size)

