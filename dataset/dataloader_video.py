import os
import cv2
import sys
import pdb
import six
import glob
import time
import torch
import random
import pandas
import warnings
from torchvision import transforms
import torch.nn.functional as F
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
# import pyarrow as pa
from PIL import Image
import torch.utils.data as data
import matplotlib.pyplot as plt
from utils import video_augmentation
from torch.utils.data.sampler import Sampler

sys.path.append("..")
global kernel_sizes 



class BaseFeeder(data.Dataset):
    def __init__(self, prefix, gloss_dict: dict[str, list[int]], dataset='phoenix2014', drop_ratio=1, num_gloss=-1, mode="train", transform_mode=True,
                 datatype="lmdb", frame_interval=1, image_scale=1.0, kernel_size=1, input_size=224):
        self.mode = mode
        self.ng = num_gloss
        self.prefix = prefix
        self.dict = gloss_dict
        self.data_type = datatype
        self.dataset = dataset
        self.input_size = input_size
        global kernel_sizes 
        kernel_sizes = kernel_size
        self.frame_interval = frame_interval # not implemented for read_features()
        self.image_scale = image_scale # not implemented for read_features()
        self.feat_prefix = f"{prefix}/features/fullFrame-256x256px/{mode}"
        self.transform_mode = "train" if transform_mode else "test"
        self.inputs_list = np.load(f"./preprocess/{dataset}/{mode}_info.npy", allow_pickle=True).item()
        print(mode, len(self))
        self.data_aug = self.transform()
        print("")

    def __getitem__(self, idx):
        if not hasattr(self, '_cache'):
            self._cache = {}

        if idx in self._cache:
            return self._cache[idx]

        if self.data_type == "video":
            input_data, label, text = self.read_video(idx)
            input_data, label = self.normalize(input_data, label)
            result = (input_data, torch.LongTensor(label), text, self.inputs_list[idx]['original_info'])
        elif self.data_type == "lmdb":
            raise NotImplementedError()
            input_data, label, fi = self.read_lmdb(idx)
            input_data, label = self.normalize(input_data, label)
            result = (input_data, torch.LongTensor(label), self.inputs_list[idx]['original_info'])
        else:
            input_data, label = self.read_features(idx)
            result = (input_data, label, self.inputs_list[idx]['original_info'])

        self._cache[idx] = result
        return result

    def read_video(self, index):
        # load file info
        file_info = self.inputs_list[index]
        if 'phoenix' in self.dataset:
            img_folder = os.path.join(self.prefix, "features/fullFrame-256x256px/" + file_info['folder'])  
        elif self.dataset == 'how2sign':
            img_folder = os.path.join(self.prefix, file_info['folder'] + '/*')
        img_list = sorted(glob.glob(img_folder))
        img_list = img_list[int(torch.randint(0, self.frame_interval, [1]))::self.frame_interval]
        label_list: list[int] = []
        tokens = file_info["tokens"] if 'tokens' in file_info else file_info['label'].split(" ")
        for phase in tokens:
            if phase == '':
                continue
            if phase in self.dict.keys():
                label_list.append(self.dict[phase][0])
        
        if len(img_list) == 0: raise Exception(f"NO IMAGE UNDER {img_folder} path")

        return [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_list], label_list, None if 'label' not in file_info else file_info['label']

    def read_features(self, index):
        # load file info
        fi = self.inputs_list[index]
        data = np.load(f"./features/{self.mode}/{fi['fileid']}_features.npy", allow_pickle=True).item()
        return data['features'], data['label']

    def normalize(self, video, label, file_id=None):
        video = self.data_aug(video)
        # video = video.float() / 127.5 - 1 # made each pixel from -1 to 1 (moved from range 0 to 255)
        T = video.size(0)
        # print("adding padding", T)
        # if T < 2592:
        #     pad_size = 2592 - T
        #     padding = (0, 0, 0, 0, 0, 0, 0, pad_size)  # Pad at end of time dimension
        #     video = F.pad(video, padding, mode='constant', value=0)
        # print(video.shape)
        # print("original shape", video.shape)
        if T > 200 or True:
            x5d = video.permute(1, 0, 2, 3).unsqueeze(0)

            # Interpolate only along the “depth” axis (T)
            scale = 200 / video.shape[0]
            x5d_up = F.interpolate(x5d,
                                scale_factor=(scale, 1, 1),   # (T, H, W) factors
                                mode="trilinear",
                                align_corners=False)

            # (1, C, T’, H, W) → (T’, C, H, W)
            video = x5d_up.squeeze(0).permute(1, 0, 2, 3)

        return video, label

    def transform(self):
        if self.transform_mode == "train":
            print("Apply training transform.")
            return transforms.Compose([
                video_augmentation.CenterCrop(self.input_size),
                # video_augmentation.RandomCrop(self.input_size),
                video_augmentation.Resize(self.image_scale),
                video_augmentation.ToTensor(),
                video_augmentation.NormalizeVideo(),
                # video_augmentation.TemporalRescale(0.2, self.frame_interval),
            ])
        else:
            print("Apply testing transform.")
            return transforms.Compose([
                video_augmentation.CenterCrop(self.input_size),
                video_augmentation.Resize(self.image_scale),
                video_augmentation.ToTensor(),
                video_augmentation.NormalizeVideo(),
            ])

    @staticmethod
    def collate_fn(batch):
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, label, text, info = list(zip(*batch))
        
        left_pad = 0
        last_stride = 1
        total_stride = 1
        global kernel_sizes 
        for layer_idx, ks in enumerate(kernel_sizes):
            if ks[0] == 'K':
                left_pad = left_pad * last_stride 
                left_pad += int((int(ks[1])-1)/2)
            elif ks[0] == 'P':
                last_stride = int(ks[1])
                total_stride = total_stride * last_stride
        if len(video[0].shape) > 3:
            max_len = len(video[0])
            video_length = torch.LongTensor([np.ceil(len(vid) / total_stride) * total_stride + 2*left_pad for vid in video])
            right_pad = int(np.ceil(max_len / total_stride)) * total_stride - max_len + left_pad
            max_len = max_len + left_pad + right_pad
            padded_video = [torch.cat(
                (
                    vid[0][None].expand(left_pad, -1, -1, -1),
                    vid,
                    vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video)
        else:
            max_len = len(video[0])
            video_length = torch.LongTensor([len(vid) for vid in video])
            padded_video = [torch.cat(
                (
                    vid,
                    vid[-1][None].expand(max_len - len(vid), -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video).permute(0, 2, 1)
        label_length = torch.LongTensor([len(lab) for lab in label])
        if max(label_length) == 0:
            return padded_video, video_length, [], [], info
        else:
            padded_label = []
            for lab in label:
                padded_label.extend(lab)
            padded_label = torch.LongTensor(padded_label)
            return padded_video, video_length, padded_label, label_length, text, info

    def __len__(self):
        return len(self.inputs_list) - 1

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time


if __name__ == "__main__":
    feeder = BaseFeeder()
    dataloader = torch.utils.data.DataLoader(
        dataset=feeder,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    for data in dataloader:
        pdb.set_trace()
