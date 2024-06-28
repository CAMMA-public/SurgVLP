# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
import os
import random
import torch
import torch.utils.data
import numpy as np
import json
import pickle as pkl
import re
from PIL import Image
import math
import copy
import pandas as pd
from . import utils
from ..registry import DATASETS
from torch.utils.data import Dataset

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

@DATASETS.register_module(name='Recognition_frame')
class CholecDataset(Dataset):
    def __init__(self, csv_root, vid, video_root, transforms=None, loader=pil_loader):
        csv_name = os.path.join(csv_root, vid)
        df = pd.read_csv(csv_name)
        self.video_root = video_root

        self.file_list = df['path'].tolist()
        self.label_list = df['label'].tolist()
        assert len(self.file_list) == len(self.label_list)
        self.transform = transforms
        self.loader = loader


    def __getitem__(self, index):
        img_names = self.file_list[index]
        v_id, f_id = img_names.split('.png')[0].split('_')
        f_id = str(int(f_id) + 1)
        img_names = os.path.join(self.video_root, v_id+'_'+f_id+'.png')
        imgs = self.loader(img_names)

        labels_phase = self.label_list[index]

        # print(imgs.size)
        if self.transform is not None:
            imgs = self.transform(imgs)

        final_dict = {'video':imgs, 'label': labels_phase-1} # the label id starts from 1
        return final_dict

    def __len__(self):
        return len(self.file_list)