import os
import pandas as pd
import numpy as np
from torch.utils import data
from PIL import Image
import torch
import torchvision.transforms as t
import sys
ignore_label=255
num_classes = 3
root = '/home/nclab/dabeeo_data/satellite/'
trans = t.ToTensor()

def make_dataset(mode):
    assert mode in ['train', 'val']
    switch = {'train':'list_train.csv', 'val':'list_test.csv'}
    train_path = root + switch[mode]
    map_frame = pd.read_csv(train_path, header=-1)
    items = []
    for i in zip(map_frame[0], map_frame[1]):
        if 'annotations' in i[0]:
            z = (i[1], i[0])
            items.append(z)
            continue
        items.append(i)
    return items

class DataS(data.Dataset):
    def __init__(self, mode="train"):
        self.img_name = make_dataset(mode)
        if len(self.img_name) == 0 :
            raise RuntimeError('Found 0 images!')
        self.mode = mode
    def __len__(self):
        return len(self.img_name)
    def __getitem__(self,index):
        img_path, mask_path = self.img_name[index]
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)
        mask = np.array(mask, dtype=np.float_)
        mask = torch.from_numpy(mask).long()
        mask = (mask == 2)
        mask = mask.type(torch.FloatTensor)
        mask = mask.unsqueeze(0)
        img = trans(img)
        return img, mask