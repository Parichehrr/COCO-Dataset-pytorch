'''
From: https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
'''

import torch.utils.data as data
import os
from PIL import Image

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def default_loader(path):
    return pil_loader(path)

class cocodataset(data.Dataset):
    def __init__(self, root, path, transform=None):
        self.root = root

        self.loader = default_loader

        if os.path.isfile(path):
            with open(path) as listfile:
                lines = listfile.readlines()

        self.imgs = []
        self.targets = []
        
        for l in lines:
            img_path, target = l.rstrip().split(" ")
            self.targets.append(int(target))
            self.imgs.append((img_path, int(target)))

        self.transform = transform
   
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(os.path.join(self.root, path))

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target




    def __len__(self):
        return len(self.imgs)

    def num_labels(self):
        return 3
