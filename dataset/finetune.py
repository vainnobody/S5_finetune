import os
import math
import random
import numpy as np
from PIL import Image
from copy import deepcopy
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
from dataset.transform import *
from torchvision import transforms as T

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class SemiDataset(Dataset):
    def __init__(self, name, root, mode, transform=None, size=None, ignore_value=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.ignore_value = ignore_value
        self.reduce_zero_label = True if name == 'ade20k' else False
        self.transform = transform


        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open('splits/%s/val.txt' % name, 'r') as f:
            #     self.ids = f.read().splitlines()
            # with open('splits/%s/test.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()


    def _trainid_to_class(self, label):

        return label

    def tes_class_to_trainid(self, label):
   
        return label

    def _class_to_trainid(self, label):

        return label

    def process_mask(self, mask):
        mask = np.array(mask) - 1
        return Image.fromarray(mask)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
        mask = Image.open(os.path.join(self.root, id.split(' ')[1]))

        if self.name == 'loveda': 
            mask = self.process_mask(mask)

        img, mask = resize(img, mask, (0.5, 2.0))
        ignore_value = 254 if self.mode == 'train_u' else self.ignore_value
        img, mask = crop(img, mask, self.size, ignore_value)
        if self.name == 'OpenEarthMap':
            return normalize(img, mask)
        
        img, mask = hflip(img, mask, p=0.5)
        img, mask = bflip(img, mask, p=0.5)
        img, mask = Rotate_90(img, mask, p=0.5)
        # img, mask = Rotate_180(img, mask, p=0.5)
        # img, mask = Rotate_270(img, mask, p=0.5)
        if random.random() < 0.8:
            try:
                # 检查图像数据是否正常
                img_array = np.array(img)
                if img_array.min() < 0 or img_array.max() > 255:
                    print(f"[WARNING] Image pixel values out of range: min={img_array.min()}, max={img_array.max()}")
                    print(f"  Image ID: {id}")
                img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            except OverflowError as e:
                print(f"[WARNING] ColorJitter OverflowError on image: {id}")
                print(f"  Image mode: {img.mode}, size: {img.size}")
                img_array = np.array(img)
                print(f"  Pixel range: min={img_array.min()}, max={img_array.max()}")
                print(f"  Error: {e}")
                # 跳过 ColorJitter，保留原图
                pass
        try:
            img = transforms.RandomGrayscale(p=0.2)(img)
        except OverflowError as e:
            print(f"[WARNING] RandomGrayscale OverflowError on image: {id}")
            print(f"  Error: {e}")
        img = blur(img, p=0.5)

        return normalize(img, mask)



class ValDataset(Dataset):
    def __init__(self, name, root, mode, size=None,  ignore_value=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.ignore_value = ignore_value
        self.reduce_zero_label = True if name == 'ade20k' else False

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            if id_path is not None:
                with open(id_path, 'r') as f:
                    self.ids = f.read().splitlines()
            else:
                with open('splits/%s/val.txt' % name, 'r') as f:
                    self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
        mask =Image.open(os.path.join(self.root, id.split(' ')[1]))
    
        if self.name == 'loveda': 
            mask = self.process_mask(mask)

        if self.mode == 'val':
            ori_img = img
            img, mask = normalize(img, mask)
            # return img, mask, id
            return img, mask

    def process_mask(self, mask):
        mask = np.array(mask) - 1
        return Image.fromarray(mask)

    def __len__(self):
        return len(self.ids)
