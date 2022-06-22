import paddle.vision.transforms as T
import paddleseg.transforms as ST

import numpy as np
import random

from PIL import Image

from pathlib import Path

from paddle.io import Dataset


class SegmentDataset(Dataset):
    def __init__(self, base_path:str, file_list:str, img_size, mode='train',enable_aug=True,
        enable_cutmix=True,
        cutmix_threshold=0.8,
        cutmix_lambda=0.5,
        max_size=0x7fffffff):

        self.base_path= Path(str(base_path))
        self.img_size=img_size
        self.enable_cutmix=enable_cutmix

        self.cutmix_threshold=cutmix_threshold
        self.cutmix_lambda=cutmix_lambda

        if mode=='train':
            self.transforms=T.Compose([
                T.Resize(img_size),
                T.ColorJitter(),
            ])
            self.seg_transforms=ST.Compose([
                # ST.RandomRotation(),
                ST.RandomHorizontalFlip(),
                ST.RandomVerticalFlip(),
            ],to_rgb=False)
        else:
            self.transforms=T.Compose([
                T.Resize(img_size),
            ])
            self.seg_transforms=None

        if not enable_aug:
            self.seg_transforms=None

        with open(file_list,'r') as f:
            self.files=[[Path(p) for p in l.split()] for l in f.readlines()]

        if len(self.files)>max_size:
            self.files=random.sample(self.files,max_size)

    def do_getitem(self,index):
        image_path,label_path = self.files[index]

        lbl = Image.open(self.base_path/label_path)
        lbl=lbl.resize(self.img_size, Image.NEAREST)

        lbl = np.array(lbl).astype('int64')

        img = Image.open(self.base_path/image_path)

        if img.mode != 'RGB':
            img = img.convert('RGB') 

        img = self.transforms(img)

        img = np.array(img)
        
        if self.seg_transforms is not None:
            img,lbl=self.seg_transforms(img,lbl)
        
        img = img.astype('float32')

        if self.seg_transforms is not None:
            img = img / 255 # paddleseg automatically transposed it
        else:
            img = img.transpose((2, 0, 1)) / 255

        return img,lbl
    
    def __getitem__(self, index):
        img,lbl = self.do_getitem(index)
        return self.do_cutmix(img, lbl)

    def rand_bbox(self,size, lam):
        W,H=size
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def do_cutmix(self,img,lbl):
        if not self.enable_cutmix:
            return img,lbl
        
        if random.uniform(0,1)<self.cutmix_threshold:
            return img,lbl

        idx = random.randrange(1,len(self.files)-1)
        nimg,nlbl=self.do_getitem(idx)
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(self.img_size, self.cutmix_lambda)#随机产生一个box的四个坐标
        img[:, bbx1:bbx2, bby1:bby2] = nimg[:, bbx1:bbx2, bby1:bby2]
        lbl[bbx1:bbx2, bby1:bby2] = nlbl[bbx1:bbx2, bby1:bby2]

        return img,lbl

    def print_sample(self, index: int = 0):
        i,l=self.files[index]
        print("文件", i, "\t标签",l)

    def __len__(self):
        return len(self.files)