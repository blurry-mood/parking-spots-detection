import torch
from torch.utils.data import Dataset, DataLoader, random_split

import albumentations as A
import albumentations.pytorch as P

from pytorch_lightning import LightningDataModule
from omegaconf import OmegaConf

import pandas as pd
import numpy as np
import PIL

import os
from glob import glob

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

__all__ = ['CNRDataModule']
_HERE = os.path.split(os.path.abspath(__file__))[0]


class CNRDataset(Dataset):
    
    def __init__(self, root, csv, cameras, transform=None, max_boxes=1000):
        super().__init__()
        
        self.imgs = csv['path'].apply(lambda x: os.path.join(root, x)).tolist()
        self.camera_ids =  csv['camera'].tolist()
        self.cameras = cameras        
        self.transform = transform
        self.max_boxes = max_boxes
        
    def __len__(self,):
        return len(self.imgs)
    
    def __getitem__(self, i):
        image = self.imgs[i]
        cam = self.camera_ids[i]
        
        image = np.array(PIL.Image.open(image))
        bboxes = self.cameras[cam].to_numpy()
        
        n_pad = self.max_boxes - bboxes.shape[0]
        assert n_pad>=0
        
        if self.transform:
            transformed = self.transform(image=image, bboxes=bboxes )
            image, bboxes = transformed['image'], transformed['bboxes']
        else:
            image = image.astype('float32').transpose(2, 0, 1)
            bboxes = bboxes.astype('int64')
            bboxes = np.pad(bboxes, [(0, n_pad), (0, 0)], mode='constant', constant_values=-1)
            
        return image, bboxes     
            
        
class CNRDataModule(LightningDataModule):
    
    def __init__(self, dataset_path, cfg_name):
        super().__init__()
        
        self.dataset_path = dataset_path
        self.cfg = OmegaConf.load(glob(os.path.join(_HERE, '**', cfg_name), recursive=True)[0])        
        
        # Init transforms
        def get_transform(cfg):
            transforms = []
            try:
                for op, kwargs in cfg.items():
                    try:
                        transforms.append(getattr(A, op)(**dict(kwargs)))
                    except:
                        transforms.append(getattr(P, op)(**dict(kwargs)))

                return A.Compose(transforms)
            except:
                return None
        self.transforms = dict(
                            train=get_transform(self.cfg.train_transforms),
                            val=get_transform(self.cfg.val_transforms),
                            test=get_transform(self.cfg.test_transforms)
                              )
        
    def prepare_data(self, ):
        self.csv = pd.read_csv(os.path.join(self.dataset_path, 'dataset', 'data.csv'))
        self.cameras = {i:pd.read_csv(os.path.join(self.dataset_path, 'dataset', f'camera{i}.csv')).drop('SlotId', axis=1) for i in range(1, 10)}
        self.data = CNRDataset(self.dataset_path, self.csv, self.cameras)
    
    def setup(self, stage=None):
        if stage is None or stage == 'fit':
            n = self.csv.shape[0]
            test, val = int(n*self.cfg.split.test), int(n*self.cfg.split.val)
            train = n - (test + val)
            self.train, self.val, self.test = random_split(self.data, [train, val, test])
            self.train.transform = self.transforms['train']
            self.val.transform = self.transforms['val']
            self.test.transform = self.transforms['test']
            
    
    def train_dataloader(self,):
        return DataLoader(self.train, **dict(self.cfg.train_loader) )
    
    def val_dataloader(self,):
        return DataLoader(self.val, **dict(self.cfg.val_loader) )
    
    def test_dataloader(self,):
        return DataLoader(self.test, **dict(self.cfg.test_loader) )
    

if __name__=='__main__':
    dataset = os.path.join(_HERE, '..', '..', 'data')
    dm = CNRDataModule(dataset, 'config.yaml')
    dm.prepare_data()
    dm.setup()
    dm.train_dataloader()
    dm.val_dataloader()
    dm.test_dataloader()
    
    img, boxes = dm.train[0]
    print(img.dtype, img.shape, boxes.dtype, boxes.shape)
    dm.val[0]
    dm.test[0]
    print('======> Success!')