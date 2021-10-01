import os
from subprocess import run
from PIL import Image
from tqdm.auto import tqdm

import numpy as np

_HERE = os.path.split(os.path.abspath(__file__))[0]
import sys
sys.path.append(os.path.join(_HERE, '..'))
from datamodule import CNRDataModule


dm = CNRDataModule(os.path.join(_HERE, '..', '..', 'data'),'yolo.yaml')
dm.prepare_data()
dm.setup()

dataloaders = dict(train=dm.train_dataloader(), val=dm.val_dataloader(), test=dm.test_dataloader())

for split in dataloaders:
    images_path = os.path.join(_HERE, '..', '..', 'yolo_dataset', split, 'images')
    labels_path = os.path.join(_HERE, '..', '..', 'yolo_dataset', split, 'labels')
    
    run(['mkdir', '-p', images_path])
    run(['mkdir', '-p', labels_path])
    
    for i, (images, boxes) in tqdm(enumerate(dataloaders[split]), desc=f'Preparing {split}'):
        for j, (image, box) in enumerate(zip(images.numpy(), boxes.numpy())):
            name = f'img_{i}_{j}'
            image = Image.fromarray(image.astype('uint8').transpose(1, 2, 0))
            image.save(os.path.join(images_path, f'{name}.jpg'))
            
            # remove padded part
            for i in range(box.shape[0]):
                if (box[i] == -np.ones(4)).all():
                    break
            if i==0:
                continue
            box = box[:i]
            box = box.astype('float32')
            
            # normalize box coordinates
            box[:, [0, 2]] = box[:, [0, 2]]/1000
            box[:, [1, 3]] = box[:, [1, 3]]/750
            
            # format to (xc, yc, w, h)
            box[:, 0] = box[:, 0] + box[:, 2]/2
            box[:, 1] = box[:, 1] + box[:, 3]/2          
            
            # prepend box with classes columns, with zeros
            bbox = np.hstack((np.zeros((box.shape[0], 1)), box))

            # save bbox in {name}.txt
            np.savetxt(os.path.join(labels_path, f'{name}.txt'), bbox, delimiter=' ', fmt='%.8f')