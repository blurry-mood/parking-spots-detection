from glob import glob
import pandas as pd

import os
from os.path import join, split, abspath

from tqdm.auto import tqdm

images = glob(join('dataset','FULL_IMAGE_1000x750', '**', '*.jpg'), recursive=True)

data = []
for img in tqdm(images, desc='Making CSV'):
    split = img.split(os.sep)
    data.append({'path': img,
                 'camera': split[-2].split('camera')[-1], 
                 'date':split[-3], 
                 'time':split[-1].split('_')[-1].split('.')[0], 
                 'weather': split[-4]})
    
pd.DataFrame(data).to_csv(join('dataset', 'data.csv'), index=False)

cameras = glob(join('dataset', 'camera*.csv'))
assert len(cameras)==9
print('Scaling bounding boxes to fit image dimensions...')
for cam in cameras:
    df = pd.read_csv(cam)
    df.X = df.X*1000/2592
    df.W = df.W*1000/2592
    
    df.Y = df.Y*750/1944
    df.H = df.H*750/1944 
    
    df.to_csv(cam, index=False)