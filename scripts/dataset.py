import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class brainDataset(Dataset):
    # transform  needs to be changes
    def __init__(self,image_dir,seg_dir, transform = None):
        self.image_dir = image_dir
        self.seg_dir = seg_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,index):
        img_path = os.path.join(self.image_dir,self.image[index])
        seg_path = os.path.join(self.seg_dir,self.image[index])
        image = np.array(Image.open(img_path))
        #++++++++++++++++++++++++++ 
        segment = np.array(Image.open(seg_path).convert("L"),dtype=np.float32)
        segment[segment == 255.0] = 1.0
        #++++++++++++++++++++++++++ 
        if self.transform is not None:
            augmentation = self.transform(image=image, segment=segment)
            image = augmentations["image"]
            segment = augmentations["segment"]
        
        return image, segment
        
