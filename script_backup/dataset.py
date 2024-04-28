import nibabel as nib
from torch.utils.data import Dataset
import numpy as np

def load_nii_slice(nii_file_path, slice = 78):
    nii_image = nib.load(nii_file_path)
    image_data = nii_image.get_fdata()
    return image_data[:, :, slice]

def one_hot_encoder(matrix, values_map = {0: 0, 1: 1, 2: 2, 4: 3}):
    one_hot_encoded_matrix = np.zeros((matrix.shape[0], matrix.shape[1], 4))
    # Populate the one hot encoded matrix
    for key, value in values_map.items():
        one_hot_encoded_matrix[:, :, value] = (matrix == key).astype(int)
    return one_hot_encoded_matrix

class BRATS21_Dataset(Dataset):
    def __init__(self, image_paths, label_paths):
        self.images, self.labels = image_paths, label_paths
    def __len__(self):
        return len(self.images)
    def __getitem__(self,index):
        img = load_nii_slice(self.images[index], slice = 78)
        label = load_nii_slice(self.labels[index], slice = 78) # (240, 240)

        # Apply Min-Max Normalization to [0,1]
        img = (img - img.min()) / (img.max() - img.min())
        
        #one_hot_encoded_label = one_hot_encoder(label) # (240, 240, 4)
        multiclass_fix_map = {0: 0, 1: 1, 2: 2, 4: 3}
        binaryclass_map = {0: 0, 1: 1, 2: 1, 4: 1}
        vectorized_map = np.vectorize(multiclass_fix_map.get)
        label = vectorized_map(label)
        return img, label

