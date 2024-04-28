import nibabel as nib
from torch.utils.data import Dataset
import numpy as np

def load_nii_slice(nii_file_path, slice_idx=78):
    nii_image = nib.load(nii_file_path)
    image_data = nii_image.get_fdata()
    return image_data[:, :, slice_idx]

# def one_hot_encoder(matrix, values_map = {0: 0, 1: 1, 2: 2, 4: 3}):
#     one_hot_encoded_matrix = np.zeros((matrix.shape[0], matrix.shape[1], 4))
#     # Populate the one hot encoded matrix
#     for key, value in values_map.items():
#         one_hot_encoded_matrix[:, :, value] = (matrix == key).astype(int)
#     return one_hot_encoded_matrix

class BRATS21_Dataset(Dataset):
    def __init__(self, image_paths, label_paths, transform=None, test=False):
        self.images = image_paths
        self.labels = label_paths
        self.test = test
        self.transform = transform
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        modal_imgs = [load_nii_slice(modal_path, slice_idx=78) for modal_path in self.images[index]]
        img = np.stack(modal_imgs, axis=-1)  # Stack along the first axis
        
        # Normalize the concatenated image
        img = img.astype(np.float32)
        
        
        #img = load_nii_slice(self.images[index], slice_idx=78)
        
        # Normalize the image
        img = (img - img.min()) / (img.max() - img.min())
        
        if not self.test:
            label = load_nii_slice(self.labels[index], slice_idx=78)
            # Apply a map for multiclass or binary classification
            multiclass_fix_map = {0: 0, 1: 1, 2: 2, 4: 3}
            #binaryclass_map = {0: 0, 1: 1, 2: 1, 4: 1}
            vectorized_map = np.vectorize(multiclass_fix_map.get)
            label = vectorized_map(label)
            # Check the dimensions before transformation
            if self.transform:
                augmented = self.transform(image=img, mask=label)
                img, label = augmented['image'], augmented['mask']
            return img, label
        else:
            if self.transform:
                augmented = self.transform(image=img)
                img = augmented['image']
            return img


