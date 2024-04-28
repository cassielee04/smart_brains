import os
import matplotlib.pyplot as plt
import numpy as np
from utils import get_loader  # Assuming this is a custom function you've defined
import seaborn as sns
from tqdm import tqdm
import torch

PIN_MEMORY = True
BATCH_SIZE = 341
SPLIT_RATIO = 0.8
MODAL_TYPE = "FLAIR"
DATA_SET = "Validation"
save_path = "/projectnb/ds598/projects/smart_brains"

def collect_masked_pixels_by_value(loader, mask_values=[1, 2, 3], exclude_zeros=True):
    pixels_by_mask = {value: [] for value in mask_values}
    for img, target in loader:
        if DATA_SET == "Validation":
            img = img.squeeze(1)
        print(f'Image pixel value range: {torch.min(img).item()} to {torch.max(img).item()}')

        # Assuming img and target are numpy arrays or compatible formats

        for value in mask_values:
            mask = (target == value)
            masked_pixels = img[mask]
            # if exclude_zeros:
            #     masked_pixels = masked_pixels[masked_pixels != 0]
            pixels_by_mask[value].extend(masked_pixels.flatten().tolist())  # Ensuring it's a list
    return pixels_by_mask

train_loader,val_loader = get_loader(MODAL_TYPE = [MODAL_TYPE],
                                         BATCH_SIZE = BATCH_SIZE, 
                                         PIN_MEMORY = PIN_MEMORY,
                                         SPLIT_RATIO = SPLIT_RATIO)

if DATA_SET == "Training":
    pixels_by_mask = collect_masked_pixels_by_value(train_loader, mask_values=[1, 2, 3])
elif DATA_SET == "Validation":
    pixels_by_mask = collect_masked_pixels_by_value(val_loader, mask_values=[1, 2, 3])
else: 
    print("Please specify Training or Validation")

plt.figure(figsize=(12, 8))

colors = ['skyblue', 'salmon', 'limegreen']#, 'goldenrod']
mask_labels = ['Class 1', 'Class 2', 'Class 3']#,'Mask4']

for (mask_value, pixels), color, label in zip(pixels_by_mask.items(), colors, mask_labels):
    # Check for NaNs, infs, and ensure there's more than one element
    pixels = np.array(pixels)  # Convert list to np.array for processing
    pixels = pixels[np.isfinite(pixels)]  # Remove NaNs and infs
    if len(pixels) > 1:  # Ensure there are multiple elements
        sns.histplot(pixels, bins=100, kde=True, color=color, stat='density', label=label, alpha=0.5, line_kws={'linewidth': 2})
    else:
        print(f"Skipping {label}: Not enough data points after cleaning.")

plt.title(f'Histogram and KDE of Normalized Pixel Values for Different Mask Labels ({MODAL_TYPE}-{DATA_SET})')
plt.xlabel('Normalized Pixel Values')
plt.ylabel('Density')
plt.legend()
plt.xlim(0, 1)
plt.savefig(f"{save_path}/{MODAL_TYPE}_{DATA_SET}_distribution_new.png")
plt.show()
