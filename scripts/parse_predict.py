import os
import logging
import torch
import torch.optim as optim
import torch.cuda.amp as amp
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import UNET, DiceLoss,MulticlassDiceLoss
from dataset import BRATS21_Dataset
from torch.utils.data import DataLoader
from matplotlib.colors import ListedColormap
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
from utils import (
    load_checkpoint, 
    save_checkpoint, 
    get_loader, 
    check_accuracy,
    save_prediction_as_imgs,)

### Put Data folder in the following format to MICCAI_FeTS2021_TestingData folder
### FeTS21_Training_{patient_id1} 
### - FeTS21_Training_{patient_id1}_flair.nii
### - FeTS21_Training_{patient_id1}_t1ce.nii
### FeTS21_Training_{patient_id2} 
### - FeTS21_Training_{patient_id2}_flair.nii
### - FeTS21_Training_{patient_id2}_t1ce.nii
### ...

# export MODEL_PATH=./default_model_path
# python parse_predict.py --load_model_name exp_flair_opAdam_lr0.0001_bs16_epoch0_200
# python parse_predict.py --load_model_name exp_t1ce-flair_opAdam_lr0.0001_bs16_epoch0_200

parser = argparse.ArgumentParser(description="Testing script for U-Net model.")
parser.add_argument('--load_model_name', type=str, default='', help='Model name to load if LOAD_MODEL is True.')

# Parse the arguments
args = parser.parse_args()
load_model_name = args.load_model_name

LEARN_RATE = 1e-8 
BATCH_SIZE = 16
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAIN_PATH = os.getenv('MODEL_PATH', './default_model_path')
# load_model_name = "exp_flair_opAdam_lr0.0001_bs16_epoch0_200"
MODAL_TYPE = load_model_name.split("_")[1].split('-')
print(f"MODAL_TYPE = {MODAL_TYPE}")
RESULT_PATH = f"{MAIN_PATH}/RESULTS/{load_model_name}"




# model initialization
n_out_channels = 4
optimizer_name = "Adam"
MODEL = UNET(in_channels=len(MODAL_TYPE), out_channels=n_out_channels).to(DEVICE)
# specify optimizer
if optimizer_name == "Adam":
    OPTIMIZER = optim.Adam(MODEL.parameters(), lr=LEARN_RATE)
elif optimizer_name == "SGD":
    OPTIMIZER = optim.SGD(MODEL.parameters(), lr=LEARN_RATE)
else:
    print("Please use Adam or SGD!")

# Load model
model, optimizer, start_epoch, dice_log = load_checkpoint(f"{MAIN_PATH}/model_checkpoint/{load_model_name}_model_checkpoint.pth.tar", MODEL, OPTIMIZER)

# Load dataset
testing_folder_path = f"{MAIN_PATH}/dataset/MICCAI_FeTS2021_TestingData"
testing_folder_names = sorted([fname for fname in os.listdir(testing_folder_path) if "FeTS21" in fname])
print(f"There are {len(testing_folder_names)} Testing Data, from {testing_folder_names[0].split('_')[-1]} to {testing_folder_names[-1].split('_')[-1]}")
MODAL_TYPE = [modal.lower() for modal in MODAL_TYPE]

testing_image_paths = [[f"{testing_folder_path}/{testing_folder_name}/{testing_folder_name}_{modality}.nii" for modality in MODAL_TYPE] for testing_folder_name in testing_folder_names]
print(f"len(testing_image_paths) = {len(testing_image_paths)}")
# print(f"testing_image_paths = {testing_image_paths}")

empty_transform = A.Compose([
        ToTensorV2()
    ])
test_dataset = BRATS21_Dataset(image_paths=testing_image_paths, 
                               label_paths=[], 
                               transform=empty_transform, 
                               test=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)


def torch2img_image_overlay_multimodal(image_tensor, predicted_tensor, id_in_batch, file_path):
    # Convert tensors to numpy arrays
    image = image_tensor[id_in_batch].squeeze().cpu().numpy()
    image_t1ce = image[0,:,:]
    image_flair = image[1,:,:]
    predicted = predicted_tensor[id_in_batch, :, :].cpu().numpy()

    colors = [(0, 0, 0, 0),  # Transparent
              (0.94, 0.5, 0.5, 1),  # Coral
              (0, 0.5, 0.5, 1),  # Teal
              (1, 0.84, 0, 1)]  # Gold
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(2, 1, figsize=(5, 10))

    # Original T1CE Image
    ax[0].imshow(image_t1ce, cmap='gray')
    ax[0].set_title('Predicted on Original T1CE Image')
    ax[0].axis('off')
    ax[0].imshow(predicted, cmap=cmap, alpha=0.7, interpolation='nearest', vmin=0, vmax=len(colors)-1)

    # Original FLAIR Image
    ax[1].imshow(image_flair, cmap='gray')
    ax[1].set_title('Predicted on Original FLAIR Image')
    ax[1].axis('off')
    ax[1].imshow(predicted, cmap=cmap, alpha=0.7, interpolation='nearest', vmin=0, vmax=len(colors)-1)

    # Save the figure
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure to free memory
    
def torch2img_image_overlay(image_tensor, predicted_tensor, id_in_batch, file_path):
    # Convert tensors to numpy arrays
    image = image_tensor[id_in_batch].squeeze().cpu().numpy()
    predicted = predicted_tensor[id_in_batch, :, :].cpu().numpy()
    
    colors = [(0, 0, 0, 0),  # Transparent
              (0.94, 0.5, 0.5, 1),  # Coral
              (0, 0.5, 0.5, 1),  # Teal
              (1, 0.84, 0, 1)]  # Gold
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.imshow(image, cmap='gray')
    ax.set_title('Predicted on Original Image')
    ax.axis('off')
    ax.imshow(predicted, cmap=cmap, alpha=0.7, interpolation='nearest', vmin=0, vmax=len(colors)-1)

    # Save the figure
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure to free memory



def save_prediction_as_imgs(loader, model, folder, device="cuda"):
    model.eval()
    for idx, data in enumerate(loader):  # Assuming you don't need targets for plotting
        
        data = data.float().to(device)
        
        with torch.no_grad():
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability

        # Assuming the first dimension of data is the batch dimension
        for id_in_batch in range(data.size(0)):  # Loop through the batch
            if data.size(1) == 1:
                torch2img_image_overlay(data, predicted, id_in_batch, file_path=f"{folder}/pred_{idx}_{id_in_batch}_epoch.png")
            else: 
                assert data.size(1) == 2
                torch2img_image_overlay_multimodal(data, predicted, id_in_batch, file_path=f"{folder}/pred_{idx}_{id_in_batch}_epoch.png")


prediction_img_path = f"{RESULT_PATH}/predicted_img_predict"
os.makedirs(prediction_img_path,exist_ok=True)
save_prediction_as_imgs(test_loader, model,folder=prediction_img_path, device=DEVICE) 

