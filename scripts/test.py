import os
import logging
import torch
import torch.optim as optim
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as functional
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import UNET, DiceLoss, MulticlassDiceLoss
from dataset import BRATS21_Dataset
from torch.utils.data import DataLoader
from matplotlib.colors import ListedColormap
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import (
    load_checkpoint, 
    save_checkpoint, 
    get_loader, 
    check_accuracy,
    save_prediction_as_imgs,)

LEARN_RATE = 1e-8 
BATCH_SIZE = 16
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/projectnb/ds598/projects/smart_brains"
load_model_name = "expAugmented_t1ce-flair_opAdam_lr0.0001_bs16_epoch0_200"
MODAL_TYPE = load_model_name.split("_")[1].split('-')
print(f"MODAL_TYPE = {MODAL_TYPE}")
RESULT_PATH = f"/projectnb/ds598/projects/smart_brains/RESULTS/{load_model_name}"

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
model, optimizer, start_epoch, dice_log = load_checkpoint(f"{MODEL_PATH}/model_checkpoint/{load_model_name}_model_checkpoint.pth.tar", MODEL, OPTIMIZER)

# Load dataset
# testing_folder_path = "/projectnb/ds598/projects/smart_brains/dataset/MICCAI_FeTS2021_ValidationData"
testing_folder_path = "/projectnb/ds598/projects/smart_brains/dataset/MICCAI_FeTS2021_TestingData"
testing_folder_names = sorted([fname for fname in os.listdir(testing_folder_path) if "FeTS21" in fname])
print(f"There are {len(testing_folder_names)} Testing Data, from {testing_folder_names[0].split('_')[-1]} to {testing_folder_names[-1].split('_')[-1]}")
MODAL_TYPE = [modal.lower() for modal in MODAL_TYPE]

testing_image_paths = [[f"{testing_folder_path}/{testing_folder_name}/{testing_folder_name}_{modality}.nii" for modality in MODAL_TYPE] for testing_folder_name in testing_folder_names]
print(f"len(testing_image_paths) = {len(testing_image_paths)}")
testing_label_paths = [f"{testing_folder_path}/{testing_folder_name}/{testing_folder_name}_seg.nii" for testing_folder_name in testing_folder_names]

empty_transform = A.Compose([
        ToTensorV2()
    ])
test_dataset = BRATS21_Dataset(image_paths=testing_image_paths, label_paths=testing_label_paths, transform=empty_transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

test_features, test_labels = next(iter(test_loader))
test_features = test_features.squeeze(1)
print(f"Feature batch shape: {test_features.size()}")
print(f"Labels batch shape: {test_labels.size()}")

def dice_score(input_tensor, target_tensor):
    smooth = 1e-6
    intersection = torch.sum(input_tensor * target_tensor)
    union = torch.sum(input_tensor) + torch.sum(target_tensor)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice
    
def show_dice_score(predicted_tensor, targets_tensor, NUM_CLASSES = 4):
    OneHotPred = functional.one_hot(predicted_tensor,num_classes=NUM_CLASSES)
    OneHotTargets = functional.one_hot(targets_tensor,num_classes=NUM_CLASSES)
    OneHotPred = OneHotPred.permute(2, 0, 1)
    OneHotTargets = OneHotTargets.permute(2, 0, 1)
    
    return dice_score(OneHotPred[1:,:,:].cuda(), OneHotTargets[1:,:,:].cuda())

def torch2img_image_overlay(image_tensor, predicted_tensor, targets_tensor, id_in_batch, file_path):
    # Convert tensors to numpy arrays
    image = image_tensor[id_in_batch].squeeze().cpu().numpy()
    predicted = predicted_tensor[id_in_batch, :, :].cpu().numpy()
    targets = targets_tensor[id_in_batch, :, :].cpu().numpy()

    dice_score = show_dice_score(predicted_tensor[id_in_batch, :, :],targets_tensor[id_in_batch, :, :], NUM_CLASSES = 4)
    
    # Define a custom color map: you can adjust the RGBA values as needed
    colors = [(0, 0, 0, 0),  # Transparent
              (0.94, 0.5, 0.5, 1),  # Coral
              (0, 0.5, 0.5, 1),  # Teal
              (1, 0.84, 0, 1)]  # Gold
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Original Image
    ax[0].imshow(image, cmap='gray')  # Assuming the original image is grayscale
    ax[0].set_title(f'Targets on Original Image')
    ax[0].axis('off')  # Hide the axes
    ax[0].imshow(targets, cmap=cmap, alpha=0.7, interpolation='nearest', vmin=0, vmax=len(colors)-1)

    # Predicted Label
    ax[1].imshow(image, cmap='gray')  # Colormap for predicted labels
    ax[1].set_title('Predicted on Original Image')
    ax[1].axis('off')  # Hide the axes
    ax[1].imshow(predicted, cmap=cmap, alpha=0.7, interpolation='nearest', vmin=0, vmax=len(colors)-1)

    fig.suptitle(f'Dice Score = {dice_score:.4f}', fontsize=16)

    # Save the figure
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure to free memory

def torch2img_image_overlay_multimodal(image_tensor, predicted_tensor, targets_tensor, id_in_batch, file_path):
    # Convert tensors to numpy arrays
    image = image_tensor[id_in_batch].squeeze().cpu().numpy()
    image_t1ce = image[0,:,:]
    image_flair = image[1,:,:]
    predicted = predicted_tensor[id_in_batch, :, :].cpu().numpy()
    targets = targets_tensor[id_in_batch, :, :].cpu().numpy()

    dice_score = show_dice_score(predicted_tensor[id_in_batch, :, :], targets_tensor[id_in_batch, :, :], NUM_CLASSES = 4)

    colors = [(0, 0, 0, 0),  # Transparent
              (0.94, 0.5, 0.5, 1),  # Coral
              (0, 0.5, 0.5, 1),  # Teal
              (1, 0.84, 0, 1)]  # Gold
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    # Original Image
    ax[0,0].imshow(image_t1ce, cmap='gray')  # Assuming the original image is grayscale
    ax[0,0].set_title(f'Targets on Original T1CE Image')
    ax[0,0].axis('off')  # Hide the axes
    ax[0,0].imshow(targets, cmap=cmap, alpha=0.7, interpolation='nearest', vmin=0, vmax=len(colors)-1)

    # Predicted Label
    ax[0,1].imshow(image_t1ce, cmap='gray')  # Colormap for predicted labels
    ax[0,1].set_title('Predicted on Original T1CE Image')
    ax[0,1].axis('off')  # Hide the axes
    ax[0,1].imshow(predicted, cmap=cmap, alpha=0.7, interpolation='nearest', vmin=0, vmax=len(colors)-1)

    ax[1,0].imshow(image_flair, cmap='gray')  # Assuming the original image is grayscale
    ax[1,0].set_title(f'Targets on Original FLAIR Image')
    ax[1,0].axis('off')  # Hide the axes
    ax[1,0].imshow(targets, cmap=cmap, alpha=0.7, interpolation='nearest', vmin=0, vmax=len(colors)-1)

    # Predicted Label
    ax[1,1].imshow(image_flair, cmap='gray')  # Colormap for predicted labels
    ax[1,1].set_title('Predicted on Original FLAIR Image')
    ax[1,1].axis('off')  # Hide the axes
    ax[1,1].imshow(predicted, cmap=cmap, alpha=0.7, interpolation='nearest', vmin=0, vmax=len(colors)-1)
    

    fig.suptitle(f'Dice Score = {dice_score:.4f}', fontsize=16)

    # Save the figure
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure to free memory
    
def save_prediction_as_imgs(loader, model, folder, device="cuda"):
    model.eval()
    for idx, (data, targets) in enumerate(loader):
        
        data = data.float().to(device)
        targets = targets.long().to(device)
        with torch.no_grad():
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)

        for id_in_batch in range(data.size(0)):  # Loop through the batch
            if data.size(1) == 1:
                torch2img_image_overlay(data, predicted, targets, id_in_batch, file_path=f"{folder}/pred_{idx}_{id_in_batch}_epoch.png")
            else: 
                assert data.size(1) == 2
                torch2img_image_overlay_multimodal(data, predicted, targets, id_in_batch, file_path=f"{folder}/pred_{idx}_{id_in_batch}_epoch.png")
                    

prediction_img_path = f"{RESULT_PATH}/predicted_img_test"
os.makedirs(prediction_img_path,exist_ok=True)
# predictions = test_function(test_loader, model, DEVICE)
save_prediction_as_imgs(test_loader, model,folder=prediction_img_path, device=DEVICE) 

