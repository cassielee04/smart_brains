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
MODAL_TYPE = "FLAIR"
load_model_name = "exp_flair_opAdam_lr0.0001_bs16_epoch0_200"
RESULT_PATH = f"/projectnb/ds598/projects/smart_brains/RESULTS/{load_model_name}"

# model initialization
n_out_channels = 4
optimizer_name = "Adam"
MODEL = UNET(in_channels=1, out_channels=n_out_channels).to(DEVICE)
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
testing_folder_path = "/projectnb/ds598/projects/smart_brains/dataset/MICCAI_FeTS2021_ValidationData"
# testing_folder_path = "/projectnb/ds598/projects/smart_brains/dataset/MICCAI_FeTS2021_TestingData"
# these are unlabeled files
testing_folder_names = sorted([fname for fname in os.listdir(testing_folder_path) if "FeTS21" in fname])
print(f"There are {len(testing_folder_names)} Testing Data, from {testing_folder_names[0].split('_')[-1]} to {testing_folder_names[-1].split('_')[-1]}")
MODAL_TYPE = MODAL_TYPE.lower()
testing_image_paths = [f"{testing_folder_path}/{testing_folder_name}/{testing_folder_name}_{MODAL_TYPE}.nii" for testing_folder_name in testing_folder_names]
test_dataset = BRATS21_Dataset(image_paths=testing_image_paths, label_paths=[], test=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)


def torch2img(image_tensor, predicted_tensor, id_in_batch, file_path):
    # Convert the tensors to numpy arrays for plotting
    # Ensure to squeeze or select the channel dimension if necessary
    image = image_tensor[id_in_batch].squeeze().cpu().numpy()  # This removes the channel dim if it's 1
    predicted = predicted_tensor[id_in_batch].cpu().numpy()

    # Adjusting predicted tensor if it has an extra channel dimension
    if predicted.shape[0] == 1:  # If predicted also has a channel dimension
        predicted = predicted.squeeze(0)  # This assumes the channel is the first dimension

    colors = [(1, 1, 1, 1),  # White
              (0.94, 0.5, 0.5, 1),  # Coral
              (0, 0.5, 0.5, 1),  # Teal
              (1, 0.84, 0, 1)]  # Gold
    cmap = ListedColormap(colors)
    # Plotting both arrays side by side
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Original Image
    ax[0].imshow(image, cmap='gray')  # Assuming the original image is grayscale
    ax[0].set_title('Original Image')
    ax[0].axis('off')  # Hide the axes

    # Predicted Label
    ax[1].imshow(predicted, cmap=cmap, interpolation='nearest')  # Colormap for predicted labels
    ax[1].set_title('Predicted Label')
    ax[1].axis('off')  # Hide the axes

    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure to free memory


def save_prediction_as_imgs(loader, model, folder, device="cuda"):
    model.eval()
    for idx, data in enumerate(loader):  # Assuming you don't need targets for plotting
        
        data = data.float().unsqueeze(1).to(device)
        
        with torch.no_grad():
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability

        # Assuming the first dimension of data is the batch dimension
        for id_in_batch in range(data.size(0)):  # Loop through the batch
            # Generate a path for each image-prediction pair
            file_path = f"{folder}/pred_{idx}_{id_in_batch}_test.png"
            torch2img(data, predicted, id_in_batch, file_path)


def test_function(loader, model, device):
    model.eval()  # Set the model to evaluation mode
    predictions = []  # List to store predictions
    with torch.no_grad():  # No need to track gradients
        for batch_idx, data in enumerate(tqdm(loader)):
            data = data.unsqueeze(1)  # Adjust based on your model's input format
            data = data.float().to(device=device)

            output = model(data)  # Get the model's predictions

            # Optionally process your output here (e.g., apply a threshold or take argmax)
            predictions.append(output.cpu())  # Move predictions to CPU and store them

    return predictions

prediction_img_path = f"{RESULT_PATH}/predicted_img_test"
os.makedirs(prediction_img_path,exist_ok=True)
predictions = test_function(test_loader, model, DEVICE)
save_prediction_as_imgs(test_loader, model,folder=prediction_img_path, device=DEVICE) 

