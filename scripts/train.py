import torch
import torch.optim as optim
import torch.cuda.amp as amp
import torch.nn as nn
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UNET, DiceLoss
from utils import get_loader
# #from utils import (
#     load_checkpoint, 
#     save_checkpoint, 
#     get_loader, 
#     check accuracy,
#     save_prediction_as_imgs,)


LEARN_RATE = [1e-4]
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = True
LOSS_TYPE = "CrossEntropy" # or "BCEWithLogitLoss", "Dice"
IMAGE_HEIGHT = 250
IMAGE_WIDTH = 250
TRAIN_IMG_DIR = "/projectnb/ds598/projects/smart_brains/dataset/MICCAI_FeTS2021_TrainingData/input/"
TRAIN_SEG_DIR = "/projectnb/ds598/projects/smart_brains/dataset/MICCAI_FeTS2021_TrainingData/segmentation/"
VAL_IMG_DIR = "/projectnb/ds598/projects/smart_brains/dataset/MICCAI_FeTS2021_ValidationData/input/"
VAL_SEG_DIR = "/projectnb/ds598/projects/smart_brains/dataset/MICCAI_FeTS2021_ValidationData/segmentation/"
n_out_channels = 4


def train_function(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):  
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)  

        with amp.autocast():  
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())  

def main():
    ''' define data augmentation
    train_transform = A.Compose(
        [
            # A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            # A.Rotate(limit=35, p=1.0)
            A.Normalize(
                mean=[0, 0, 0],
                std=[1, 1, 1],
                max_pixel_value=255
            ),
            ToTensorV2(),
        ],
    )

val_transform = A.Compose(
        [
            # A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            # A.Rotate(limit=35, p=1.0)
            A.Normalize(
                mean=[0, 0, 0],
                std=[1, 1, 1],
                max_pixel_value=255
            ),
            ToTensorV2(),
        ],
    )
    '''
    model = UNET(in_channels=1, out_channels=n_out_channels).to(DEVICE)

    # Create the loss function based on LOSS_TYPE
    if LOSS_TYPE == "CrossEntropy":
        loss_fn = nn.CrossEntropyLoss()
    elif LOSS_TYPE == "BCEWithLogitLoss":
        loss_fn = nn.BCEWithLogitsLoss()
    elif LOSS_TYPE == "Dice":
        # Using custom DiceLoss object
        loss_fn = DiceLoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
 
    # Pass appropriate arguments into get_loader
    train_loader,val_loader = get_loader(
        TRAIN_IMG_DIR,TRAIN_SEG_DIR,
    VAL_IMG_DIR, VAL_SEG_DIR,
        BATCH_SIZE,
        NUM_WORKERS,
        PIN_MEMORY,
        train_transform,
    val_transform
    )
    if LOAD_MODEL:
        load_checkpoint(torch.load("model_checkpoint.pth.tar"), model)
    scaler = amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_function(train_loader, model, optimizer, loss_fn, scaler)  # Using dice_loss for training

        # Save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),

        }
        save_checkpoint(checkpoint)

        check_accuracy(val_loader, model, device=DEVICE)
        # Print predictions ...
        save_prediction_as_imgs(val_loader,model,folder="saved_images/", device=DEVICE)



if __name__ == "__main__":
    main()






