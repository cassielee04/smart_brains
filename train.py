import os
import torch
import torch.optim as optim
import torch.cuda.amp as amp
import torch.nn as nn
from tqdm import tqdm
#import albumentations as A
#from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from model import UNET, DiceLoss,MulticlassDiceLoss
from utils import (
    load_checkpoint, 
    save_checkpoint, 
    get_loader, 
    check_accuracy,
    save_prediction_as_imgs,)

# qrsh -pe omp 4 -P ds598 -l gpus=1
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

NUM_EPOCHS = 1
LEARN_RATE = 1e-7 #1e-4  reduced to 1e-5 for epoch 30-50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")#"cuda" #"cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 16
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = True
LOSS_TYPE = "MultiDice" #"CrossEntropy" # or "BCEWithLogitLoss", "Dice"
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 240
SPLIT_RATIO = 0.8
expName = "yuke"
expEpochNum = "random"
load_model_name = "yuke50-60"# "selena_adam30-50"

# TRAIN_IMG_DIR = "/projectnb/ds598/projects/smart_brains/dataset/MICCAI_FeTS2021_TrainingData/input/"
# TRAIN_SEG_DIR = "/projectnb/ds598/projects/smart_brains/dataset/MICCAI_FeTS2021_TrainingData/segmentation/"
# VAL_IMG_DIR = "/projectnb/ds598/projects/smart_brains/dataset/MICCAI_FeTS2021_ValidationData/input/"
# VAL_SEG_DIR = "/projectnb/ds598/projects/smart_brains/dataset/MICCAI_FeTS2021_ValidationData/segmentation/"
n_out_channels = 4


def train_function(loader, model, optimizer, loss_fn, scaler):
    
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop): 
        data = data.unsqueeze(1) 
        data = data.float().to(device=DEVICE)

        targets = targets.squeeze(1).long().to(device=DEVICE)  
        #print(f"Feature batch shape: {data.size()}")
        #print(f"Labels batch shape: {targets .size()}")

        with amp.autocast():  
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())  

# def train_function(train_loader, model, optimizer, loss_fn, scaler):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         optimizer.zero_grad()
#         # Assuming data has shape [batch_size, channels, height, width]
#         # If your data has a different shape, adjust accordingly
#         data = data[:, :1, :, :]  # Extracting only the first channel if it has 16 channels
#         data = data.to(device)
#         target = target.to(device)
#         with torch.cuda.amp.autocast():
#             output = model(data)
#             loss = loss_fn(output, target)
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()

def plot_learning_curve(score_allepoch, score_type, train_or_val):
    """
    score_allepoch -> list of scores = [...]
    score_type -> string
    train_or_val -> string
    """
    # Prepare the data for plotting
    #print(score_allepoch)
    epochs = range(1, len(score_allepoch) + 1)
    lines = list(zip(*score_allepoch))  # This transposes the list of lists

    # Plotting
    plt.figure(figsize=(10, 6))
    for i, line in enumerate(lines):
        plt.plot(epochs, line, label=f'Line {i+1}')

    plt.xlabel('Epoch Number')
    plt.ylabel(score_type.capitalize())
    plt.title(f'{score_type.capitalize()} by Epoch')
    plt.legend()
    plt.grid(True)
    plt.xticks(epochs)  # Ensure all epochs are shown
    file_path = f"{train_or_val}_{score_type}_{expName}.png"
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    global expName
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
    elif LOSS_TYPE == "MultiDice":
        #weights = torch.tensor([1,10,10,10]).to(DEVICE)
        #loss_fn = MulticlassDiceLoss(weight=weights)
        loss_fn = MulticlassDiceLoss()
    elif LOSS_TYPE == "Dice":
        # Using custom DiceLoss object
        loss_fn = DiceLoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)
 
    # Pass appropriate arguments into get_loader
    train_loader,val_loader = get_loader(BATCH_SIZE = BATCH_SIZE, 
                                         PIN_MEMORY = PIN_MEMORY,
                                         SPLIT_RATIO = SPLIT_RATIO)
    
    if LOAD_MODEL:
        load_checkpoint(torch.load(f"{load_model_name}_model_checkpoint.pth.tar"), model)
    expName = ''.join((expName, expEpochNum))
    print("expName",expName)
    
    scaler = amp.GradScaler()
    
    accuracy_all_val = []
    dice_all_val = []
    accuracy_all_train = []
    dice_all_train = []

    for epoch in range(NUM_EPOCHS):
        print("Epoch: ",epoch+1)
        #check_accuracy(val_loader, model, device=DEVICE,NUM_CLASSES = n_out_channels)
        train_function(train_loader, model, optimizer, loss_fn, scaler)  # Using dice_loss for training

    #     # Save model
        checkpoint = {
           "state_dict": model.state_dict(),
           "optimizer":optimizer.state_dict(),

        }
        save_checkpoint(checkpoint, expName)

        
        print("===Training===")
        accuracy_train, dice_train = check_accuracy(train_loader, model, device=DEVICE,NUM_CLASSES = n_out_channels)
        accuracy_all_train.append(accuracy_train)
        dice_all_train.append(dice_train)

        print("===Validation===")
        accuracy_val, dice_val = check_accuracy(val_loader, model, device=DEVICE,NUM_CLASSES = n_out_channels)
        accuracy_all_val.append(accuracy_val)
        dice_all_val.append(dice_val)

    #     # Print predictions ...   

        save_image_path = f"/projectnb/ds598/projects/smart_brains/saved_images_{expName}"
        os.makedirs(save_image_path,exist_ok=True)
        save_prediction_as_imgs(epoch, train_loader, model,folder=save_image_path, device=DEVICE)

   
        plot_learning_curve(accuracy_all_val, score_type="accuracy", train_or_val="validation")
        plot_learning_curve(accuracy_all_train, score_type="accuracy", train_or_val="training")



if __name__ == "__main__":
    main()






