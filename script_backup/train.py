import os
import logging
import torch
import torch.optim as optim
import torch.cuda.amp as amp
import torch.nn as nn
from tqdm import tqdm
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

NUM_EPOCHS = 30
LEARN_RATE = 1e-5 #1e-4  reduced to 1e-5 for epoch 30-50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")#"cuda" #"cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 16
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOSS_TYPE = "MultiDice" #"CrossEntropy" # or "BCEWithLogitLoss", "Dice"
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 240
SPLIT_RATIO = 0.8
MODEL_PATH = "/projectnb/ds598/projects/smart_brains"
EXP_NAME = "SelenaNormT1ce"
EXP_EPOCH_RANGE = ""
load_model_name = ""# "selena_adam30-50"


EXP_NAME = ''.join((EXP_NAME, EXP_EPOCH_RANGE))
logging.basicConfig(level=logging.INFO, filename=f'{EXP_NAME}{EXP_EPOCH_RANGE}_training.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')

# TRAIN_IMG_DIR = "/projectnb/ds598/projects/smart_brains/dataset/MICCAI_FeTS2021_TrainingData/input/"
# TRAIN_SEG_DIR = "/projectnb/ds598/projects/smart_brains/dataset/MICCAI_FeTS2021_TrainingData/segmentation/"
# VAL_IMG_DIR = "/projectnb/ds598/projects/smart_brains/dataset/MICCAI_FeTS2021_ValidationData/input/"
# VAL_SEG_DIR = "/projectnb/ds598/projects/smart_brains/dataset/MICCAI_FeTS2021_ValidationData/segmentation/"
n_out_channels = 4


def train_function(loader, model, optimizer, loss_fn, scaler):
    model.train()
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop): 
        data = data.unsqueeze(1) 
        data = data.float().to(device=DEVICE)

        targets = targets.squeeze(1).long().to(device=DEVICE)  


        with amp.autocast():  
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())  




def plot_learning_curve(score_allepoch, score_type, train_or_val,save_path):
    """
    score_allepoch -> list of scores = [...]
    score_type -> string
    train_or_val -> string
    """
    # Prepare the data for plotting
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
    file_path = f"{save_path}/{train_or_val}_{score_type}_{EXP_NAME}.png"
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close()



def main():
    model = UNET(in_channels=1, out_channels=n_out_channels).to(DEVICE)

    # Create the loss function based on LOSS_TYPE
    if LOSS_TYPE == "CrossEntropy":
        loss_fn = nn.CrossEntropyLoss()
    elif LOSS_TYPE == "MultiDice":
        loss_fn = MulticlassDiceLoss()
    else:
        logging.error('Need to define loss function.')

    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)
 
    # Pass appropriate arguments into get_loader
    train_loader,val_loader = get_loader(BATCH_SIZE = BATCH_SIZE, 
                                         PIN_MEMORY = PIN_MEMORY,
                                         SPLIT_RATIO = SPLIT_RATIO)
    
    if LOAD_MODEL:
        load_checkpoint(torch.load(f"{MODEL_PATH}/model_checkpoint/{load_model_name}_model_checkpoint.pth.tar"), model)
    
    scaler = amp.GradScaler()
    
    accuracy_all_val = []
    dice_all_val = []
    accuracy_all_train = []
    dice_all_train = []
    dice_log = 0 

    for epoch in range(NUM_EPOCHS):
        print("Epoch:", epoch + 1)
        logging.info("Epoch: %d", epoch + 1)
        train_function(train_loader, model, optimizer, loss_fn, scaler)  # Using dice_loss for training

        
        logging.info("===Training===")
        accuracy_train, dice_train = check_accuracy(train_loader, model, device=DEVICE,NUM_CLASSES = n_out_channels)
        accuracy_all_train.append(accuracy_train)
        dice_all_train.append(dice_train)

        logging.info("===Validation===")
        accuracy_val, dice_val = check_accuracy(val_loader, model, device=DEVICE,NUM_CLASSES = n_out_channels)
        accuracy_all_val.append(accuracy_val)
        dice_all_val.append(dice_val)

        # save model if pass condition
        if dice_train > dice_log:
            checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, save_model_path)
            logging.info("==> Saving checkpoint at epoch %d", epoch + 1)
            dice_log = dice_train


    #     # Print predictions ...   
        if epoch % 3 == 0:
            save_prediction_as_imgs(epoch+1, train_loader, model,folder=prediction_img_path, device=DEVICE)
            plot_learning_curve(accuracy_all_val, score_type="accuracy", train_or_val="validation",save_path = performance_path)
            plot_learning_curve(accuracy_all_train, score_type="accuracy", train_or_val="training",save_path = performance_path)



if __name__ == "__main__":
    # make folders
    logging.info("EXPERIMENT NAME:",EXP_NAME)
    os.makedirs(f"{MODEL_PATH}/model_checkpoint/",exist_ok=True)
    result_path = f"{MODEL_PATH}/RESULTS/{EXP_NAME}"
    prediction_img_path = f"{result_path}/predicted_img"
    os.makedirs(prediction_img_path,exist_ok=True)
    performance_path = f"{result_path}/model_performance"
    os.makedirs(performance_path,exist_ok=True)
    save_model_path = f"{MODEL_PATH}/model_checkpoint/{EXP_NAME}"
    os.makedirs(save_model_path,exist_ok=True)

    main()






