import os
import logging
import argparse
import torch
import ast
import torch.optim as optim
import torch.cuda.amp as amp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import UNET, DiceLoss,MulticlassDiceLoss
from plot_dice import process_DiceLog_plot
from utils import (
    load_checkpoint, 
    save_checkpoint, 
    get_loader, 
    check_accuracy,
    save_prediction_as_imgs,
    find_newest_ModelFolder_name,)

# qrsh -pe omp 4 -P ds598 -l gpus=1
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# export MODEL_PATH=/projectnb/ds598/projects/smart_brains
# python parse_train.py --num_epochs "200" --learn_rate "1e-4" --modal_type [t1ce] --exp_name "exp"
# python parse_train.py --num_epochs "200" --learn_rate "1e-4" --modal_type [t1ce] --with_transform true --exp_name "expAugmented"
# python parse_train.py --num_epochs "200" --learn_rate "1e-4" --modal_type [flair] --exp_name "exp"
# python parse_train.py --num_epochs "200" --learn_rate "1e-4" --modal_type [flair] --with_transform true --exp_name "expAugmented"
# python parse_train.py --num_epochs "200" --learn_rate "1e-4" --modal_type [t1ce,flair] --exp_name "exp"
# python parse_train.py --num_epochs "200" --learn_rate "1e-4" --modal_type [t1ce,flair] --with_transform true --exp_name "expAugmented"

# python parse_train.py --num_epochs "2" --learn_rate "1e-4" --modal_type [t1ce,flair] --exp_name "temp"
# python parse_train.py --num_epochs "2" --learn_rate "1e-4" --modal_type [t1ce,flair] --with_transform true --exp_name "temp"


# Setup the argument parser
parser = argparse.ArgumentParser(description="Training script for U-Net model.")
parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train for.')
parser.add_argument('--learn_rate', type=float, default=1e-4, help='Learning rate for optimizer.')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
parser.add_argument('--loss_type', type=str, default='MultiDice', choices=['CrossEntropy', 'Dice', 'MultiDice'], help='Loss function to use.')
#parser.add_argument('--model_path', type=str, default='/projectnb/ds598/projects/smart_brains', help='Path to save the model.')
parser.add_argument('--modal_type', type=str, default='T1CE', help='Type of imaging modality.')
parser.add_argument('--exp_name', type=str, default='MyExperiment', help='Experiment name.')
parser.add_argument('--optimizer_name', type=str, default='Adam', help='Name of the optimizer.')
parser.add_argument('--load_model_name', type=str, default='', help='Model name to load if LOAD_MODEL is True.')
parser.add_argument('--load_last_model', action='store_true', help='Load the last model saved or start straining from scratch.')
parser.add_argument('--with_transform', type=bool, default=False, help='Model name to load if LOAD_MODEL is True.')



# Parse the arguments
args = parser.parse_args()

# Variables
NUM_EPOCHS = int(args.num_epochs)
LEARN_RATE = float(args.learn_rate)
BATCH_SIZE = int(args.batch_size)
LOSS_TYPE = args.loss_type
#MODEL_PATH = args.model_path
## Set MODEL_PATH from an environment variable
MODEL_PATH = os.getenv('MODEL_PATH', './default_model_path')
WITH_TRANSFORM = args.with_transform
if "," in args.modal_type :
    MODAL_TYPE = args.modal_type[1:-1].split(",")
else:
    MODAL_TYPE = [args.modal_type[1:-1]]
print(f"MODAL_TYPE = {MODAL_TYPE}")
EXP_NAME = args.exp_name
optimizer_name = args.optimizer_name
load_model_name = args.load_model_name

# DEVICE is determined by checking the availability of CUDA
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The rest of the constants remain unchanged
NUM_WORKERS = 4
PIN_MEMORY = True
# LOAD_MODEL = False
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 240
SPLIT_RATIO = 0.8
START_EPOCH = 0

# Determine the model name to load
if args.load_last_model:
    load_model_name = find_newest_ModelFolder_name(f"{MODEL_PATH}/RESULTS")
    if not load_model_name:
        print("No last model found. Starting training from scratch.")
else:
    load_model_name = args.load_model_name

if load_model_name: 
    LOAD_MODEL = True
    print(f"Loading model: {load_model_name}")
    # Code to load the model goes here
else:
    LOAD_MODEL = False
    print("No model specified. Starting training from scratch.")


# model initialization
n_out_channels = 4
MODEL = UNET(in_channels=len(MODAL_TYPE), out_channels=n_out_channels).to(DEVICE)

# specify optimizer
if optimizer_name == "Adam":
    OPTIMIZER = optim.Adam(MODEL.parameters(), lr=LEARN_RATE)
elif optimizer_name == "SGD":
    OPTIMIZER = optim.SGD(MODEL.parameters(), lr=LEARN_RATE)
else:
    print("Please use Adam or SGD!")

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer = OPTIMIZER, T_max = NUM_EPOCHS)
# scheduler = optim.lr_scheduler.StepLR(optimizer = OPTIMIZER, step_size=30, gamma=0.01)


if LOAD_MODEL:
    checkpoint_path = f"{MODEL_PATH}/model_checkpoint/{load_model_name}_model_checkpoint.pth.tar"
    try:
        pretrained_model, optimizer_pretrained, saved_epoch_pretrained, dice_log_pretrained = load_checkpoint(checkpoint_path, MODEL, OPTIMIZER)
        print(f"Previous model dice score is {dice_log_pretrained}")
        START_EPOCH = saved_epoch_pretrained + 1  # To continue from the next epoch
    except FileNotFoundError as e:
        print(e)

EXP_NAME = f"{EXP_NAME}_{'-'.join(MODAL_TYPE)}_op{optimizer_name}_lr{LEARN_RATE}_bs{BATCH_SIZE}_epoch{START_EPOCH}_{START_EPOCH+NUM_EPOCHS}"

logging.basicConfig(level=logging.INFO, filename=f'{MODEL_PATH}/{EXP_NAME}_training.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')

def train_function(loader, model, optimizer, loss_fn, scaler):
    model.train()
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop): 
       # print(data.size())
        #data = data.squeeze(1) 
        
        data = data.float().to(device=DEVICE)
        targets = targets.squeeze(1).long().to(device=DEVICE)  
        #targets = targets.long().to(device=DEVICE)  


        with amp.autocast():  
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())
    
    scheduler.step()

def plot_learning_curve(score_allepoch, score_type, train_or_val, save_path):
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
        plt.plot(epochs, line, label=f'Class {i}')

    plt.xlabel('Epoch Number')
    plt.ylabel(score_type.capitalize())
    plt.title(f'{score_type.capitalize()} by Epoch')
    plt.legend()
    plt.grid(True)
    # plt.xticks(epochs)  # Ensure all epochs are shown
    file_path = f"{save_path}/{train_or_val}_{score_type}_{EXP_NAME}.png"
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close()



def main():
    #define transform
    with_transformation = WITH_TRANSFORM
    if with_transformation:
        print("with transformation")
        prob = 0.5
    else:
        print("No transformation")
        prob = 0
    
    transform_augment = A.Compose([
        A.GaussNoise(var_limit=(0.00001, 0.00001), p=prob),
        A.HorizontalFlip(p=prob),
        A.VerticalFlip(p=prob),
        A.Rotate(limit=10, p=prob),  # Small rotation, up to 10 degrees
        A.ShiftScaleRotate(shift_limit=0.025, scale_limit=0.0, rotate_limit=0, p=prob),  # Small shift, no scaling
        A.Affine(shear={'x': (-5, 5), 'y': (-5, 5)}, p=prob),  # Small shear, -5 to +5 degrees
        ToTensorV2()
        ])
    
    # Create the loss function based on LOSS_TYPE
    model = MODEL
    optimizer = OPTIMIZER
    if LOSS_TYPE == "CrossEntropy":
        loss_fn = nn.CrossEntropyLoss()
    elif LOSS_TYPE == "Dice":
        loss_fn = DiceLoss()
    elif LOSS_TYPE == "MultiDice":
        loss_fn = MulticlassDiceLoss()
        
    else:
        logging.error('Need to define loss function.')

    
 
    # Pass appropriate arguments into get_loader
    train_loader,val_loader = get_loader(MODAL_TYPE = MODAL_TYPE,
                                         BATCH_SIZE = BATCH_SIZE, 
                                         PIN_MEMORY = PIN_MEMORY,
                                         SPLIT_RATIO = SPLIT_RATIO,
                                         transform = transform_augment)
    
    
    scaler = amp.GradScaler()
    
    accuracy_all_val = []
    dice_all_val = []
    accuracy_all_train = []
    dice_all_train = []
    dice_log = 0 

    if LOAD_MODEL:
        model = pretrained_model
        optimizer = optimizer_pretrained
        dice_log = dice_log_pretrained
        
        # model, optimizer, start_epoch, dice_log = load_checkpoint(f"{MODEL_PATH}/model_checkpoint/{load_model_name}_model_checkpoint.pth.tar", model)

    for epoch in range(START_EPOCH, START_EPOCH + NUM_EPOCHS):
        print("Epoch:", epoch + 1)
        logging.info("Epoch: %d", epoch + 1)
        train_function(train_loader, model, optimizer, loss_fn, scaler)  # Using dice_loss for training
        logging.info(f"learning rate = {optimizer.param_groups[0]['lr']}")
        
        
        logging.info("===Training===")
        accuracy_train, dice_train = check_accuracy("Training", train_loader, model, device=DEVICE,NUM_CLASSES = n_out_channels)
        accuracy_all_train.append(accuracy_train)
        dice_all_train.append(dice_train)

        logging.info("===Validation===")
        accuracy_val, dice_val = check_accuracy("Validation", val_loader, model, device=DEVICE,NUM_CLASSES = n_out_channels)
        accuracy_all_val.append(accuracy_val)
        dice_all_val.append(dice_val)

        # save model if pass condition
        print("dice_val",dice_val)
        print("dice_log", dice_log)
        if dice_val > dice_log:
            checkpoint = {
            'epoch': epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "dice_log": dice_log
            }
            save_checkpoint(checkpoint, save_model_path)
            save_prediction_as_imgs(epoch+1, val_loader, model,folder=prediction_img_path, device=DEVICE) # Do we plot? for validation
            logging.info("==> Saving checkpoint at epoch %d", epoch + 1)
            dice_log = dice_val
        



    #     # Print predictions ...   
        plot_learning_curve(accuracy_all_val, score_type="accuracy", train_or_val="validation",save_path = performance_path)
        plot_learning_curve(accuracy_all_train, score_type="accuracy", train_or_val="training",save_path = performance_path)
    process_DiceLog_plot(performance_path, '-'.join(MODAL_TYPE), EXP_NAME, [f'{MODEL_PATH}/{EXP_NAME}_training.log'])



if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # make folders
    logging.info("EXPERIMENT NAME: %s", EXP_NAME) 
    save_model_path = f"{MODEL_PATH}/model_checkpoint/{EXP_NAME}"
    result_path = f"{MODEL_PATH}/RESULTS/{EXP_NAME}"
    prediction_img_path = f"{result_path}/predicted_img"
    os.makedirs(prediction_img_path,exist_ok=True)
    performance_path = f"{result_path}/model_performance"
    os.makedirs(performance_path,exist_ok=True)
    # os.makedirs(save_model_path,exist_ok=True)
    main()






