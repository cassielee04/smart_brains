import os
import logging
import torch
import torch.optim as optim
import torch.cuda.amp as amp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import UNET, DiceLoss, MulticlassDiceLoss
from utils import (
    load_checkpoint, 
    save_checkpoint, 
    get_loader, 
    check_accuracy,
    save_prediction_as_imgs,)

# qrsh -pe omp 4 -P ds598 -l gpus=1
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

NUM_EPOCHS = 50
LEARN_RATE = 1e-8  #reduced to 1e-5 for epoch 30-50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")#"cuda" #"cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 16
NUM_WORKERS = 4
PIN_MEMORY = True
LOAD_MODEL = True
LOSS_TYPE = "MultiDice" # "MultiDice" #"CrossEntropy" # or "BCEWithLogitLoss", "Dice"
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 240
SPLIT_RATIO = 0.8
MODEL_PATH = "/projectnb/ds598/projects/smart_brains"
MODAL_TYPE = ["Flair","T1ce"]
EXP_NAME = "Selena"
optimizer_name = "Adam"
load_model_name = "Selena_Flair-T1ce_opAdam_lr1e-06_bs16_epoch13_33" # "SelenaTemp_Flair-T1ce_opAdam_lr0.0001_bs16_epoch0_20"
START_EPOCH = 0

FineTune = False

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

# scheduler = optim.lr_scheduler.CosineAnnealingLR(OPTIMIZER, Tmax = NUM_EPOCHS)

if LOAD_MODEL:
    pretrained_model, optimizer_pretrained, saved_epoch_pretrained, dice_log_pretrained = load_checkpoint(f"{MODEL_PATH}/model_checkpoint/{load_model_name}_model_checkpoint.pth.tar", MODEL, OPTIMIZER)
    print(f"previous model dice score is {dice_log_pretrained}")
    START_EPOCH = saved_epoch_pretrained


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
    
    # scheduler.step()

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
        plt.plot(epochs, line, label=f'Line {i+1}')

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
    # transform_augment = A.Compose([
    #     A.ElasticTransform(alpha=1, sigma=30, alpha_affine=30, p=0.5),# or p=0.5 apply with prob 0.5.
    #     ToTensorV2()  # Convert to PyTorch tensor 
    # ])
    transform_augment = A.Compose([
        A.GaussNoise(var_limit=(0.001,0.001), p=0.5),
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
        if not FineTune:
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



if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # make folders
    logging.info("EXPERIMENT NAME:",EXP_NAME)
    save_model_path = f"{MODEL_PATH}/model_checkpoint/{EXP_NAME}"
    #os.makedirs(save_model_path,exist_ok=True)
    result_path = f"{MODEL_PATH}/RESULTS/{EXP_NAME}"
    prediction_img_path = f"{result_path}/predicted_img"
    os.makedirs(prediction_img_path,exist_ok=True)
    performance_path = f"{result_path}/model_performance"
    os.makedirs(performance_path,exist_ok=True)
    # os.makedirs(save_model_path,exist_ok=True)
    main()






