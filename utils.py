import torch
import torchvision
from dataset import BRATS21_Dataset
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as functional
import os
import matplotlib.pyplot as plt


def save_checkpoint(state,expName):
    filename=f"{expName}_model_checkpoint.pth.tar"
    print("==> Saving checkpoint")
    torch.save(state,filename)

def load_checkpoint(checkpoint,model):
    print("==> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loader(
        modal_type = "t1ce",
        BATCH_SIZE = 8,
        PIN_MEMORY = True, # Setting to True, it enables fast data transfer to CUDA-enabled GPUs
        SPLIT_RATIO = 0.8,  # train_val split
        training_folder_path = "/projectnb/ds598/projects/smart_brains/dataset/MICCAI_FeTS2021_TrainingData",
        testing_folder_path = "/projectnb/ds598/projects/smart_brains/dataset/MICCAI_FeTS2021_ValidationData_nolabel"
    ):
    # get image/label file path saved
    training_folder_names = sorted([fname for fname in os.listdir(training_folder_path) if "FeTS21" in fname])
    print(f"There are {len(training_folder_names)} Training Data, from {training_folder_names[0].split('_')[-1]} to {training_folder_names[-1].split('_')[-1]}")

    # these are unlabeled files
    testing_folder_names = sorted([fname for fname in os.listdir(testing_folder_path) if "FeTS21" in fname])
    print(f"There are {len(testing_folder_names)} Testing Data, from {testing_folder_names[0].split('_')[-1]} to {testing_folder_names[-1].split('_')[-1]}")

    training_image_paths = [f"{training_folder_path}/{training_folder_name}/{training_folder_name}_{modal_type}.nii" for training_folder_name in training_folder_names]
    training_label_paths = [f"{training_folder_path}/{training_folder_name}/{training_folder_name}_seg.nii" for training_folder_name in training_folder_names]
    testing_image_paths = [f"{testing_folder_path}/{testing_folder_name}/{testing_folder_name}_{modal_type}.nii" for testing_folder_name in testing_folder_names]
    
    # create dataset object
    train_val_data = BRATS21_Dataset(training_image_paths, training_label_paths)
    train_size = int(SPLIT_RATIO * train_val_data.__len__())
    val_size = train_val_data.__len__() - train_size
    # Generate indices: train_indices for the first part, val_indices for the second
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    # Create training and validation subsets
    training_Dataset = Subset(train_val_data, train_indices)
    validation_Dataset = Subset(train_val_data, val_indices)
    #training_Dataset, validation_Dataset = torch.utils.data.random_split(train_val_data, [train_size, val_size])
    print(f"Training dataset size = {train_size}")
    print(f"Validation dataset size = {val_size}")

    # create data loader
    training_dataloader = DataLoader(training_Dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY)
    validation_dataloader = DataLoader(validation_Dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY)
    
    train_features, train_labels = next(iter(training_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    
    return training_dataloader, validation_dataloader

def dice_score(input_tensor, target_tensor):
    smooth = 1e-6
    intersection = torch.sum(input_tensor * target_tensor)
    union = torch.sum(input_tensor) + torch.sum(target_tensor)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice
    
def check_accuracy(loader, model, device="cuda", NUM_CLASSES = 4):
    #change
    num_correct = [0,0,0,0]
    num_pixels = [0,0,0,0]
    accuracy = [0,0,0,0]
    dice =  0
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        for data, targets in loader:
            # Feature batch shape: torch.Size([16, 240, 240])
            # Labels batch shape: torch.Size([16, 240, 240])
            data, targets = data.float().unsqueeze(1).to(device), targets.squeeze(1).long().to(device)  # Corrected variable name (labels to targets)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
            #print("predicted size:",predicted.size())
            OneHotPred = functional.one_hot(predicted,num_classes=NUM_CLASSES)
            #print("OneHotPred sizde:",OneHotPred.size())
            OneHotTargets = functional.one_hot(targets,num_classes=NUM_CLASSES)
            #print("OneHotTargets size:",OneHotTargets.size())
            # Permute the OneHotPred to shape [sub_id, num_class, height, width]
            OneHotPred = OneHotPred.permute(0, 3, 1, 2)
            OneHotTargets = OneHotTargets.permute(0, 3, 1, 2)

            for i in range(NUM_CLASSES):
                input_layer = OneHotPred[:,i, :, :]
                target_layer = OneHotTargets[:, i, :, :]
                num_correct[i] += (input_layer == target_layer).sum().item()
                num_pixels[i] += target_layer.numel()
            dice += dice_score(OneHotPred, OneHotTargets)
            
        for i in range(NUM_CLASSES):
            accuracy[i] = num_correct[i] / num_pixels[i]  
    dice = dice/len(loader)
    print(f"Accuracy: {accuracy}")
    print(f"Average Dice Score: {dice}")
    model.train()
    return accuracy, dice


def torch2img(predicted_tensor, targets_tensor, id_in_batch, file_path):
    # image_data = tensor[id_in_batch, :, :].cpu().numpy()
    # plt.imshow(image_data)
    # plt.axis('off')
    # plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    # cmap = plt.get_cmap('tab10')
    # image_data = tensor[id_in_batch, :, :].cpu().numpy()
    # plt.imshow(image_data, cmap=cmap, interpolation='nearest', vmin=0, vmax=3)
    # plt.colorbar(ticks=[0, 1, 2, 3])  # Adjust colorbar ticks
    # plt.savefig(file_path, bbox_inches='tight', pad_inches=0)


    predicted = predicted_tensor[id_in_batch, :, :].cpu().numpy()
    targets = targets_tensor[id_in_batch, :, :].cpu().numpy()
    # Plotting both arrays side by side
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # Using the same colormap for both for consistency
    cmap = plt.get_cmap('viridis')
    
    # Targets
    ax[0].imshow(targets, cmap=cmap, interpolation='nearest')
    ax[0].set_title('Targets')
    ax[0].axis('off')  # Hide the axes
    
    # Predicted
    ax[1].imshow(predicted, cmap=cmap, interpolation='nearest')
    ax[1].set_title('Predicted')
    ax[1].axis('off')  # Hide the axes
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    

def save_prediction_as_imgs(epoch,loader, model, folder, device="cuda"):
    model.eval()
    for idx, (data, targets) in enumerate(loader):
        
        data = data.float().unsqueeze(1).to(device)
        #print("data",data)
        with torch.no_grad():
            outputs = model(data)
            #print("output",outputs)
            _, predicted = torch.max(outputs, 1)

        torch2img(predicted, targets, id_in_batch=0, file_path=f"{folder}/pred_{idx}_epoch{epoch}.png")
    model.train()









