import torch
import torchvision
from dataset import BRATS21_Dataset
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as functional
import os
import logging
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def get_last_model_name(model_path):
    last_model_file_path = os.path.join(model_path, "last_model.txt")
    if os.path.exists(last_model_file_path):
        with open(last_model_file_path, 'r') as file:
            return file.read().strip()
    return ""


def save_last_model_name(model_name, save_path):
    """
    Saves the last model name to a file.

    Parameters:
    - model_name: The name of the model to save.
    - save_path: The directory path where the model and "last_model.txt" are saved.
    """
    last_model_file_path = os.path.join(save_path, "last_model.txt")
    with open(last_model_file_path, 'w') as file:
        file.write(model_name)

def save_checkpoint(state,expName): 
    filename=f"{expName}_model_checkpoint.pth.tar"
    print("==> Saving checkpoint")
    torch.save(state, filename)
    # Use rpartition to split the string at the last '/'
    # base_path, _, tail = path.rpartition('/')

    # Add the '/' back to the base path for correctness
    # base_path += '/'
    # save_last_model_name(tail,base_path)


def load_checkpoint(checkpoint_fpath, model, optimizer):
    print("==> Loading checkpoint")
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    dice_log = checkpoint['dice_log']
    
    return model, optimizer, checkpoint['epoch'], dice_log.item()


def get_loader(
        MODAL_TYPE = ["t1ce"],
        BATCH_SIZE = 8,
        PIN_MEMORY = True, # Setting to True, it enables fast data transfer to CUDA-enabled GPUs
        SPLIT_RATIO = 0.8,  # train_val split
        NUM_WORKERS = 0,
        training_folder_path = "/projectnb/ds598/projects/smart_brains/dataset/MICCAI_FeTS2021_TrainingData",
        testing_folder_path = "/projectnb/ds598/projects/smart_brains/dataset/MICCAI_FeTS2021_ValidationData_nolabel",
        transform = None 
    ):
    # get image/label file path saved
    training_folder_names = sorted([fname for fname in os.listdir(training_folder_path) if "FeTS21" in fname])
    print(f"There are {len(training_folder_names)} Training Data, from {training_folder_names[0].split('_')[-1]} to {training_folder_names[-1].split('_')[-1]}")

    # these are unlabeled files
    # testing_folder_names = sorted([fname for fname in os.listdir(testing_folder_path) if "FeTS21" in fname])
    # print(f"There are {len(testing_folder_names)} Testing Data, from {testing_folder_names[0].split('_')[-1]} to {testing_folder_names[-1].split('_')[-1]}")
    
    MODAL_TYPE = [modal.lower() for modal in MODAL_TYPE]

    #training_image_paths = [f"{training_folder_path}/{training_folder_name}/{training_folder_name}_{MODAL_TYPE}.nii" for training_folder_name in training_folder_names]
    training_image_paths = [[f"{training_folder_path}/{training_folder_name}/{training_folder_name}_{modality}.nii" for modality in MODAL_TYPE] for training_folder_name in training_folder_names]
    training_label_paths = [f"{training_folder_path}/{training_folder_name}/{training_folder_name}_seg.nii" for training_folder_name in training_folder_names]
    
    # testing_image_paths = [f"{testing_folder_path}/{testing_folder_name}/{testing_folder_name}_{MODAL_TYPE}.nii" for testing_folder_name in testing_folder_names]
    
    # create dataset object
    train_val_data = BRATS21_Dataset(training_image_paths, training_label_paths, transform=transform)
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
    training_dataloader = DataLoader(training_Dataset, 
                                     batch_size=BATCH_SIZE, 
                                     shuffle=True, 
                                     num_workers=NUM_WORKERS, 
                                     pin_memory=PIN_MEMORY,)
    validation_dataloader = DataLoader(validation_Dataset, 
                                       batch_size=BATCH_SIZE, 
                                       shuffle=False, 
                                       num_workers=NUM_WORKERS, 
                                       pin_memory=PIN_MEMORY)
    
    train_features, train_labels = next(iter(training_dataloader))
    train_features = train_features.squeeze(1)
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    
    return training_dataloader, validation_dataloader

def dice_score(input_tensor, target_tensor):
    smooth = 1e-6
    intersection = torch.sum(input_tensor * target_tensor)
    union = torch.sum(input_tensor) + torch.sum(target_tensor)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice
    
def check_accuracy(TYPE, loader, model, device="cuda", NUM_CLASSES = 4):
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
            #data, targets = data.float().unsqueeze(1).to(device), targets.squeeze(1).long().to(device)  
            data, targets = data.float().to(device), targets.squeeze(1).long().to(device)# Corrected variable name (labels to targets)
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

            for i in range(0, NUM_CLASSES):
                input_layer = OneHotPred[:,i, :, :]
                target_layer = OneHotTargets[:, i, :, :]
                num_correct[i] += (input_layer == target_layer).sum().item()
                num_pixels[i] += target_layer.numel()
                # ignore background
            dice += dice_score(OneHotPred[:,1:,:,:], OneHotTargets[:,1:,:,:])
            
        for i in range(NUM_CLASSES):
            accuracy[i] = num_correct[i] / num_pixels[i]  
    dice = dice/len(loader)
    # print(f"Accuracy: {accuracy}")
    # print(f"Average Dice Score: {dice}")
    logging.info(f"{TYPE} Accuracy per class: {accuracy}")
    logging.info(f"{TYPE} Average Dice Score: {dice}")
    #model.train()
    return accuracy, dice



def show_dice_score(predicted_tensor, targets_tensor, NUM_CLASSES = 4):
    OneHotPred = functional.one_hot(predicted_tensor,num_classes=NUM_CLASSES)
    OneHotTargets = functional.one_hot(targets_tensor,num_classes=NUM_CLASSES)
    OneHotPred = OneHotPred.permute(0, 3, 1, 2)
    OneHotTargets = OneHotTargets.permute(0, 3, 1, 2)
    return dice_score(OneHotPred[:,1:,:,:], OneHotTargets[:,1:,:,:])


def torch2img_image_overlay_multimodal(image_tensor, predicted_tensor, targets_tensor, id_in_batch, file_path):
    # Convert tensors to numpy arrays
    image = image_tensor[id_in_batch].squeeze().cpu().numpy()
    image_t1ce = image[0,:,:]
    image_flair = image[1,:,:]
    predicted = predicted_tensor[id_in_batch, :, :].cpu().numpy()
    targets = targets_tensor[id_in_batch, :, :].cpu().numpy()

    #####dice_score = show_dice_score(predicted_tensor, targets_tensor, NUM_CLASSES = 4)

    colors = [(0, 0, 0, 0),  # Transparent
              (0.94, 0.5, 0.5, 1),  # Coral
              (0, 0.5, 0.5, 1),  # Teal
              (1, 0.84, 0, 1)]  # Gold
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    # Original Image
    ax[0,0].imshow(image_t1ce, cmap='gray')  # Assuming the original image is grayscale
    ax[0,0].set_title('Targets on Original T1CE Image')
    ax[0,0].axis('off')  # Hide the axes
    ax[0,0].imshow(targets, cmap=cmap, alpha=0.7, interpolation='nearest', vmin=0, vmax=len(colors)-1)

    # Predicted Label
    ax[0,1].imshow(image_t1ce, cmap='gray')  # Colormap for predicted labels
    ax[0,1].set_title('Predicted on Original T1CE Image')
    ax[0,1].axis('off')  # Hide the axes
    ax[0,1].imshow(predicted, cmap=cmap, alpha=0.7, interpolation='nearest', vmin=0, vmax=len(colors)-1)

    ax[1,0].imshow(image_flair, cmap='gray')  # Assuming the original image is grayscale
    ax[1,0].set_title('Targets on Original FLAIR Image')
    ax[1,0].axis('off')  # Hide the axes
    ax[1,0].imshow(targets, cmap=cmap, alpha=0.7, interpolation='nearest', vmin=0, vmax=len(colors)-1)

    # Predicted Label
    ax[1,1].imshow(image_flair, cmap='gray')  # Colormap for predicted labels
    ax[1,1].set_title('Predicted on Original FLAIR Image')
    ax[1,1].axis('off')  # Hide the axes
    ax[1,1].imshow(predicted, cmap=cmap, alpha=0.7, interpolation='nearest', vmin=0, vmax=len(colors)-1)
    

    #####fig.suptitle(f'Dice Score = {dice_score:.4f}', fontsize=16)

    # Save the figure
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure to free memory


def torch2img_image_overlay(image_tensor, predicted_tensor, targets_tensor, id_in_batch, file_path):
    # Convert tensors to numpy arrays
    image = image_tensor[id_in_batch].squeeze().cpu().numpy()
    predicted = predicted_tensor[id_in_batch, :, :].cpu().numpy()
    targets = targets_tensor[id_in_batch, :, :].cpu().numpy()

    #####dice_score = show_dice_score(predicted_tensor, targets_tensor, NUM_CLASSES = 4)
    
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
    
    #####fig.suptitle(f'Dice Score = {dice_score:.4f}', fontsize=16)

    # Save the figure
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure to free memory


# def torch2img_three(predicted_tensor, targets_tensor, id_in_batch, file_path):
#     # Convert tensors to numpy arrays
#     predicted = predicted_tensor[id_in_batch, :, :].cpu().numpy()
#     targets = targets_tensor[id_in_batch, :, :].cpu().numpy()

#     dice_score = show_dice_score(predicted_tensor, targets_tensor, NUM_CLASSES = 4)
    
#     # Define a custom color map for predictions
#     prediction_colors = [(1, 1, 1, 1),  # White
#                          (0.94, 0.5, 0.5, 1),  # Coral
#                          (0, 0.5, 0.5, 1),  # Teal
#                          (1, 0.84, 0, 1)]  # Gold
#     prediction_cmap = ListedColormap(prediction_colors)

#     # Define a custom color map for targets
#     target_colors = [(1, 1, 1, 1),  # White
#                      (0.75, 0.75, 0.75, 1),  # Light Gray
#                      (0.5, 0.5, 0.5, 1),  # Medium Gray
#                      (0.25, 0.25, 0.25, 1)]  # Dark Gray
#     target_cmap = ListedColormap(target_colors)

#     # Creating a figure with 3 subplots
#     fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
#     # Target image
#     axs[0].imshow(targets, cmap=target_cmap, interpolation='nearest', vmin=0, vmax=len(target_colors)-1)
#     axs[0].set_title('Target')
#     axs[0].axis('off')
    
#     # Predicted image
#     axs[1].imshow(predicted, cmap=prediction_cmap, interpolation='nearest', vmin=0, vmax=len(prediction_colors)-1)
#     axs[1].set_title('Predicted')
#     axs[1].axis('off')
    
#     # Overlay of predicted on targets
#     axs[2].imshow(targets, cmap=target_cmap, interpolation='nearest', vmin=0, vmax=len(target_colors)-1)
#     axs[2].imshow(predicted, cmap=prediction_cmap, alpha=0.5, interpolation='nearest', vmin=0, vmax=len(prediction_colors)-1)
#     axs[2].set_title('Overlay: Target & Predicted')
#     axs[2].axis('off')

#     fig.suptitle(f'Dice Score = {dice_score:.4f}', fontsize=16)

#     # Save the combined figure
#     plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
#     plt.close()

# def torch2img_overlay(predicted_tensor, targets_tensor, id_in_batch, file_path):
#     # Convert tensors to numpy arrays
#     predicted = predicted_tensor[id_in_batch, :, :].cpu().numpy()
#     targets = targets_tensor[id_in_batch, :, :].cpu().numpy()
    
#     # Define a custom color map: you can adjust the RGBA values as needed
#     colors = [(1, 1, 1, 1),  # White
#               (0.94, 0.5, 0.5, 1),  # Coral
#               (0, 0.5, 0.5, 1),  # Teal
#               (1, 0.84, 0, 1)]  # Gold
#     cmap = ListedColormap(colors)

#     target_colors = [(1, 1, 1, 1),  # White
#                      (0.75, 0.75, 0.75, 1),  # Light Gray
#                      (0.5, 0.5, 0.5, 1),  # Medium Gray
#                      (0.25, 0.25, 0.25, 1)]  # Dark Gray
#     target_cmap = ListedColormap(target_colors)


#     # Plotting both arrays overlaid
#     fig, ax = plt.subplots(figsize=(5, 5))

#     # Targets with custom colormap
#     ax.imshow(targets, cmap=target_cmap, interpolation='nearest', vmin=0, vmax=len(colors)-1)
#     ax.set_title('Overlay: Targets & Predicted')
#     ax.axis('off')  # Hide the axes

#     # Predicted with custom colormap and transparency
#     ax.imshow(predicted, cmap=cmap, alpha=0.5, interpolation='nearest', vmin=0, vmax=len(colors)-1)

#     # Save the figure
#     plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
#     plt.close()  # Close the figure to free memory
    

# def torch2img(predicted_tensor, targets_tensor, id_in_batch, file_path):

#     # Convert tensors to numpy arrays
#     predicted = predicted_tensor[id_in_batch, :, :].cpu().numpy()
#     targets = targets_tensor[id_in_batch, :, :].cpu().numpy()
    
#     # Define a custom color map: you can adjust the RGBA values as needed
#     colors = [(1, 1, 1, 1),  # White
#               (0.94, 0.5, 0.5, 1),  # Coral
#               (0, 0.5, 0.5, 1),  # Teal
#               (1, 0.84, 0, 1)]  # Gold
#     cmap = ListedColormap(colors)

#     # Plotting both arrays side by side
#     fig, ax = plt.subplots(1, 2, figsize=(10, 5))

#     # Targets with custom colormap
#     ax[0].imshow(targets, cmap=cmap, interpolation='nearest', vmin=0, vmax=len(colors)-1)
#     ax[0].set_title('Targets')
#     ax[0].axis('off')  # Hide the axes

#     # Predicted with custom colormap
#     ax[1].imshow(predicted, cmap=cmap, interpolation='nearest', vmin=0, vmax=len(colors)-1)
#     ax[1].set_title('Predicted')
#     ax[1].axis('off')  # Hide the axes

#     # Save the figure
#     plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
#     plt.close()  # Close the figure to free memory
    

def save_prediction_as_imgs(epoch, loader, model, folder, device="cuda"):
    model.eval()
    for idx, (data, targets) in enumerate(loader):
        
        #data = data.float().unsqueeze(1).to(device)
        data = data.float().to(device)
        #print("data",data)
        with torch.no_grad():
            outputs = model(data)
            #print("output",outputs)
            _, predicted = torch.max(outputs, 1)

        # torch2img(predicted, targets, id_in_batch=0, file_path=f"{folder}/pred_{idx}_epoch{epoch}.png")
        # torch2img(predicted, targets, id_in_batch=0, file_path=f"{folder}/pred_{idx}_epoch_best.png")
        # torch2img_overlay(predicted, targets, id_in_batch=0, file_path=f"{folder}/pred_{idx}_epoch_best.png")
        for id_in_batch in range(data.size(0)):  # Loop through the batch
            if data.size(1) == 1:
                torch2img_image_overlay(data, predicted, targets, id_in_batch, file_path=f"{folder}/pred_{idx}_{id_in_batch}_epoch_best.png")
            else: 
                assert data.size(1) == 2
                torch2img_image_overlay_multimodal(data, predicted, targets, id_in_batch, file_path=f"{folder}/pred_{idx}_{id_in_batch}_epoch_best.png")
           
    #model.train()

