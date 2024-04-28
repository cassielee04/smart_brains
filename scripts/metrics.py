import os
import argparse
import logging
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import UNET, DiceLoss,MulticlassDiceLoss
from dataset import BRATS21_Dataset
from torch.utils.data import DataLoader
from utils import (
    get_loader, 
    dice_score,
    )



def explore_models(model_folder, load_model_names):
    """
    Explore model names either from the provided list or from the model folder.

    Parameters:
        model_folder (str): Path to the folder containing model names.
        load_model_names (list): List of specific model names to load.

    Returns:
        list: Model names to load.
    """
    if load_model_names:
        return load_model_names
    else:
        # Explore all model names in the folder
        model_names =  [os.path.basename(f.path) for f in os.scandir(model_folder) if f.is_dir()]
        return model_names


def load_model_for_inference(checkpoint_path, model):
    """
    Load a trained model from a checkpoint file for inference.

    Parameters:
        checkpoint_path (str): Path to the model checkpoint file.
        model (torch.nn.Module): The model architecture, must match the architecture of the trained model.

    Returns:
        torch.nn.Module: The loaded model ready for inference.
    """
    # Load the saved model checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Update model with the saved weights
    model.load_state_dict(checkpoint['state_dict'])
    
    # Set the model to evaluation mode
    model.eval()

    return model


def compute_confusion_matrices_dice(loader, model, device="cuda", NUM_CLASSES=4):
    # Initialize a list of confusion matrices, one per class
    confusion_matrices = [torch.zeros(2, 2, dtype=torch.int64) for _ in range(NUM_CLASSES)]
    
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for data, targets in loader:
            #data= data.permute(0, 3, 1, 2).float().to(device)
            data= data.float().to(device)
            targets = targets.squeeze(1).long().to(device)  # Corrected variable name (labels to targets)
            print("datasize", data.size())
            print("targetsize", targets.size())
            # data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            
            dice_scores = []
            # Loop over each class and update the confusion matrix
            for class_index in range(NUM_CLASSES):
                # For the current class, calculate true positives, false positives, true negatives, and false negatives
                true_positive = ((predicted == class_index) & (targets == class_index)).sum().item()
                false_positive = ((predicted == class_index) & (targets != class_index)).sum().item()
                true_negative = ((predicted != class_index) & (targets != class_index)).sum().item()
                false_negative = ((predicted != class_index) & (targets == class_index)).sum().item()

                #compute dice score
                input_tensor = (predicted == class_index)
                target_tensor = (targets == class_index)
                dice_scores.append(dice_score(input_tensor,target_tensor).cpu().numpy())
                
                # Update the confusion matrix for the current class
                confusion_matrices[class_index][0, 0] += true_negative
                confusion_matrices[class_index][0, 1] += false_positive
                confusion_matrices[class_index][1, 0] += false_negative
                confusion_matrices[class_index][1, 1] += true_positive

    return confusion_matrices,dice_scores

def plot_confusion_matrices(confusion_matrices, class_names, model_name):
    """
    Plots confusion matrices.
    
    Parameters:
    - confusion_matrices: List of confusion matrices, one per class.
    - class_names: List of names of the classes.
    """
    num_classes = len(confusion_matrices)
    fig, axes = plt.subplots(1, num_classes, figsize=(num_classes * 8, 6), squeeze=False)
    
    for i, cm in enumerate(confusion_matrices):
        ax = axes.flat[i]
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=['Not Class '+class_names[i], 'Class '+class_names[i]],
               yticklabels=['Not Class '+class_names[i], 'Class '+class_names[i]],
               title=f'Confusion Matrix for Class {class_names[i]} with {model_name}',
               ylabel='True label',
               xlabel='Predicted label')
        
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Loop over data dimensions and create text annotations.
        fmt = 'd'
        thresh = cm.max() / 2.
        for y in range(cm.shape[0]):
            for x in range(cm.shape[1]):
                ax.text(x, y, format(cm[y, x], fmt),
                        ha="center", va="center",
                        color="white" if cm[y, x] > thresh else "black")
    fig.tight_layout()
    plt.savefig(f"{MAIN_PATH}/{model_name}_confusion_matrix.png")

def calculate_metrics(confusion_matrix):
    # Extracting True Positives, True Negatives, False Positives, and False Negatives
    TP = confusion_matrix[1, 1]
    TN = confusion_matrix[0, 0]
    FP = confusion_matrix[0, 1]
    FN = confusion_matrix[1, 0]
    
    # Calculating the metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    
    return accuracy, precision, recall, specificity,f1

def find_modalities(input_string):
    """
    Identifies specific modalities in a given string and returns a list of found modalities in the order they first appear.
    Ensures that 't1ce' is recognized distinctly from 't1'..

    Parameters:
        input_string (str): The string to search for modalities.

    Returns:
        list: A list of modalities found in the string.
    """
    # Convert the string to lowercase to ensure case-insensitive matching
    lower_string = input_string.lower()

    # Dictionary to track modalities and their first index of appearance
    modality_indices = {}

    # Search for modalities and record the index of their first appearance
    for modality in ['t1ce', 'flair', 't1', 't2']:
        index = lower_string.find(modality)
        if index != -1:
            # Check for t1 to ensure it's not just part of t1ce
            if modality == 't1' and 't1ce' in lower_string and lower_string.find('t1ce') == index:
                continue
            modality_indices[modality] = index

    # Sort modalities by the index of their first appearance
    sorted_modalities = sorted(modality_indices, key=modality_indices.get)

    return sorted_modalities



def main():
    class_names = ["0:NoTumor","1:NecroticCore","2:Edema","3:Enhancing"]
    NUM_CLASSES = 4
    df_all = pd.DataFrame(columns=['Model Name', 'Class Name', 'Accuracy', 'Precision', 'Recall', 'Specificity','F1','Dice'])
        # model initialization
    for model_name in load_model_names:
        modalities = find_modalities(model_name)
        model = UNET(in_channels=len(modalities), out_channels=NUM_CLASSES).to(DEVICE)
        load_path = f"{MAIN_PATH}/model_checkpoint/{model_name}_model_checkpoint.pth.tar"
        model = load_model_for_inference(load_path, model)
        _,val_loader = get_loader(MODAL_TYPE = modalities,
                                            BATCH_SIZE = 500, 
                                            PIN_MEMORY = PIN_MEMORY,
                                            SPLIT_RATIO = SPLIT_RATIO)
        CMs,dice_score = compute_confusion_matrices_dice(val_loader, model, device="cuda")
        confusion_matrices = [tensor.numpy() for tensor in CMs]

        print(f"confusion_matrices of model {model_name}", confusion_matrices)
        plot_confusion_matrices(confusion_matrices , class_names, model_name)
        model_parts = model_name.split('_')[:-5]
        model_name = '_'.join(model_parts)
            # print(model_name)

        for i, class_name in enumerate(class_names):
            cm = confusion_matrices[i]
            accuracy, precision, recall, specificity, f1 = calculate_metrics(cm)
            
            # Append data to the DataFrame
            df_all = pd.concat([ df_all, pd.DataFrame([[model_name,class_name,accuracy,precision,recall,specificity,f1,dice_score[i]]], columns=df_all.columns)], ignore_index=True)
        df_sorted = df_all.sort_values(by=['Class Name', 'Model Name'])
        print(df_sorted)
    df_sorted.to_csv(f"{MAIN_PATH}/metrics.csv",index=False)


if __name__ == "__main__":
    MAIN_PATH = os.getenv('MODEL_PATH', './default_model_path')
    print("MAIN_PATH",MAIN_PATH)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PIN_MEMORY = True
    SPLIT_RATIO = 0.8

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Explore model names.")

    # Add arguments
    parser.add_argument("--model_folder", type=str, default="path/to/model/RESULTr", help="Path to the folder containing all models, metric will be computed for all models")
    parser.add_argument("--load_model_names", nargs="+", help="List of specific model names to load.")

    # Parse the arguments
    args = parser.parse_args()

    # Call the explore_models function
    load_model_names = explore_models(args.model_folder, args.load_model_names)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #load_model_names = ["expAugmented_t1ce_opAdam_lr0.0001_bs16_epoch0_200","expAugmented_flair_opAdam_lr0.0001_bs16_epoch0_200","expAugmented_t1ce-flair_opAdam_lr0.0001_bs16_epoch0_200","exp_t1ce_opAdam_lr0.0001_bs16_epoch0_200","exp_flair_opAdam_lr0.0001_bs16_epoch0_200","exp_t1ce-flair_opAdam_lr0.0001_bs16_epoch0_200"]
    print("Models to load:", load_model_names)
    main()
    
    
    