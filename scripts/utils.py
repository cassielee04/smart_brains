import torch
import torchvision
from dataset import brainDataset
from torch.utils.data import DataLoader
import torch.nn.functional as functional

def save_checkpoint(state,filenames="model_checkpoint.pth.tar"):
    print("==> Saving checkpoint")
    torch.save(state,filename)

def load_checkpoint(checkpoint,model):
    print("==> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loader(
        TRAIN_IMG_DIR,TRAIN_SEG_DIR,
        VAL_IMG_DIR,VAL_SEG_DIR,
        BATCH_SIZE,
        NUM_WORKERS,
        PIN_MEMORY,
        train_transform,
        val_transform
    ):
    raise NotImplementedError
    train_ds = brainDataset(image_dir=TRAIN_IMG_DIR, seg_dir=TRAIN_SEG_DIR, transform = train_transform)
    train_loader = DataLoader(
        train_ds,
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        pin_memory = PIN_MEMORY,
        shuffle = False,
    )

    val_ds brainDataset(image_dir=VAL_IMG_DIR, seg_dir=VAL_SEG_DIR, transform = train_transform)
    va_loader = DataLoader(
        val_ds,
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        pin_memory = PIN_MEMORY,
        shuffle = False,
    )

    return train_loader,val_loader

def check_accuracy(loader, model, device="cuda"):
    #change
    num_correct = 0
    num_pixels = 0
    intersection = 0
    union = 0
    dice_score = 0
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device).unsqueeze(1)  # Corrected variable name (labels to targets)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
    
            num_correct += (predicted == targets).sum().item()
            num_pixels += targets.numel()

            targets_flat = targets.view(targets.size(0), targets.size(1), -1)
            predicted_flat = predicted.view(predicted.size(0), predicted.size(1), -1)

            # Compute intersection and union for each class
            intersection += torch.sum(targets_flat * predicted_flat, dim=2)
            union += torch.sum(targets_flat, dim=2) + torch.sum(predicted_flat, dim=2)

    # Compute Dice coefficient for each class
    epsilon = 1e-7  # Smoothing factor to avoid division by zero
    dice_per_class = (2.0 * intersection + epsilon) / (union + epsilon)

    # Average Dice coefficient across all classes
    dice_score = torch.mean(dice_per_class, dim=1)
    accuracy = num_correct / num_pixels  # Corrected variable name (correct to num_correct)
    print(f"Accuracy: {100 * accuracy:.2f}%")
    model.train()


def save_prediction_as_imgs(loader, model, folder,device="cuda"):
    model.eval()
    for idx, (x,y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
        torchvision.utils.save_image(
            preds,f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1),f"{folder}")
    model.train()




