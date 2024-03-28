import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as functional

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # First half
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Second half
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Downsampling
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        # Upsampling
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = transforms.Resize(skip_connection.shape[2:])(x)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, predicted, target, smooth=1e-6):
        # Flatten predictions and targets
        predicted_flat = predicted.view(-1)
        print("predicted size",predicted.size())
        target_flat = target.view(-1)
        print("target size",target.size())

        intersection = (predicted_flat * target_flat).sum()
        union = predicted_flat.sum() + target_flat.sum()

        dice_coeff = (2.0 * intersection + smooth) / (union + smooth)
        
        # Dice Loss is 1 - Dice coefficient
        dice_loss = 1 - dice_coeff

        return dice_loss

class MulticlassDiceLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean', smooth=1e-6):
        super(MulticlassDiceLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: Batch Size x Number of Classes x H x W
        # targets: Batch Size x H x W (LongTensor, not one-hot)
        num_classes = inputs.shape[1]

        if self.weight is None:
            self.weight = torch.ones(num_classes, device=inputs.device)

        inputs = functional.softmax(inputs, dim=1)
        targets_one_hot = functional.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        inputs = inputs.view(inputs.shape[0], num_classes, -1)
        targets_one_hot = targets_one_hot.view(targets_one_hot.shape[0], num_classes, -1)

        intersection = (inputs * targets_one_hot).sum(-1)
        cardinality = inputs.sum(-1) + targets_one_hot.sum(-1)

        dice_score = 2. * intersection / (cardinality + self.smooth)
        #print("self.weight", self.weight)
        weighted_dice_score = dice_score * self.weight
        dice_loss = 1 - weighted_dice_score

        if self.reduction == 'mean':
            #print("mean", dice_loss.mean())
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss   
            

def test():
    x = torch.randn((3, 1, 160, 160))
    model = UNET(in_channels=1, out_channels=4)
    preds = model(x)

    print(preds.shape)

if __name__ == "__main__":
    test()
