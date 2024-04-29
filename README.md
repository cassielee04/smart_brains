# smart_brains
![python3.10](https://img.shields.io/badge/python-3.10-blue.svg)


## Multimodal U-Net for Tumor segmentation

We implemented U-Net for the BraTS challenge using data downloaded from https://www.kaggle.com/datasets/kanisfatemashanta/brats2021-training-and-validation. Since the validation folder in this dataset does not include ground truth label data, we randomly selected 10 images from the training folder for testing purposes. We then applied an 8:2 ratio split for training and validation.

## Model Structure

![caption_1](Model_structure.png)

## Data Structure
### Training Data Structure
```command
MICCAI_FeTS2021_TrainingData/
├── FeTS21_Training_001
│  ├── FeTS21_Training_001_flair.nii
│  └── FeTS21_Training_001_seg.nii
│  └── FeTS21_Training_001_t1ce.nii
├── FeTS21_Training_002
│  ├── FeTS21_Training_002_flair.nii
│  └── FeTS21_Training_002_seg.nii
│  └── FeTS21_Training_002_t1ce.nii
...
```
### Testing Data Structure
```command
MICCAI_FeTS2021_TestingData/
├── FeTS21_Testing_001
│  ├── FeTS21_Testing_001_flair.nii
│  └── FeTS21_Testing_001_t1ce.nii
├── FeTS21_Testing_002
│  ├── FeTS21_Testing_002_flair.nii
│  └── FeTS21_Testing_002_t1ce.nii
...
```

 

## U-Net Model Training Script
This script is designed for training the U-Net model on medical imaging data, particularly for tasks like segmentation of brain tumors. It leverages PyTorch for deep learning functionalities and supports various data augmentations and loss functions to optimize model performance.

### Features

- Training U-Net with customizable epochs, learning rates, and batch sizes.
- Support for multiple loss functions and optimizers.
- Option to load a pre-trained model or start training from scratch.
- Data augmentation capabilities for enhanced model robustness.
- Integration with albumentations for image transformations.

### Prerequisites
Before running the script, ensure all libraries in the **'requiremetns.txt'** are installed:
```command
pip install -r requirements.txt
```

### Setup: specify your working directory
1. Clone this repository or download the script to your local machine.
2. Set the environment variable MODEL_PATH to specify the default path to your models. If not set, the script uses ./default_model_path as the fallback directory.
   
```command
export MODEL_PATH=./default_model_path
```

### Usage

Run the script from the command line by providing the necessary arguments. Below are the supported command-line arguments along with examples on how to use them.

#### Command-Line Arguments

- `--num_epochs`: Number of epochs to train the model (default: 20).
- `--learn_rate`: Learning rate for the optimizer (default: 1e-4).
- `--batch_size`: Batch size for training (default: 16).
- `--loss_type`: Specifies the loss function to use; options include 'CrossEntropy', 'Dice', 'MultiDice' (default: 'MultiDice').
- `--modal_type`: Type of imaging modality to use (default: 'T1CE').
- `--exp_name`: Name for the experiment (default: 'MyExperiment').
- `--optimizer_name`: Name of the optimizer to use (default: 'Adam').
- `--load_model_name`: Specifies the model name to load if resuming from a previous state.
- `--load_last_model`: Flag to load the last saved model; use this to resume training.
- `--with_transform`: Enable or disable data transformations during training (default: False).
### Examples

**Train a new model**
```bash
python parse_train.py --num_epochs 2 --learn_rate 1e-4 --modal_type t1ce,flair --exp_name "temp"
```
**Training with transformations**
```bash
python parse_train.py --num_epochs 2 --learn_rate 1e-4 --modal_type [t1ce,flair] --with_transform true --exp_name "MyExperiment"
```


## Model Analysis
This step will generate metrics--accuracy, precision, recall, specificity, F1, and Dice--to summarize models' performance and save outputs in a csv file.
### Usage
```command
python scripts/metrics.py --model_folder <path_to_models_directory> [--load_model_names <model1 model2 ...>]
```
#### Arguments
Run the script from the command line, providing either one of following argument:
- **`--model_folder`** : Specifies the path to the folder containing all models. Metrics will be computed for all models in this directory unless specific models are named using the `--load_model_names` option.
- **`--load_model_names`** : A space-separated list of specific model names to load. This parameter allows you to target specific models within the specified directory for processing.

### Examples
**1. Using the default settings:**
```bash
python scripts/metrics.py--model_folder path/to/model/RESULTS
```
This command runs the script using all models in the specified folder without specifying any particular models to load.

**2. Specifying models to load:**
```command
python scripts/metrics.py --load_model_names "myModel1_flair_opAdam_lr0.0001_bs16_epoch0_200" "myModel2_t1ce_opAdam_lr0.0001_bs16_epoch0_200" "myModel3_t1ce-flair_opAdam_lr0.0001_bs16_epoch0_200"
```


## Model Predicting
This script is designed for applying a user defined U-Net model on medical imaging data folder that does not have a ground truth segmentations.

### Usage
To run predictions using the pre-trained models, use the following command. Replace **`myModel1_flair_opAdam_lr0.0001_bs16_epoch0_200`** with the model name you want to load:
```command
export MODEL_PATH=./default_model_path
python parse_predict.py --load_model_name myModel1_flair_opAdam_lr0.0001_bs16_epoch0_200
```
You may also change the testing folder path inside the **`parse_predict.py`** to the one you want. The default setting is:

```command
testing_folder_path = f"{MAIN_PATH}/dataset/MICCAI_FeTS2021_TestingData"
```

### Arguments
- **`--load_model_name`**: Specifies the name of the model file to load for making predictions.











