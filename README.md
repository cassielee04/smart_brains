# DS598 DL4DS Final Project Smart Brains

## Introduction

The project focuses on enhancing the detection of brain tumors in medical image databases through automated segmentation and classification.
Leveraging deep convolutional neural networks with multiscale feature extraction, this approach aims to efficiently identify discriminant texture features of various tumor types.
By integrating multiscale receptive fields, it captures essential local and contextual information within the images. 
 The goal is to improve MRI's effectiveness as a non-invasive diagnostic tool, facilitating more accurate diagnosis, growth prediction, and treatment of brain tumors, underscoring the importance of automation in medical diagnostics.



## Dataset

For our project, we utilize the BraTS 2021 Dataset from the Brain Tumor Segmentation (BraTS) challenge, featuring a rich collection of multimodal MRI scans for brain tumor segmentation, focusing on gliomas. Gliomas, prevalent brain tumors, are classified into low-grade (LGG) and high-grade (HGG), the latter being more severe. Our data, sourced from a Kaggle dataset titled "BRATS2021 Training and Validation," includes labeled images for 341 subjects across 4 imaging modalities. We've structured our dataset into training and validation subsets, adhering to an 80:20 ratio. [BraTS 2021 Dataset](https://www.kaggle.com/datasets/kanisfatemashanta/brats2021-training-and-validation/data) 


## Model

The BraTS multimodal MRI dataset is a crucial resource for brain tumor research, offering a diverse collection of MRI scans in four key modalities: T1-weighted, post-contrast T1-weighted (T1Gd), T2-weighted, and T2-FLAIR volumes. Sourced from various institutions, this dataset reflects real-world clinical diversity, making it ideal for developing and testing brain tumor detection and segmentation algorithms. It includes detailed annotations for tumor regions, aiding in the accurate training of models to distinguish tumors from healthy tissue. As a benchmark in the research community, the BraTS dataset is instrumental in advancing diagnostic and treatment planning techniques in neuro-oncology.

## Set Up

To get started with the `smart_brains` project, follow these steps:

1. **Clone the repository**
   - Open a terminal and run the following command to clone the repository:
     ```
     git clone <repository-url>
     ```

2. **Navigate to the project directory**
   - Change directory to the cloned repository:
     ```
     cd smart_brains
     ```

3. **Submit the training job**
   - Use the `qsub` command to submit the training job to the queue. This command requests 4 cores with OpenMP, 1 GPU, and specifies project code `ds598`. It also sets files for output and error logging.
     ```
     qsub -pe omp 4 -P ds598 -l gpus=1 -o output.txt -e error.txt train.sh
     ```




