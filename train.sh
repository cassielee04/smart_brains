#!/bin/bash -l

module load miniconda
conda activate dl4ds2 # activate your conda environment
##module load python3
##module load pytorch 

##pip install nibabel
python3 train.py 

##--bs "$bs" --lr "$lr" --epochs "$epoch" --model_name resnet18

## Notes for job submission:
## To submit the job to SCC, run the following command in the terminal:
## qsub -pe omp 4 -P ds598 -l gpus=1 -o output.txt -e error.txt train.sh

## After submission, you can check the status of your job with:
## qstat -u $USER