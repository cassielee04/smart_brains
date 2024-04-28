#!/bin/bash -l

# Set SCC project
#$ -P ds598
#$ -l h_rt=3:00:00 
#$ -m beas
#$ -M xhu07@bu.edu


module load miniconda
conda activate dl4ds2 # activate your conda environment

export MODEL_PATH=/projectnb/ds598/projects/smart_brains

python scripts/parse_train.py --num_epochs "50" --learn_rate "1e-4" --modal_type [t1ce,flair] --with_transform true --exp_name "MyExperiment"
#python scripts/metrics2.py --load_model_names "all"

#python scripts/parse_train.py --num_epochs "200" --learn_rate "1e-4" --modal_type [t1ce] --exp_name "Selena"
# python scripts/parse_train.py --num_epochs "200" --learn_rate "1e-4" --modal_type [t1ce] --with_transform true --exp_name "SelenaDataAugment"

# python scripts/parse_train.py --num_epochs "200" --learn_rate "1e-4" --modal_type [flair] --with_transform true --exp_name "SelenaDataAugment"
# python scripts/parse_train.py --num_epochs "200" --learn_rate "1e-4" --modal_type [flair] --exp_name "Selena"

# python scripts/parse_train.py --num_epochs "200" --learn_rate "1e-4" --modal_type [t1ce,flair] --exp_name "Selena"
# python scripts/parse_train.py --num_epochs "200" --learn_rate "1e-4" --modal_type [t1ce,flair] --with_transform true --exp_name "SelenaDataAugment"



## Notes for job submission:
## To submit the job to SCC, run the following command in the terminal:
## qsub -pe omp 4 -P ds598 -l gpus=1 -o output.txt -e error.txt train.sh
## qsub -pe omp 4 -P ds598 -l gpus=1 train.sh
## After submission, you can check the status of your job with:
## qstat -u $USER