#!/bin/bash -l

# Set SCC project
#$ -P ds598

module load miniconda
conda activate dl4ds # activate your conda environment

export PYTHONPATH="/projectnb/ds598/projects/smart_brains/scripts:$PYTHONPATH"
python scripts/train.py 



## Notes for job submission:
## To submit the job to SCC, run the following command in the terminal:
## qsub -pe omp 4 -P ds598 -l gpus=1 -o output.txt -e error.txt train2.sh




