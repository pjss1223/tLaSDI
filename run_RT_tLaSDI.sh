#!/bin/bash
#BSUB -nnodes 1
#BSUB -q pdebug
#BSUB -W 60

#Loading cuda will cause linking error
#module load cuda/11.4.1

source anaconda/bin/activate
conda activate opence-1.8.0

python main_RT_tLaSDI.py
