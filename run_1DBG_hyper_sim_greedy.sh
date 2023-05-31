#!/bin/bash
#BSUB -nnodes 1
#BSUB -q pdebug
#BSUB -W 60

source anaconda/bin/activate
module load cuda/11.4.1

python main_1DBG_hyper_sim_greedy.py
