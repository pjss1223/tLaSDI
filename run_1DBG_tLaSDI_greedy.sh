#!/bin/bash
#BSUB -nnodes 1
#BSUB -q pbatch
#BSUB -W 120

latent_dim="10"
net="ESP3"  # 'ESP3' (GFINNs) or 'ESP3_soft' (SPNN)
iterations="10000"
# loss weights  (Integrator loss weight: 1)
lambda_r_SAE="1e-1"  # reconstruction
lambda_jac_SAE="1e-6"  # Jacobian
lambda_dx="1e-4" # Consistency
lambda_dz="1e-4" # Model approximation

#Loading cuda will cause linking error
#module load cuda/11.4.1

source anaconda/bin/activate
conda activate opence-1.8.0

TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S") 
OUTPUT_PREFIX=${latent_dim}_${net}_${lambda_r_SAE}_${lambda_jac_SAE}_${lambda_dx}_${lambda_dz}

python main_1DBG_tLaSDI_greedy.py --latent_dim ${latent_dim} --net ${net} --iterations ${iterations} --lambda_r_SAE ${lambda_r_SAE} --lambda_jac_SAE ${lambda_jac_SAE} --lambda_dx ${lambda_dx} --lambda_dz ${lambda_dz} > ${OUTPUT_PREFIX}.log
