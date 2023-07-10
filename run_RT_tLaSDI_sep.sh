#!/bin/bash
#BSUB -nnodes 1
#BSUB -q pdebug
#BSUB -W 120

problem="RT"
latent_dim="4"
net="ESP3"  # 'ESP3' (GFINNs) or 'ESP3_soft' (SPNN)
method="SEPERATE"
iterations="17"
max_epoch_SAE="17"
# loss weights  (Integrator loss weight: 1)
lambda_r_SAE="1e-1"  # reconstruction 1e-1
lambda_jac_SAE="1e-6"  # Jacobian 1e-6
lambda_dx="1e-4" # Consistency 1e-4
lambda_dz="1e-4" # Model approximation 1e-4

load_model="False"
load_iterations="20000" # total iterations before loaded

weight_decay_GFINNs="1e-7"
weight_decay_AE="0"


#Loading cuda will cause linking error
#module load cuda/11.4.1

source anaconda/bin/activate
conda activate opence-1.8.0

TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S") 
OUTPUT_PREFIX=${problem}_${latent_dim}_${net}_${method}_${lambda_r_SAE}_${lambda_jac_SAE}_${lambda_dx}_${lambda_dz}_${iterations}


python main_RT_tLaSDI_sep.py --latent_dim ${latent_dim} --net ${net} --iterations ${iterations} --max_epoch_SAE ${max_epoch_SAE} --lambda_r_SAE ${lambda_r_SAE} --lambda_jac_SAE ${lambda_jac_SAE} --lambda_dx ${lambda_dx} --lambda_dz ${lambda_dz} --load_model ${load_model} --weight_decay_GFINNs ${weight_decay_GFINNs} --weight_decay_AE ${weight_decay_AE} --load_iterations ${load_iterations} > ${OUTPUT_PREFIX}.log
