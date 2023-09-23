#!/bin/bash
#BSUB -nnodes 1
#BSUB -q pdebug
#BSUB -W 120

problem="RT"
latent_dim="10"
latent_dim_q="8"
latent_dim_v="8"
latent_dim_sigma="8"
extraD_L="7" #2-12
extraD_M="7" #2-12
net="ESP3"  # 'ESP3' (GFINNs) or 'ESP3_soft' (SPNN)
method="SEPERATE"
iterations="17"
max_epoch_SAE="17"
# loss weights  (Integrator loss weight: 1)
lambda_r_SAE="1e-1"  # reconstruction 1e-1
lambda_jac_SAE="1e-6"  # Jacobian 1e-6
lambda_dx="1e-4" # Consistency 1e-4
lambda_dz="1e-4" # Model approximation 1e-4
lam="1e-2"   # degeneracy for SPNN 1e-2 or 1e-3


activation="tanh"
activation_SAE="relu"

load_model="False"
load_iterations="20000" # total iterations before loaded

weight_decay_GFINNs="1e-7"
weight_decay_AE="0"

lr="1e-4"


#Loading cuda will cause linking error
#module load cuda/11.4.1

source anaconda/bin/activate
conda activate opence-1.8.0

TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S") 
OUTPUT_PREFIX=${problem}_${latent_dim_q}_${latent_dim_v}_${latent_dim_sigma}_exDL${extraD_L}_exDM${extraD_M}_${net}_${method}_${activation}_${lambda_r_SAE}_${lambda_jac_SAE}_${lambda_dx}_${lambda_dz}_${lam}_${lr}_${iterations}


# python main_RT_tLaSDI_sep.py --latent_dim ${latent_dim} --net ${net} --iterations ${iterations} --max_epoch_SAE ${max_epoch_SAE} --lambda_r_SAE ${lambda_r_SAE} --lambda_jac_SAE ${lambda_jac_SAE} --lambda_dx ${lambda_dx} --lambda_dz ${lambda_dz} --load_model ${load_model} --weight_decay_GFINNs ${weight_decay_GFINNs} --weight_decay_AE ${weight_decay_AE} --load_iterations ${load_iterations} > ${OUTPUT_PREFIX}.log


python main_RT_tLaSDI_sep.py --lr ${lr} --latent_dim_q ${latent_dim_q} --latent_dim_v ${latent_dim_v} --latent_dim_sigma ${latent_dim_sigma} --extraD_L ${extraD_L} --extraD_M ${extraD_M} --net ${net} --iterations ${iterations} --lambda_r_SAE ${lambda_r_SAE} --lambda_jac_SAE ${lambda_jac_SAE} --lambda_dx ${lambda_dx} --lambda_dz ${lambda_dz} --load_model ${load_model} --load_iterations ${load_iterations} --lam ${lam} --activation ${activation} --activation_SAE ${activation_SAE} --weight_decay_GFINNs ${weight_decay_GFINNs} --weight_decay_AE ${weight_decay_AE} > ${OUTPUT_PREFIX}.log