#!/bin/bash
#BSUB -nnodes 1
#BSUB -q pdebug
#BSUB -W 180

problem="VC"
latent_dim="10"
extraD_L="9" #2-12
extraD_M="9" #2-12
xi_scale="1e-2"

net="ESP3"  # 'ESP3' (GFINNs) or 'ESP3_soft' (SPNN)
method="SEPERATE"
iterations="40122"
max_epoch_SAE="40122"
# loss weights  (Integrator loss weight: 1)
lambda_r_SAE="1e-1"  # reconstruction 1e-1
lambda_jac_SAE="0"  # Jacobian 1e-6
lambda_dx="1e-4" # Consistency 1e-4
lambda_dz="1e-4" # Model approximation 1e-4 or 1e-6 for wo matrixJac, wo consi
lam="0"

activation="tanh"
activation_SAE="relu"

load_model="False"
load_iterations="20000" # total iterations before loaded



#Loading cuda will cause linking error
#module load cuda/11.4.1

source anaconda/bin/activate
conda activate opence-1.8.0

TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S") 
OUTPUT_PREFIX=${problem}_${latent_dim}_${net}_exDL${extraD_L}_exDM${extraD_M}_xs${xi_scale}_${method}_${lambda_r_SAE}_${lambda_jac_SAE}_${lambda_dx}_${lambda_dz}_${lam}_${iterations}


python main_VC_tLaSDI_sep.py --latent_dim ${latent_dim} --net ${net} --extraD_L ${extraD_L} --extraD_M ${extraD_M} --xi_scale ${xi_scale} --iterations ${iterations} --max_epoch_SAE ${max_epoch_SAE} --lambda_r_SAE ${lambda_r_SAE} --lambda_jac_SAE ${lambda_jac_SAE} --lambda_dx ${lambda_dx} --lambda_dz ${lambda_dz} --activation ${activation} --activation_SAE ${activation_SAE} --load_model ${load_model} --load_iterations ${load_iterations} --lam ${lam} > ${OUTPUT_PREFIX}.log
