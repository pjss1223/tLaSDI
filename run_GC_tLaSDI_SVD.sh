#!/bin/bash
#BSUB -nnodes 1
#BSUB -q pdebug
#BSUB -W 120

problem="GC_SVD_Q_ortho" #GC_SVD or VC_SVD or GC_SVD_concat #Check batch size, lr!!!!!

seed="0"
latent_dim="4"
extraD_L="3" #2-12
extraD_M="3" #2-12
net="ESP3_soft"  # 'ESP3' (GFINNs) or 'ESP3_soft' (SPNN)
iterations="50203"
# loss weights  (Integrator loss weight: 1)
lambda_r_SAE="1e-1"  # reconstruction 1e-1
lambda_jac_SAE="0"  # Jacobian 1e-4 or 1e-4(wo jac loss, with consistency),1e-6(wo jac loss, WO consistency)
lambda_dx="1e-7" # Consistency 1e-4 1e-7
lambda_dz="1e-7" # Model approximation 1e-4 1e-7 
lam="1e-2"   # degeneracy for SPNN 1e-2 or 1e-3

load_model="False"
load_iterations="20000" # total iterations before loaded



#Loading cuda will cause linking error
#module load cuda/11.4.1

source anaconda/bin/activate
conda activate opence-1.8.0

TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S") 
OUTPUT_PREFIX=${problem}_${latent_dim}_${net}_exDL${extraD_L}_exDM${extraD_M}_${lambda_r_SAE}_${lambda_jac_SAE}_${lambda_dx}_${lambda_dz}_${lam}_${iterations}_${seed}


python main_GC_tLaSDI_SVD.py --seed ${seed} --latent_dim ${latent_dim} --extraD_L ${extraD_L} --extraD_M ${extraD_M} --net ${net} --iterations ${iterations} --lambda_r_SAE ${lambda_r_SAE} --lambda_jac_SAE ${lambda_jac_SAE} --lambda_dx ${lambda_dx} --lambda_dz ${lambda_dz} --load_model ${load_model} --load_iterations ${load_iterations} --lam ${lam} > ${OUTPUT_PREFIX}.log