#!/bin/bash
#BSUB -nnodes 1
#BSUB -q pbatch
#BSUB -W 360

problem="RT"
latent_dim="6"
latent_dim_q="4"
latent_dim_v="4"
latent_dim_sigma="4"
net="ESP3"  # 'ESP3' (GFINNs) or 'ESP3_soft' (SPNN)
iterations="40000"
# loss weights  (Integrator loss weight: 1)
lambda_r_SAE="1e-3"  # reconstruction 1e-3
lambda_jac_SAE="0"  # Jacobian 1e-6
lambda_dx="0" # Consistency 1e-4
lambda_dz="0" # Model approximation 1e-7
lam="1e-2"   # degeneracy for SPNN 1e-2 or 1e-3

lr="1e-4"
seed="20"

load_model="False"
load_iterations="30000" # total iterations before loaded

#Loading cuda will cause linking error
#module load cuda/11.4.1

source anaconda/bin/activate
conda activate opence-1.8.0

TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S") 
OUTPUT_PREFIX=${problem}_${latent_dim_q}_${latent_dim_v}_${latent_dim_sigma}_${net}_${lambda_r_SAE}_${lambda_jac_SAE}_${lambda_dx}_${lambda_dz}_${lam}_${lr}_${iterations}_${seed}

python main_RT_tLaSDI.py --seed ${seed} --lr ${lr} --latent_dim_q ${latent_dim_q} --latent_dim_v ${latent_dim_v} --latent_dim_sigma ${latent_dim_sigma} --net ${net} --iterations ${iterations} --lambda_r_SAE ${lambda_r_SAE} --lambda_jac_SAE ${lambda_jac_SAE} --lambda_dx ${lambda_dx} --lambda_dz ${lambda_dz} --load_model ${load_model} --load_iterations ${load_iterations} --lam ${lam} > ${OUTPUT_PREFIX}.log
