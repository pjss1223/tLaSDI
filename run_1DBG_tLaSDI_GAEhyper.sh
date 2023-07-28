#!/bin/bash
#BSUB -nnodes 1
#BSUB -q pbatch
#BSUB -W 720

problem="1DBG"
latent_dim="4"
net="ESP3"  # 'ESP3' (GFINNs) or 'ESP3_soft' (SPNN)

method="AEhyper"
epochs="18002"
# loss weights  (Integrator loss weight: 1)
lambda_r_SAE="1e-2"  # reconstruction 1e-1
lambda_jac_SAE="0"  # Jacobian 1e-6
lambda_dx="0" # Consistency 1e-4
lambda_dz="0" # Model approximation 1e-4
lam="1e-2"

load_model="False"
load_epochs="10001" # total iterations before loaded


#Loading cuda will cause linking error
#module load cuda/11.4.1

source anaconda/bin/activate
conda activate opence-1.8.0

TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S") 
OUTPUT_PREFIX=${problem}_${latent_dim}_${net}_${method}_${lambda_r_SAE}_${lambda_jac_SAE}_${lambda_dx}_${lambda_dz}_${lam}_${epochs}

# jsrun --nrs 4 --rs_per_host 4 --np 1 python main_1DBG_tLaSDI_greedy.py --latent_dim ${latent_dim} --net ${net} --iterations ${iterations} --lambda_r_SAE ${lambda_r_SAE} --lambda_jac_SAE ${lambda_jac_SAE} --lambda_dx ${lambda_dx} --lambda_dz ${lambda_dz} --load_model ${load_model} --load_iterations ${load_iterations} > ${OUTPUT_PREFIX}.log

python main_1DBG_tLaSDI_GAEhyper.py --latent_dim ${latent_dim} --net ${net} --epochs ${epochs} --lambda_r_SAE ${lambda_r_SAE} --lambda_jac_SAE ${lambda_jac_SAE} --lambda_dx ${lambda_dx} --lambda_dz ${lambda_dz} --load_model ${load_model} --load_epochs ${load_epochs} --lam ${lam} > ${OUTPUT_PREFIX}.log


# lrun -T4 python main_1DBG_tLaSDI_greedy.py --latent_dim ${latent_dim} --net ${net} --iterations ${iterations} --lambda_r_SAE ${lambda_r_SAE} --lambda_jac_SAE ${lambda_jac_SAE} --lambda_dx ${lambda_dx} --lambda_dz ${lambda_dz} --load_model ${load_model} --load_iterations ${load_iterations} > ${OUTPUT_PREFIX}.log