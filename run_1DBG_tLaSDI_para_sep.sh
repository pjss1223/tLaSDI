#!/bin/bash
#BSUB -nnodes 1
#BSUB -q pdebug
#BSUB -W 120


device="gpu"
problem="1DBG"
latent_dim="4"

extraD_L="3" #2-12
extraD_M="3" #2-12

batch_size="60"

net="ESP3"  # 'ESP3' (GFINNs) or 'ESP3_soft' (SPNN)

method="para_sep"

epochs="20001"
# loss weights  (Integrator loss weight: 1)
lambda_r_SAE="1e-1"  # reconstruction 1e-1
lambda_jac_SAE="1e-9"  # Jacobian 1e-6
lambda_dx="0" # Consistency 1e-4
lambda_dz="0" # Model approximation 1e-4
lam="0"


load_model="False"
load_epochs="10001" # total iterations before loaded


if [ "$net" == "ESP3" ]; then
    xi_scale=$(echo "scale=4; 1/sqrt($latent_dim-1)" | bc)
else
    xi_scale="0"
fi
#Loading cuda will cause linking error
#module load cuda/11.4.1

source anaconda/bin/activate
conda activate opence-1.8.0

TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S") 
OUTPUT_PREFIX=${problem}_${method}_${latent_dim}_${net}_exDL${extraD_L}_exDM${extraD_M}_${lambda_r_SAE}_${lambda_jac_SAE}_${lambda_dx}_${lambda_dz}_${lam}_${epochs}

# jsrun --nrs 4 --rs_per_host 4 --np 1 python main_1DBG_tLaSDI_greedy.py --latent_dim ${latent_dim} --net ${net} --iterations ${iterations} --lambda_r_SAE ${lambda_r_SAE} --lambda_jac_SAE ${lambda_jac_SAE} --lambda_dx ${lambda_dx} --lambda_dz ${lambda_dz} --load_model ${load_model} --load_iterations ${load_iterations} > ${OUTPUT_PREFIX}.log

python main_1DBG_tLaSDI_para_sep.py --device ${device} --max_epoch_SAE ${epochs} --latent_dim ${latent_dim} --extraD_L ${extraD_L} --extraD_M ${extraD_M} --xi_scale ${xi_scale} --net ${net} --lambda_r_SAE ${lambda_r_SAE} --lambda_jac_SAE ${lambda_jac_SAE} --lambda_dx ${lambda_dx} --lambda_dz ${lambda_dz} --batch_size ${batch_size} --batch_size_AE ${batch_size} --load_model ${load_model} --load_epochs ${load_epochs} --lam ${lam} --epochs ${epochs} > ${OUTPUT_PREFIX}.log
# --epochs ${epochs} 

# lrun -T4 python main_1DBG_tLaSDI_greedy.py --latent_dim ${latent_dim} --net ${net} --iterations ${iterations} --lambda_r_SAE ${lambda_r_SAE} --lambda_jac_SAE ${lambda_jac_SAE} --lambda_dx ${lambda_dx} --lambda_dz ${lambda_dz} --load_model ${load_model} --load_iterations ${load_iterations} > ${OUTPUT_PREFIX}.log