#!/bin/bash
#BSUB -nnodes 1
#BSUB -q pbatch
#BSUB -W 720

device="gpu"
seed="0"
problem="GC_SVD_NoAE" #GC_SVD or VC_SVD #Check batch size, lr!!!!!
latent_dim="4"
extraD_L="5" #2-12
extraD_M="5" #2-12
xi_scale="0.5773"
layers_sk="5"
width_sk="50"

net="ESP3_soft"  # 'ESP3' (GFINNs) or 'ESP3_soft' (SPNN) or 'ESP' (GFINNs case1) or 'ESP_soft' (SPNN case1)
iterations="200128" 
batch_size="15000"
# loss weights  (Integrator loss weight: 1)
lambda_r_SAE="1e-1"  # reconstruction 1e-1
lambda_jac_SAE="0"  # Jacobian 1e-4 or 1e-4(wo jac loss, with consistency),1e-6(wo jac loss, WO consistency)
lambda_dx="0" # Consistency 1e-4
lambda_dz="0" # Model approximation 1e-4 
lam="1e-2"   # degeneracy for SPNN 1e-2 or 1e-3

lr="1e-3"

load_model="False"
load_iterations="20000" # total iterations before loaded

activation="gelu"
activation_SAE="relu"

gamma_lr="0.99" # can be either multistep or periodics update

#Loading cuda will cause linking error
#module load cuda/11.4.1

source anaconda/bin/activate
conda activate opence-1.8.0

TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S") 
OUTPUT_PREFIX=TriuS${problem}_${latent_dim}_${net}_exDL${extraD_L}_exDM${extraD_M}_xs${xi_scale}_${lambda_r_SAE}_${lambda_jac_SAE}_${lambda_dx}_${lambda_dz}_${lam}_gamma${gamma_lr}_lr${lr}_${activation}_sknn${layers_sk}_${width_sk}_${iterations}_batch${batch_size}_seed${seed}


python main_GC_tLaSDI_NoAE_triuSigma.py --device ${device} --seed ${seed} --latent_dim ${latent_dim} --extraD_L ${extraD_L} --extraD_M ${extraD_M} --xi_scale ${xi_scale} --net ${net} --iterations ${iterations} --lambda_r_SAE ${lambda_r_SAE} --lambda_jac_SAE ${lambda_jac_SAE} --lambda_dx ${lambda_dx} --lambda_dz ${lambda_dz} --activation ${activation} --activation_SAE ${activation_SAE} --layers_sk ${layers_sk} --width_sk ${width_sk} --gamma_lr ${gamma_lr} --load_model ${load_model} --load_iterations ${load_iterations} --lam ${lam} --lr ${lr} --batch_size ${batch_size} > ${OUTPUT_PREFIX}.log
