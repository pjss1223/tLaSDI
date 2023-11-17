#!/bin/bash
#BSUB -nnodes 1
#BSUB -q pdebug
#BSUB -W 120

seed="0"
problem="VC_SAE_fn3" #test2 nov 8 11:44 pm
latent_dim="8"
extraD_L="9" #2-12
extraD_M="9" #2-12
# xi_scale="1e-2"
data_type="last"
device="gpu"

layers="5" #4 20 works well
width="24"

AE_width1="160"  #80 40 works well
AE_width2="160"


net="ESP3"  # 'ESP3' (GFINNs) or 'ESP3_soft' (SPNN)
method="SEPERATE"
iterations="40008"
max_epoch_SAE="40008"
# loss weights  (Integrator loss weight: 1)
lambda_int="1e3"
lambda_r_sparse="1e-4"
lambda_r_SAE="1"  # fixed
lambda_jac_SAE="0"  # Jacobian 1e-2
lambda_dx="0" # Consistency 1e-7
lambda_dz="0" # Model approximation 1e-7 or 1e-6 for wo matrixJac, wo consi

activation="tanh"
activation_SAE="relu"

load_model="False"
load_iterations="20000" # total iterations before loaded

if [ "$net" == "ESP3_soft" ]; then
    lam="1" #fixed
    extraD_L="0"
    extraD_M="0"
else
    lam="0" # degeneracy for SPNN 1e-2 or 1e-3

fi

if [ "$net" == "ESP3" ]; then
    xi_scale=$(echo "scale=4; 1/sqrt($latent_dim-1)" | bc)
else
    xi_scale="0"
fi

lr="1e-5"

load_model="False"

if [ "$load_model" == "False" ]; then
    load_iterations="0"
else
    load_iterations="70000"
fi

total_iteration=$(echo "$iterations+$load_iterations" | bc)


gamma_lr="1"

weight_decay_AE="0"
weight_decay_GFINNs="1e-5"
source anaconda/bin/activate
conda activate opence-1.8.0

TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S") 
OUTPUT_PREFIX=${problem}_${method}_${latent_dim}_${net}_exDL${extraD_L}_exDM${extraD_M}_ly${layers}_wd${width}_Awd1${AE_width1}_Awd2${AE_width2}_${data_type}_${lambda_r_SAE}_${lambda_jac_SAE}_${lambda_dx}_${lambda_dz}_${lam}_lr${lr}_gam${gamma_lr}_WDGF${weight_decay_GFINNs}_${activation}_${activation_SAE}_${seed}_${total_iteration}


# python main_VC_tLaSDI_sep.py --latent_dim ${latent_dim} --net ${net} --extraD_L ${extraD_L} --extraD_M ${extraD_M} --xi_scale ${xi_scale} --iterations ${iterations} --max_epoch_SAE ${max_epoch_SAE} --lambda_r_SAE ${lambda_r_SAE} --lambda_jac_SAE ${lambda_jac_SAE} --lambda_dx ${lambda_dx} --lambda_dz ${lambda_dz} --activation ${activation} --activation_SAE ${activation_SAE} --load_model ${load_model} --load_iterations ${load_iterations} --lam ${lam} > ${OUTPUT_PREFIX}.log


python main_VC_tLaSDI_SAE_sep.py --seed ${seed} --device ${device} --data_type ${data_type} --latent_dim ${latent_dim} --extraD_L ${extraD_L} --extraD_M ${extraD_M} --xi_scale ${xi_scale} --layers ${layers} --width ${width} --AE_width1 ${AE_width1} --AE_width2 ${AE_width2} --net ${net} --iterations ${iterations} --max_epoch_SAE ${max_epoch_SAE} --lambda_r_SAE ${lambda_r_SAE} --lambda_jac_SAE ${lambda_jac_SAE} --lambda_dx ${lambda_dx} --lambda_dz ${lambda_dz} --lambda_int ${lambda_int} --lambda_r_sparse ${lambda_r_sparse} --lr ${lr} --gamma_lr ${gamma_lr} --weight_decay_AE ${weight_decay_AE} --weight_decay_GFINNs ${weight_decay_GFINNs} --activation ${activation} --activation_SAE ${activation_SAE} --load_model ${load_model} --load_iterations ${load_iterations} --lam ${lam} > ${OUTPUT_PREFIX}.log