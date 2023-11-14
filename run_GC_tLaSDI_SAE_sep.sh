#!/bin/bash
#BSUB -nnodes 1
#BSUB -q pbatch
#BSUB -W 720


problem="GC_SAE_sr" #test2 nov 8 11:44 pm
latent_dim="30"
extraD_L="29" #2-12
extraD_M="29" #2-12
# xi_scale="1e-2"
data_type="last"
device="gpu"

layers="5"
width="200"

AE_width1="200"
AE_width2="100"


net="ESP3_soft"  # 'ESP3' (GFINNs) or 'ESP3_soft' (SPNN)
method="SEPERATE"
iterations="100002"
max_epoch_SAE="100002"
# loss weights  (Integrator loss weight: 1)
lambda_int="1e3"
lambda_r_sparse="1e-4"
lambda_r_SAE="1"  # fixed 
lambda_jac_SAE="0"  # Jacobian 1e-2
lambda_dx="0" # Consistency 1e-7
lambda_dz="0" # Model approximation 1e-7 or 1e-6 for wo matrixJac, wo consi

activation="sin"
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

lr="1e-4"

load_model="False"

if [ "$load_model" == "False" ]; then
    load_iterations="0"
else
    load_iterations="70000"
fi

total_iteration=$(echo "$iterations+$load_iterations" | bc)

activation="sin"
activation_SAE="relu"

gamma_lr=".99"
weight_decay_GFINNs="1e-5"



source anaconda/bin/activate
conda activate opence-1.8.0

TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S") 
OUTPUT_PREFIX=${problem}_${method}_${latent_dim}_${net}_exDL${extraD_L}_exDM${extraD_M}_xs${xi_scale}_ly${layers}_wd${width}_Awd1${AE_width1}_Awd2${AE_width2}_${data_type}_${lambda_int}_${lambda_r_SAE}_${lambda_jac_SAE}_${lam}_lr${lr}_gam${gamma_lr}_${activation}_${activation_SAE}_${total_iteration}


# python main_VC_tLaSDI_sep.py --latent_dim ${latent_dim} --net ${net} --extraD_L ${extraD_L} --extraD_M ${extraD_M} --xi_scale ${xi_scale} --iterations ${iterations} --max_epoch_SAE ${max_epoch_SAE} --lambda_r_SAE ${lambda_r_SAE} --lambda_jac_SAE ${lambda_jac_SAE} --lambda_dx ${lambda_dx} --lambda_dz ${lambda_dz} --activation ${activation} --activation_SAE ${activation_SAE} --load_model ${load_model} --load_iterations ${load_iterations} --lam ${lam} > ${OUTPUT_PREFIX}.log


python main_GC_tLaSDI_SAE_sep.py --device ${device} --data_type ${data_type} --latent_dim ${latent_dim} --extraD_L ${extraD_L} --extraD_M ${extraD_M} --xi_scale ${xi_scale} --layers ${layers} --width ${width} --AE_width1 ${AE_width1} --AE_width2 ${AE_width2} --net ${net} --iterations ${iterations} --max_epoch_SAE ${max_epoch_SAE} --lambda_r_SAE ${lambda_r_SAE} --lambda_jac_SAE ${lambda_jac_SAE} --lambda_dx ${lambda_dx} --lambda_dz ${lambda_dz} --lambda_int ${lambda_int} --lambda_r_sparse ${lambda_r_sparse} --lr ${lr} --gamma_lr ${gamma_lr} --activation ${activation} --activation_SAE ${activation_SAE} --load_model ${load_model} --load_iterations ${load_iterations} --lam ${lam} > ${OUTPUT_PREFIX}.log