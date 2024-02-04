#!/bin/bash
#BSUB -nnodes 1
#BSUB -q pbatch
#BSUB -W 720

device="gpu"

problem="GC"
latent_dim="30"
extraD_L="29" #2-12
extraD_M="29" #2-12
# xi_scale=".3333" #"0.3333" 0.3780  0.4472  0.5774 1
data_type="last"

layers="5"
width="200"

AE_width1="200"
AE_width2="100"

net="GFINNs"  # (GFINNs) or (SPNN)

iterations="100001"
# loss weights  (Integrator loss weight: 1)
lambda_r_AE="1e-1"  # reconstruction 1e-1
lambda_jac_AE="1e-2"  # Jacobian 1e-4 or 1e-4(wo jac loss, with consistency),1e-6(wo jac loss, WO consistency)
lambda_dx="1e-7" # Consistency 1e-4
lambda_dz="1e-7" # Model approximation 1e-4 

if [ "$net" == "SPNN" ]; then
    lam="1e-1"
    extraD_L="0"
    extraD_M="0"
else
    lam="0" # degeneracy for SPNN 1e-2 or 1e-3

fi

if [ "$net" == "GFINNs" ]; then
    xi_scale=$(echo "scale=4; 1/sqrt($latent_dim-1)" | bc)
else
    xi_scale="0"
fi

lr="1e-4"

load_model="False"

if [ "$load_model" == "False" ]; then
    load_iterations="0"
else
    load_iterations="100000"
fi

total_iteration=$(echo "$iterations+$load_iterations" | bc)

activation="sin"
activation_AE="relu"

#Loading cuda will cause linking error
#module load cuda/11.4.1
gamma_lr="1"
weight_decay_AE="0"
weight_decay_GFINNs="1e-8"

source anaconda/bin/activate
conda activate opence-1.8.0

TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S") 
OUTPUT_PREFIX=${problem}_${latent_dim}_${net}_exDL${extraD_L}_exDM${extraD_M}_xs${xi_scale}_ly${layers}_wd${width}_Awd1${AE_width1}_Awd2${AE_width2}_${data_type}_${lambda_r_AE}_${lambda_jac_AE}_${lambda_dx}_${lambda_dz}_${lam}_lr${lr}_gam${gamma_lr}_WD${weight_decay_GFINNs}_${activation}_${activation_AE}_${total_iteration}


python main_GC_tLaSDI.py --device ${device} --data_type ${data_type} --latent_dim ${latent_dim} --extraD_L ${extraD_L} --extraD_M ${extraD_M} --xi_scale ${xi_scale} --layers ${layers} --width ${width} --AE_width1 ${AE_width1} --AE_width2 ${AE_width2} --net ${net} --iterations ${iterations} --lambda_r_AE ${lambda_r_AE} --lambda_jac_AE ${lambda_jac_AE} --lambda_dx ${lambda_dx} --lambda_dz ${lambda_dz} --lr ${lr} --gamma_lr ${gamma_lr} --weight_decay_GFINNs ${weight_decay_GFINNs} --activation ${activation} --activation_AE ${activation_AE} --load_model ${load_model} --load_iterations ${load_iterations} --lam ${lam} > ${OUTPUT_PREFIX}.log
