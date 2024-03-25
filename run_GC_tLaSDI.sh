#!/bin/bash
#BSUB -nnodes 1
#BSUB -q pdebug
#BSUB -W 120

seed="0" # 0, 7001, 2494, 3782, 3420, 8328, 1383, 6692, 9912, 7517
device="gpu"

order="2"

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

net="SPNN"  # (GFINNs) or (SPNN)

iterations="85017" #15010
# loss weights  (Integrator loss weight: 1)
lambda_r_AE="1e-1"  # reconstruction 1e-1
lambda_jac_AE="1e-2" 
lambda_dx="1e-7" 
lambda_dz="1e-7"

if [ "$net" == "SPNN" ]; then
    lam="1e-3"  # degeneracy for SPNN 0 or 1e-2 or 1e-3
    extraD_L="0"
    extraD_M="0"
else
    lam="0" 

fi

if [ "$net" == "GFINNs" ]; then
    xi_scale=$(echo "scale=4; 1/sqrt($latent_dim-1)" | bc)
else
    xi_scale="0"
fi

lr="1e-4"

load_model="True"

if [ "$load_model" == "False" ]; then
    load_iterations="0"
else
    load_iterations="85017"
fi

total_iteration=$(echo "$iterations+$load_iterations" | bc)

activation="sin"
activation_AE="relu"

#Loading cuda will cause linking error
#module load cuda/11.4.1

lr_scheduler_type="StepLR" # StepLR or MultiStepLR

if [ "$lr_scheduler_type" == "StepLR" ]; then
    gamma_lr=".99"
    miles_lr="1000"
else
    gamma_lr="1e-1"
    miles_lr="40000"
fi

weight_decay_AE="0"
weight_decay_GFINNs="0"

source anaconda/bin/activate
conda activate opence-1.8.0

TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S") 
OUTPUT_PREFIX=${problem}_${latent_dim}_${net}_od${order}_exDL${extraD_L}_exDM${extraD_M}_xs${xi_scale}_ly${layers}_wd${width}_Awd1${AE_width1}_Awd2${AE_width2}_${data_type}_${lambda_r_AE}_${lambda_jac_AE}_${lambda_dx}_${lambda_dz}_${lam}_lr${lr}_gam${gamma_lr}_mil${miles_lr}_WD${weight_decay_GFINNs}_${activation}_${activation_AE}_${seed}_${total_iteration}


python main_GC_tLaSDI.py --seed ${seed} --order ${order} --device ${device} --data_type ${data_type} --latent_dim ${latent_dim} --extraD_L ${extraD_L} --extraD_M ${extraD_M} --xi_scale ${xi_scale} --layers ${layers} --width ${width} --AE_width1 ${AE_width1} --AE_width2 ${AE_width2} --net ${net} --iterations ${iterations} --lambda_r_AE ${lambda_r_AE} --lambda_jac_AE ${lambda_jac_AE} --lambda_dx ${lambda_dx} --lambda_dz ${lambda_dz} --lr ${lr} --lr_scheduler_type ${lr_scheduler_type} --gamma_lr ${gamma_lr} --miles_lr ${miles_lr} --weight_decay_GFINNs ${weight_decay_GFINNs} --activation ${activation} --activation_AE ${activation_AE} --load_model ${load_model} --load_iterations ${load_iterations} --lam ${lam} > ${OUTPUT_PREFIX}.log
