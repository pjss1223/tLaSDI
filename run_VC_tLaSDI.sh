#!/bin/bash
#BSUB -nnodes 1
#BSUB -q pdebug
#BSUB -W 120

device="gpu"

problem="VC"
latent_dim="10"
extraD_L="10" #2-12
extraD_M="10" #2-12
# xi_scale=".3333" #"0.3333" 0.3780  0.4472  0.5774 1
data_type="last"

layers="4"
width="20"

AE_width1="80"
AE_width2="40"

net="ESP3"  # 'ESP3' (GFINNs) or 'ESP3_soft' (SPNN)

iterations="20000"
# loss weights  (Integrator loss weight: 1)
lambda_r_SAE="1e-1"  # reconstruction 1e-1
lambda_jac_SAE="0"  # Jacobian 1e-4 or 1e-4(wo jac loss, with consistency),1e-6(wo jac loss, WO consistency)
lambda_dx="0" # Consistency 1e-4
lambda_dz="0" # Model approximation 1e-4 

if [ "$net" == "ESP3_soft" ]; then
    lam="0"
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

load_model="True"

if [ "$load_model" == "False" ]; then
    load_iterations="0"
else
    load_iterations="20000"
fi

total_iteration=$(echo "$iterations+$load_iterations" | bc)

activation="tanh"
activation_SAE="relu"

#Loading cuda will cause linking error
#module load cuda/11.4.1
gamma_lr="0.99"

source anaconda/bin/activate
conda activate opence-1.8.0

TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S") 
OUTPUT_PREFIX=${problem}_${latent_dim}_${net}_exDL${extraD_L}_exDM${extraD_M}_xs${xi_scale}_ly${layers}_wd${width}_Awd1${AE_width1}_Awd2${AE_width2}_${data_type}_${lambda_r_SAE}_${lambda_jac_SAE}_${lambda_dx}_${lambda_dz}_${lam}_lr${lr}_gam${gamma_lr}_${activation}_${activation_SAE}_${total_iteration}


python main_VC_tLaSDI.py --device ${device} --data_type ${data_type} --latent_dim ${latent_dim} --extraD_L ${extraD_L} --extraD_M ${extraD_M} --xi_scale ${xi_scale} --layers ${layers} --width ${width} --AE_width1 ${AE_width1} --AE_width2 ${AE_width2} --net ${net} --iterations ${iterations} --lambda_r_SAE ${lambda_r_SAE} --lambda_jac_SAE ${lambda_jac_SAE} --lambda_dx ${lambda_dx} --lambda_dz ${lambda_dz} --lr ${lr} --gamma_lr ${gamma_lr} --activation ${activation} --activation_SAE ${activation_SAE} --load_model ${load_model} --load_iterations ${load_iterations} --lam ${lam} > ${OUTPUT_PREFIX}.log
