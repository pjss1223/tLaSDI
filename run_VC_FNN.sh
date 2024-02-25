#!/bin/bash
#BSUB -nnodes 1
#BSUB -q pdebug
#BSUB -W 120

seed="0"
device="gpu"

problem="VC_fn"
latent_dim="8"
data_type="last"

layers="5"  # 4 20 works well for our method
width="215"

AE_width1="160"  # 80 40 works well for our method
AE_width2="160"

net="FNN"  # (GFINNs) or (SPNN)

iterations="40015"
# loss weights  (Integrator loss weight: 1)
lambda_r_AE="1e-1"  # reconstruction 1e-1
lambda_jac_AE="0"  # Jacobian 1e-4 or 1e-4(wo jac loss, with consistency),1e-6(wo jac loss, WO consistency)
lambda_dx="0" # Consistency 1e-4
lambda_dz="0" # Model approximation 1e-4 


lr="1e-4"

load_model="False"

if [ "$load_model" == "False" ]; then
    load_iterations="0"
else
    load_iterations="20002"
fi

total_iteration=$(echo "$iterations+$load_iterations" | bc)

activation="tanh"
activation_AE="relu"

#Loading cuda will cause linking error
#module load cuda/11.4.1
gamma_lr=".99"
miles_lr="1000"
weight_decay_AE="0"
weight_decay_GFINNs="1e-8"

source anaconda/bin/activate
conda activate opence-1.8.0

TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S") 
OUTPUT_PREFIX=${problem}_${latent_dim}_${net}_ly${layers}_wd${width}_Awd1${AE_width1}_Awd2${AE_width2}_${data_type}_${lambda_r_AE}_${lambda_jac_AE}_${lambda_dx}_${lambda_dz}_lr${lr}_gam${gamma_lr}_WDGF${weight_decay_GFINNs}_${activation}_${activation_AE}_${seed}_${total_iteration}


python main_VC_FNN.py --seed ${seed} --device ${device} --data_type ${data_type} --latent_dim ${latent_dim} --layers ${layers} --width ${width} --AE_width1 ${AE_width1} --AE_width2 ${AE_width2} --net ${net} --iterations ${iterations} --lambda_r_AE ${lambda_r_AE} --lambda_jac_AE ${lambda_jac_AE} --lambda_dx ${lambda_dx} --lambda_dz ${lambda_dz} --lr ${lr} --gamma_lr ${gamma_lr} --miles_lr ${miles_lr} --weight_decay_AE ${weight_decay_AE} --weight_decay_GFINNs ${weight_decay_GFINNs} --activation ${activation} --activation_AE ${activation_AE} --load_model ${load_model} --load_iterations ${load_iterations} > ${OUTPUT_PREFIX}.log
