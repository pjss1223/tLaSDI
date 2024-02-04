#!/bin/bash
#BSUB -nnodes 1
#BSUB -q pbatch
#BSUB -W 720

problem="1DBG"
latent_dim="10"
#5430149
extraD_L="9" #2-12
extraD_M="9" #2-12

batch_size="60"

order="1"

update_epochs="2000" 

net="GFINNs"  # (GFINNs) or (SPNN)

method="AEhyper"
epochs="43001"
# loss weights  (Integrator loss weight: 1)
lambda_r_AE="1e-1"  # reconstruction 1e-1
lambda_jac_AE="1e-9"  # Jacobian 1e-6 1e-9
lambda_dx="1e-7" # Consistency 1e-7
lambda_dz="1e-7" # Model approximation 1e-7

if [ "$net" == "SPNN" ]; then
    lam="0"
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


load_model="False"

if [ "$load_model" == "True" ]; then
    load_epochs="43911"
else
    load_epochs="0"
fi

total_epochs=$(echo "$epochs+$load_epochs" | bc)


#Loading cuda will cause linking error
#module load cuda/11.4.1

source anaconda/bin/activate
conda activate opence-1.8.0

TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S") 
OUTPUT_PREFIX=${problem}_${method}_${latent_dim}_${net}_exDL${extraD_L}_exDM${extraD_M}_od${order}_${lambda_r_AE}_${lambda_jac_AE}_${lambda_dx}_${lambda_dz}_${lam}_${total_epochs}


python main_1DBG_tLaSDI_GAEhyper.py --order ${order} --latent_dim ${latent_dim} --extraD_L ${extraD_L} --extraD_M ${extraD_M} --xi_scale ${xi_scale} --net ${net} --epochs ${epochs} --update_epochs ${update_epochs} --batch_size ${batch_size}  --lambda_r_AE ${lambda_r_AE} --lambda_jac_AE ${lambda_jac_AE} --lambda_dx ${lambda_dx} --lambda_dz ${lambda_dz} --load_model ${load_model} --load_epochs ${load_epochs} --lam ${lam} > ${OUTPUT_PREFIX}.log
