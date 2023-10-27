"""main.py"""

import argparse

import numpy as np
import torch
import learner as ln
from learner import data
from utilities.utils import str2bool


#from data import Data
#from learner import Data
from data2 import Data
from nn_GFINNs import *
#from postprocess_dp import plot_DP
from learner.utils import grad
from dataset_sim import load_dataset, split_dataset

# import importlib


#------------------------------------------------- parameters changed frequently
# latent_dim = 10
# DINN = 'ESP3'  # 'ESP3' (GFINNs) or 'ESP3_soft' (SPNN)
# iterations = 50000  # 50000

# # loss weights  (Integrator loss weight: 1)
# lambda_r_SAE = 1e-1  # reconstruction
# lambda_jac_SAE = 1e-3  # Jacobian
# lambda_dx = 1e-1  # Consistency
# lambda_dz = 1e-1  # Model approximation


def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    
    device = args.device  # 'cpu' or 'gpu'
    dtype = 'double'

#     module_name = 'nn_GFINNs_' + str(args.extraD_L) if args.extraD_L in range(2, 12) else 'nn_GFINNs'
    
#     if args.extraD_L != args.extraD_M:
#         module_name = 'nn_GFINNs'
    
#     nn_module = importlib.import_module(module_name)
    
#     VC_LNN3 = nn_module.VC_LNN3
#     VC_MNN3 = nn_module.VC_MNN3
#     VC_LNN3_soft = nn_module.VC_LNN3_soft
#     VC_MNN3_soft = nn_module.VC_MNN3_soft
#     ESPNN = nn_module.ESPNN

    # data
    p = 0.8
    problem = 'GC'
    t_terminal = 40
    dt = 0.1
    trajs = 100
    order = 4
    iters = 1 #fixed to be 1
    trunc_period = 1


    if args.net == 'ESP3':
        DI_str = ''
    else:
        DI_str = 'soft'



    #print(data)
    # NN
    layers = args.layers  #4
    width = args.width  #20   #5 190 worked well
    
    AE_width1 = args.AE_width1
    AE_width2 = args.AE_width2
    
    activation = args.activation
    activation_SAE = args.activation_SAE
    #activation = 'relu'
    dataset = load_dataset('GC','data',device,dtype)
    
    weight_decay_AE = 1e-7
    weight_decay_GFINNs = 0 #1e-6
    
        
    #-----------------------------------------------------------------------------
    latent_dim = args.latent_dim
    iterations = args.iterations
    extraD_L = args.extraD_L
    extraD_M = args.extraD_M
    xi_scale = args.xi_scale
    
  
    load_model = args.load_model
    load_iterations = args.load_iterations
    
    lam = args.lam #degeneracy penalty
    lambda_r_SAE = args.lambda_r_SAE
    lambda_jac_SAE = args.lambda_jac_SAE
    lambda_dx = args.lambda_dx
    lambda_dz = args.lambda_dz
#     layer_vec_SAE = [100*4, 40*4,40*4, latent_dim]
#     layer_vec_SAE = [100*4, 200 ,100, latent_dim]
    layer_vec_SAE = [100*4, AE_width1 ,AE_width2, latent_dim]

    layer_vec_SAE_q = [4140*3, 40, 40, latent_dim]
    layer_vec_SAE_v = [4140*3, 40, 40, latent_dim]
    layer_vec_SAE_sigma = [4140*6, 40*2, 40*2, 2*latent_dim]
    #--------------------------------------------------------------------------------
    
    
    if args.load_model:
        AE_name = 'AE'+ str(latent_dim)+'_extraD_'+str(extraD_L) +DI_str+ '_REC'+"{:.0e}".format(lambda_r_SAE)  + '_JAC'+ "{:.0e}".format(lambda_jac_SAE) + '_CON'+"{:.0e}".format(lambda_dx) + '_APP' + "{:.0e}".format(lambda_dz)+ '_DEG' + "{:.0e}".format(lam)+activation+activation_SAE  + '_iter'+str(iterations+load_iterations)
    else:
        AE_name = 'AE'+ str(latent_dim)+'_extraD_'+str(extraD_L) +DI_str+ '_REC'+"{:.0e}".format(lambda_r_SAE)  + '_JAC'+ "{:.0e}".format(lambda_jac_SAE) + '_CON'+"{:.0e}".format(lambda_dx) + '_APP' + "{:.0e}".format(lambda_dz)+ '_DEG' + "{:.0e}".format(lam)+activation +activation_SAE + '_iter'+str(iterations)

   
    
    

    if args.net == 'ESP3':
        # netS = VC_LNN3(x_trunc.shape[1],5,layers=layers, width=width, activation=activation)
        # netE = VC_MNN3(x_trunc.shape[1],4,layers=layers, width=width, activation=activation)
        netS = VC_LNN3(latent_dim,extraD_L,layers=layers, width=width, activation=activation,xi_scale=xi_scale)
        netE = VC_MNN3(latent_dim,extraD_M,layers=layers, width=width, activation=activation,xi_scale=xi_scale)
        lam = 0
    elif args.net == 'ESP3_soft':
        netS = VC_LNN3_soft(latent_dim,layers=layers, width=width, activation=activation)
        netE = VC_MNN3_soft(latent_dim,layers=layers, width=width, activation=activation)
        lam = args.lam
    else:
        raise NotImplementedError

    #print(dataset.dt)  #0.006666666666666667
    net = ESPNN(netS, netE, dataset.dt / iters, order=order, iters=iters, lam=lam)

    #print(sum(p.numel() for p in net.parameters() if p.requires_grad))

    # training
    lr = args.lr #1e-5 VC, 1e-5    0.001 good with relu, 1e-4 good with tanh
    lbfgs_steps = 0
    print_every = 100
    batch_size = None
    batch_size_test = None

    load_path = problem + args.net+'AE' + str(latent_dim) + DI_str + '_REC' + "{:.0e}".format(lambda_r_SAE) + '_JAC' + "{:.0e}".format( lambda_jac_SAE) + '_CON' + "{:.0e}".format(lambda_dx) + '_APP' + "{:.0e}".format(lambda_dz) + '_iter' + str(load_iterations)
    path = problem + args.net + AE_name       # net = torch.load('outputs/'+path+'/model_best.pkl')

    args2 = {
        'net': net,
        # 'x_trunc': x_trunc,
        # 'latent_idx': latent_idx,
        'dt': dataset.dt,
        'z_gt': dataset.z,
        'sys_name':'GC',
        'output_dir': 'outputs',
        'save_plots': True,
        'criterion': None,
        'optimizer': 'adam',
        'lr': lr,
        'iterations': iterations,
        'lbfgs_steps': lbfgs_steps,
        # AE part
        'AE_name': AE_name,
        'dset_dir': 'data',
        'output_dir_AE': 'outputs',
        'save_plots_AE': True,
        'layer_vec_SAE': layer_vec_SAE,
        'layer_vec_SAE_q': layer_vec_SAE_q,
        'layer_vec_SAE_v': layer_vec_SAE_v,
        'layer_vec_SAE_sigma': layer_vec_SAE_sigma,
        'activation_SAE': activation_SAE, # linear relu tanh
        'lr_SAE': 1e-4,
        'lambda_r_SAE': lambda_r_SAE,
        'lambda_jac_SAE': lambda_jac_SAE,
        'lambda_dx':lambda_dx,
        'lambda_dz':lambda_dz,
#         'miles_lr': [1e4],
#         'gamma_lr': 1e-1,
        'weight_decay_AE':weight_decay_AE,
        'weight_decay_GFINNs':weight_decay_GFINNs,
        'path': path,
        'load_path': load_path,
        'batch_size': batch_size,
        'batch_size_test': batch_size_test,
        'print_every': print_every,
        'save': True,
        'load':load_model,
        'callback': None,
        'dtype': dtype,
        'device': device,
        'trunc_period': trunc_period
    }

    ln.Brain_tLaSDI.Init(**args2)
    ln.Brain_tLaSDI.Run()
    ln.Brain_tLaSDI.Restore()
    ln.Brain_tLaSDI.Output()
    ln.Brain_tLaSDI.Test()



if __name__ == "__main__":


    # Training Parameters


    # GFINNs
    #parser = argparse.ArgumentParser(description='Generic Neural Networks')
    parser = argparse.ArgumentParser(description='Deep learning of thermodynamics-aware reduced-order models from data')


    # # Dataset Parameters
    # parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    #parser.add_argument('--lambda_jac_SAE', default=5e2, type=float, help='Jacobian (regularization) weight SAE')#1e-4 VC, 1e-2 RT
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    #

    # ## Sparse Autoencoder
    # # Net Parameters
#     parser.add_argument('--layer_vec_SAE', default=[100*4, 40*4,40*4, latent_dim], nargs='+', type=int, help='full layer vector of the viscolastic SAE')
#     parser.add_argument('--layer_vec_SAE_q', default=[4140*3, 40, 40, 10], nargs='+', type=int, help='full layer vector (position) of the rolling tire SAE')
#     parser.add_argument('--layer_vec_SAE_v', default=[4140*3, 40, 40, 10], nargs='+', type=int, help='full layer vector (velocity) of the rolling tire SAE')
#     parser.add_argument('--layer_vec_SAE_sigma', default=[4140*6, 40*2, 40*2, 2*10], nargs='+', type=int, help='full layer vector (stress tensor) of the rolling tire SAE')
#     parser.add_argument('--activation_SAE', default='relu', type=str, help='activation function')



    # GFINNs
    #parser = argparse.ArgumentParser(description='Generic Neural Networks')
    #parser.add_argument('--net', default=DINN, type=str, help='ESP or ESP2 or ESP3')
    parser.add_argument('--lam', default=1e-2, type=float, help='lambda as the weight for consistency penalty')
    #parser.add_argument('--seed2', default=0, type=int, help='random seed')
    
    parser.add_argument('--activation', type=str, choices=["tanh", "relu","linear","sin","gelu"], default="gelu",
                        help='activation functions for GFINNs or SPNN')
    
    parser.add_argument('--device', type=str, choices=["gpu", "cpu"], default="gpu",
                        help='device used')
    
    parser.add_argument('--activation_SAE', type=str, choices=["tanh", "relu","linear","sin","gelu"], default="relu",
                        help='ESP3 for GFINN and ESP3_soft for SPNN')
    
    
    parser.add_argument('--layers', type=int, default=5,
                        help='number of layers for GFINNs.')
    parser.add_argument('--width', type=int, default=100,
                        help='width of GFINNs.')
    
    parser.add_argument('--AE_width1', type=int, default=200,
                        help='first width for AE.')
    
    parser.add_argument('--AE_width2', type=int, default=100,
                        help='second width for AE.')
                        
    parser.add_argument('--latent_dim', type=int, default=10,
                        help='Latent dimension.')
    parser.add_argument('--extraD_L', type=int, default=10,
                        help='extraD for L.')
    parser.add_argument('--extraD_M', type=int, default=10,
                        help='extraD for M.')
    parser.add_argument('--xi_scale', type=float, default=1e-2,
                        help='scale for initialized skew-symmetric matrices')

    parser.add_argument('--net', type=str, choices=["ESP3", "ESP3_soft"], default="ESP3_soft",
                        help='ESP3 for GFINN and ESP3_soft for SPNN')

    parser.add_argument('--iterations', type=int, default=53,
                        help='number of iterations')
    
    parser.add_argument('--load_iterations', type=int, default=100,
                        help='number of iterations of loaded network')

    parser.add_argument('--lambda_r_SAE', type=float, default=1e-1,
                        help='Penalty for reconstruction loss.')

    parser.add_argument('--lambda_jac_SAE', type=float, default=1e-6,
                        help='Penalty for Jacobian loss.')

    parser.add_argument('--lambda_dx', type=float, default=1e-4,
                        help='Penalty for Consistency loss.')

    parser.add_argument('--lambda_dz', type=float, default=1e-4,
                        help='Penalty for Model approximation loss.')
    
    parser.add_argument('--load_model', default=False, type=str2bool, 
                        help='load previously trained model')
    
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='rate of learning rate decay.')
    
    parser.add_argument('--miles_lr',  type=int, default= 1000,
                        help='iteration steps for learning rate decay ')

    parser.add_argument('--gamma_lr', type=float, default=1.0,
                        help='rate of learning rate decay.')

    
    
    
    
    args = parser.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    main(args)



    #args = parser.parse_args()



