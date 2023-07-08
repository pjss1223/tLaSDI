"""main.py"""

import argparse

import numpy as np
import torch
import learner as ln
from learner import data


#from data import Data
#from learner import Data
from data2 import Data
from nn_GFINNs import *
#from postprocess_dp import plot_DP
from learner.utils import grad
from dataset_sim import load_dataset, split_dataset
from utilities.utils import str2bool



device = 'gpu'  # 'cpu' or 'gpu'
dtype = 'double'

#------------------------------------------------- parameters changed frequently
# latent_dim = 10
# DINN = 'ESP3'  # 'ESP3' (GFINNs) or 'ESP3_soft' (SPNN)
# iterations = 30000   # may be 20000 should work better

# # loss weights  (Integrator loss weight: 1)
# lambda_r_SAE = 1e-1  # reconstruction
# lambda_jac_SAE = 1e-6  # Jacobian
# lambda_dx = 1e-4  # Consistency
# lambda_dz = 1e-4  # Model approximation


def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # data
    p = 0.8
    problem = 'RT'
    t_terminal = 40
    dt = 0.1
    trajs = 100
    order = 2
    iters = 1
    trunc_period=80


    if args.net == 'ESP3':
        DI_str = ''
    else:
        DI_str = 'soft'



    #print(data)
    # NN
    layers = 5  #5   
    width = 198  #198 
    activation = 'tanh'
    #activation = 'relu'
    dataset = load_dataset('rolling_tire','data',device, dtype)
    
    
    #-----------------------------------------------------------------------------
    latent_dim = args.latent_dim
    latent_dim_q = args.latent_dim_q
    latent_dim_v = args.latent_dim_v
    latent_dim_sigma = args.latent_dim_sigma
    iterations = args.iterations
    
    load_model = args.load_model
    load_iterations = args.load_iterations
    
    
    lambda_r_SAE = args.lambda_r_SAE
    lambda_jac_SAE = args.lambda_jac_SAE
    lambda_dx = args.lambda_dx
    lambda_dz = args.lambda_dz
    
    gamma_lr = args.gamma_lr
    miles_lr = args.miles_lr
    
    weight_decay_GFINNs = args.weight_decay_GFINNs
    weight_decay_AE = args.weight_decay_AE
    
    layer_vec_SAE = [101,100,latent_dim]
#     layer_vec_SAE_q = [4140*3, 40, 40, latent_dim]
#     layer_vec_SAE_v = [4140*3, 40, 40, latent_dim]
#     layer_vec_SAE_sigma = [4140*6, 40*2, 40*2, 2*latent_dim]
    
#     layer_vec_SAE_q = [2070*3, 40, 40, latent_dim]
#     layer_vec_SAE_v = [2070*3, 40, 40, latent_dim]
#     layer_vec_SAE_sigma = [2070*6, 40*2, 40*2, 2*latent_dim]
    
    layer_vec_SAE_q = [1035*3, 40, 40, latent_dim_q]
    layer_vec_SAE_v = [1035*3, 40, 40, latent_dim_v]
    layer_vec_SAE_sigma = [1035*6, 40*2, 40*2, latent_dim_sigma]
    #--------------------------------------------------------------------------------
    
    if args.load_model:
        AE_name = 'AE'+ '_'+ str(latent_dim_q)+'_'+ str(latent_dim_v)+'_'+ str(latent_dim_sigma)+'_'+ DI_str+ '_REC'+"{:.0e}".format(lambda_r_SAE)  + '_JAC'+ "{:.0e}".format(lambda_jac_SAE) + '_CON'+"{:.0e}".format(lambda_dx) + '_APP' + "{:.0e}".format(lambda_dz) + '_iter'+str(iterations+load_iterations)
    else:
        AE_name = 'AE'+ '_'+ str(latent_dim_q)+'_'+ str(latent_dim_v)+'_'+ str(latent_dim_sigma)+'_'+ DI_str+ '_REC'+"{:.0e}".format(lambda_r_SAE)  + '_JAC'+ "{:.0e}".format(lambda_jac_SAE) + '_CON'+"{:.0e}".format(lambda_dx) + '_APP' + "{:.0e}".format(lambda_dz) + '_iter'+str(iterations)

    
    
    

    if args.net == 'ESP3':
        # netS = VC_LNN3(x_trunc.shape[1],5,layers=layers, width=width, activation=activation)
        # netE = VC_MNN3(x_trunc.shape[1],4,layers=layers, width=width, activation=activation)
        #netS = VC_LNN3(4*latent_dim,10,layers=layers, width=width, activation=activation)
        #netE = VC_MNN3(4*latent_dim,8,layers=layers, width=width, activation=activation)
        netS = VC_LNN3(latent_dim_q+latent_dim_v+latent_dim_sigma,12,layers=layers, width=width, activation=activation)
        netE = VC_MNN3(latent_dim_q+latent_dim_v+latent_dim_sigma,12,layers=layers, width=width, activation=activation)
        lam = 0
    elif args.net == 'ESP3_soft':
        #netS = VC_LNN3_soft(4*latent_dim,layers=layers, width=width, activation=activation)
        #netE = VC_MNN3_soft(4*latent_dim,layers=layers, width=width, activation=activation)
        netS = VC_LNN3_soft(latent_dim_q+latent_dim_v+latent_dim_sigma,layers=layers, width=width, activation=activation)
        netE = VC_MNN3_soft(latent_dim_q+latent_dim_v+latent_dim_sigma,layers=layers, width=width, activation=activation)
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
    
    load_path = problem + args.net+'AE' + '_'+ str(latent_dim_q)+'_'+ str(latent_dim_v)+'_'+ str(latent_dim_sigma)+'_'+ DI_str + '_REC' + "{:.0e}".format(lambda_r_SAE) + '_JAC' + "{:.0e}".format( lambda_jac_SAE) + '_CON' + "{:.0e}".format(lambda_dx) + '_APP' + "{:.0e}".format(lambda_dz) + '_iter' + str(load_iterations)
    path = problem + args.net + AE_name    


    args2 = {
        'net': net,
        # 'x_trunc': x_trunc,
        # 'latent_idx': latent_idx,
        'dt': dataset.dt,
        'z_gt': dataset.z,
        'sys_name':'rolling_tire',
        'output_dir': 'outputs',
        'save_plots': True,
        'criterion': None,
        'optimizer': 'adam',
        'lr': lr,
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
        'activation_SAE': 'relu',
        'lr_SAE': 1e-4,#no need for simul training
        'lambda_r_SAE': lambda_r_SAE,
        'lambda_jac_SAE': lambda_jac_SAE,
        'lambda_dx':lambda_dx,
        'lambda_dz':lambda_dz,
        'miles_lr':miles_lr,
        'gamma_lr':gamma_lr,
        'path': path,
        'load_path': load_path,
        'batch_size': batch_size,
        'batch_size_test': batch_size_test,
        'weight_decay_AE':weight_decay_AE,
        'weight_decay_GFINNs':weight_decay_GFINNs,
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
#     parser.add_argument('--layer_vec_SAE', default=[100*4, 40*4,40*4, 10], nargs='+', type=int, help='full layer vector of the viscolastic SAE')
#     parser.add_argument('--layer_vec_SAE_q', default=[4140*3, 40, 40, latent_dim], nargs='+', type=int, help='full layer vector (position) of the rolling tire SAE')
#     parser.add_argument('--layer_vec_SAE_v', default=[4140*3, 40, 40, latent_dim], nargs='+', type=int, help='full layer vector (velocity) of the rolling tire SAE')
#     parser.add_argument('--layer_vec_SAE_sigma', default=[4140*6, 40*2, 40*2, 2*latent_dim], nargs='+', type=int, help='full layer vector (stress tensor) of the rolling tire SAE')
#     parser.add_argument('--activation_SAE', default='relu', type=str, help='activation function')


  # GFINNs
    #parser = argparse.ArgumentParser(description='Generic Neural Networks')
    #parser.add_argument('--net', default=DINN, type=str, help='ESP or ESP2 or ESP3')
    parser.add_argument('--lam', default=1, type=float, help='lambda as the weight for consistency penalty')
    #parser.add_argument('--seed2', default=0, type=int, help='random seed')
    
 
    parser.add_argument('--latent_dim', type=int, default=10,
                        help='Latent dimension.')

    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate')
    
    parser.add_argument('--latent_dim_q', type=int, default=4,
                        help='Latent dimension.')
    parser.add_argument('--latent_dim_v', type=int, default=3,
                        help='Latent dimension.')
    parser.add_argument('--latent_dim_sigma', type=int, default=2,
                        help='Latent dimension.')

    parser.add_argument('--net', type=str, choices=["ESP3", "ESP3_soft"], default="ESP3",
                        help='ESP3 for GFINN and ESP3_soft for SPNN')

    parser.add_argument('--iterations', type=int, default=10000,
                        help='number of iterations')
    
    parser.add_argument('--load_iterations', type=int, default=10,
                        help='number of iterations of loaded network')

    parser.add_argument('--lambda_r_SAE', type=float, default=1e-2,
                        help='Penalty for reconstruction loss.')

    parser.add_argument('--lambda_jac_SAE', type=float, default=0,#1e-6
                        help='Penalty for Jacobian loss.')

    parser.add_argument('--lambda_dx', type=float, default=0, #1e-4
                        help='Penalty for Consistency loss.')

    parser.add_argument('--lambda_dz', type=float, default=0,  #1e-4
                        help='Penalty for Model approximation loss.')
    
    parser.add_argument('--load_model', default=False, type=str2bool, 
                        help='load previously trained model')

    parser.add_argument('--miles_lr',  type=int, default=[10000],
                        help='iteration steps for learning rate decay ')

    parser.add_argument('--gamma_lr', type=float, default=1e-1,
                        help='rate of learning rate decay.')
    
    parser.add_argument('--weight_decay_GFINNs', type=float, default=0,
                        help='rate of learning rate decay for GFINNs.')
    
    parser.add_argument('--weight_decay_AE', type=float, default=0,
                        help='rate of learning rate decay for AE.')
    
    
    
  
    args = parser.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    main(args)



    #args = parser.parse_args()



