"""main.py"""

import argparse

import numpy as np
import torch
import learner as ln
from learner import data
from utilities.utils import str2bool


from data import Data
from nn_GFINNs import *
from dataset_sim import load_dataset


def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)  
    
    device = args.device
    dtype = 'double'

    problem = 'GC'
    
    order = args.order
    lr_scheduler_type = args.lr_scheduler_type
        
    iters = 1 #fixed to be 1
    trunc_period = 1 # when computing Jacobian, we only consider every 'trunc_period'th index
    
    data_type = args.data_type
    
    ROM_model = 'tLaSDI'

    if args.net == 'GFINNs':
        DI_str = ''
    else:
        DI_str = 'soft'


    layers = args.layers
    width = args.width
    
    AE_width1 = args.AE_width1
    AE_width2 = args.AE_width2
    
    activation = args.activation
    activation_AE = args.activation_AE
    dataset = load_dataset('GC','data',device,dtype)
    
    weight_decay_AE = args.weight_decay_AE
    weight_decay_GFINNs = args.weight_decay_GFINNs 
    
    
    miles_lr = args.miles_lr
    gamma_lr = args.gamma_lr
    
    miles_lr_print = miles_lr
    
    if lr_scheduler_type == "MultiStepLR":
        miles_lr = [miles_lr]

    #-----------------------------------------------------------------------------
    latent_dim = args.latent_dim
    iterations = args.iterations
    extraD_L = args.extraD_L
    extraD_M = args.extraD_M
    xi_scale = args.xi_scale
  
    load_model = args.load_model
    load_iterations = args.load_iterations
    
    lam = args.lam #degeneracy penalty
    lambda_r_AE = args.lambda_r_AE
    lambda_jac_AE = args.lambda_jac_AE
    lambda_dx = args.lambda_dx
    lambda_dz = args.lambda_dz

    layer_vec_AE = [100*4, AE_width1 ,AE_width2, latent_dim]

    #--------------------------------------------------------------------------------
    
    if args.load_model:
        AE_name = 'AE'+ str(latent_dim)+'_extraD_'+str(extraD_L) +DI_str+ '_REC'+"{:.0e}".format(lambda_r_AE)  + '_JAC'+ "{:.0e}".format(lambda_jac_AE) + '_CON'+"{:.0e}".format(lambda_dx) + '_APP' + "{:.0e}".format(lambda_dz)+ '_DEG' + "{:.0e}".format(lam)+activation+activation_AE+ '_Gam'+ str(int(gamma_lr * 100))+ '_Mil'+ str(int(miles_lr_print))+ '_WDG'+ "{:.0e}".format(weight_decay_GFINNs)+ '_WDA'+ "{:.0e}".format(weight_decay_AE)+'_' +str(data_type)+'_OD'+str(order) +'_'+str(seed) + '_iter'+str(iterations+load_iterations)
    else:
        AE_name = 'AE'+ str(latent_dim)+'_extraD_'+str(extraD_L) +DI_str+ '_REC'+"{:.0e}".format(lambda_r_AE)  + '_JAC'+ "{:.0e}".format(lambda_jac_AE) + '_CON'+"{:.0e}".format(lambda_dx) + '_APP' + "{:.0e}".format(lambda_dz)+ '_DEG' + "{:.0e}".format(lam)+activation+activation_AE+ '_Gam'+ str(int(gamma_lr * 100))+ '_Mil'+ str(int(miles_lr_print))+ '_WDG'+ "{:.0e}".format(weight_decay_GFINNs)+ '_WDA'+ "{:.0e}".format(weight_decay_AE)+'_' +str(data_type)+'_OD'+str(order) +'_'+str(seed) + '_iter'+str(iterations)

    load_path =  problem + args.net +'AE'+ str(latent_dim)+'_extraD_'+str(extraD_L) +DI_str+ '_REC'+"{:.0e}".format(lambda_r_AE)  + '_JAC'+ "{:.0e}".format(lambda_jac_AE) + '_CON'+"{:.0e}".format(lambda_dx) + '_APP' + "{:.0e}".format(lambda_dz)+ '_DEG' + "{:.0e}".format(lam)+activation+activation_AE+ '_Gam'+ str(int(gamma_lr * 100))+ '_Mil'+ str(int(miles_lr_print))+ '_WDG'+ "{:.0e}".format(weight_decay_GFINNs)+ '_WDA'+ "{:.0e}".format(weight_decay_AE)+'_' +str(data_type)+'_OD'+str(order) +'_'+str(seed) + '_iter'+str(load_iterations)

    path = problem + args.net + AE_name    

    if args.net == 'GFINNs':
        netS = LNN(latent_dim,extraD_L,layers=layers, width=width, activation=activation,xi_scale=xi_scale)
        netE = MNN(latent_dim,extraD_M,layers=layers, width=width, activation=activation,xi_scale=xi_scale)
        lam = 0
    elif args.net == 'SPNN':
        netS = LNN_soft(latent_dim,layers=layers, width=width, activation=activation)
        netE = MNN_soft(latent_dim,layers=layers, width=width, activation=activation)
        lam = args.lam
    else:
        raise NotImplementedError

    net = GFINNs(netS, netE, dataset.dt / iters, order=order, iters=iters, lam=lam)

    # training
    lr = args.lr 
    print_every = 100
    batch_size = None
    batch_size_test = None

    args2 = {
        'ROM_model':ROM_model,
        'net': net,
        'data_type': data_type,
        'sys_name':'GC',
        'output_dir': 'outputs',
        'save_plots': True,
        'criterion': None,
        'optimizer': 'adam',
        'lr': lr,
        'iterations': iterations,
        'AE_name': AE_name,
        'dset_dir': 'data',
        'output_dir_AE': 'outputs',
        'layer_vec_AE': layer_vec_AE,
        'activation_AE': activation_AE, # linear relu tanh
        'lr_AE': lr,
        'lambda_r_AE': lambda_r_AE,
        'lambda_jac_AE': lambda_jac_AE,
        'lambda_dx':lambda_dx,
        'lambda_dz':lambda_dz,
        'lr_scheduler_type':lr_scheduler_type,
        'miles_lr': miles_lr,
        'gamma_lr': gamma_lr,
        'weight_decay_AE':weight_decay_AE,
        'weight_decay_GFINNs':weight_decay_GFINNs,
        'path': path,
        'load_path': load_path,
        'batch_size': batch_size,
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
    parser = argparse.ArgumentParser(description='Deep learning of thermodynamics-aware reduced-order models from data')

    parser.add_argument('--seed', default=9912, type=int, help='random seed')
    #
    parser.add_argument('--lam', default=0, type=float, help='lambda as the weight for degeneracy penalty')
    
    parser.add_argument('--activation', type=str, choices=["tanh", "relu","linear","sin","gelu"], default="sin",
                        help='activation functions for GFINNs')
    
    parser.add_argument('--device', type=str, choices=["gpu", "cpu"], default="cpu",
                        help='device used')
    
    parser.add_argument('--activation_AE', type=str, choices=["tanh", "relu","linear","sin","gelu"], default="relu",
                        help='activation fonctions for AE')
    
    parser.add_argument('--data_type', type=str, default="last",
                        help='Test data type')
    
    parser.add_argument('--layers', type=int, default=5,
                        help='number of layers for GFINNs.')

    parser.add_argument('--width', type=int, default=200,
                        help='width of GFINNs.')
    
    parser.add_argument('--AE_width1', type=int, default=200,
                        help='width for the first layer of AE.')
    
    parser.add_argument('--AE_width2', type=int, default=100,
                        help='width for the second layer of AE.')
                        
    parser.add_argument('--latent_dim', type=int, default=30,
                        help='Latent space dimension.')

    parser.add_argument('--extraD_L', type=int, default=29,
                        help='# of skew-symmetric matrices generated for L.')

    parser.add_argument('--extraD_M', type=int, default=29,
                        help='# of skew-symmetric matrices generated for M.')

    parser.add_argument('--xi_scale', type=float, default=.1856,
                        help='scale for initialized skew-symmetric matrices')

    parser.add_argument('--net', type=str, choices=["GFINNs", "SPNN"], default="GFINNs",
                        help='DI model choices')

    parser.add_argument('--iterations', type=int, default=101,
                        help='number of iterations')
    
    parser.add_argument('--load_iterations', type=int, default=201,
                        help='previous number of iterations for loaded networks')

    parser.add_argument('--lambda_r_AE', type=float, default=1e-1,
                        help='Penalty for reconstruction loss.')

    parser.add_argument('--lambda_jac_AE', type=float, default=1e-2,
                        help='Penalty for Jacobian loss.')

    parser.add_argument('--lambda_dx', type=float, default=1e-7,
                        help='Penalty for consistency part of model loss')

    parser.add_argument('--lambda_dz', type=float, default=1e-7,
                        help='Penalty for model approximation part of model loss.')
    
    parser.add_argument('--load_model', default=False, type=str2bool,
                        help='load previously trained model')
    
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate for GFINNs.')
    
    parser.add_argument('--miles_lr', type=int, default= 1000,
                        help='iteration steps for learning rate decay')

    parser.add_argument('--gamma_lr', type=float, default=.99,
                        help='learning rate decay rate')

    parser.add_argument('--weight_decay_AE', type=float, default=0,
                        help='weight decay for AE')

    parser.add_argument('--weight_decay_GFINNs', type=float, default=0,
                        help='weight decay for GFINNs')
    
    parser.add_argument('--order', type=int, default=2,
                        help='integrator 1:Euler, 2:RK23, 4:RK45')
    
    parser.add_argument('--lr_scheduler_type', choices=["StepLR", "MultiStepLR"], default='StepLR', type=str, help='learning rate scheduler type')
    
    args = parser.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    main(args)



    #args = parser.parse_args()



