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
    
    device = args.device  # 'cpu' or 'gpu'
    dtype = 'double'

    problem = 'VC'
    order = args.order
    iters = 1 # fixed to be 1
    trunc_period = 1 # when computing Jacobian, we only consider every 'trunc_period'th index

    ROM_model = 'tLaSDI'

    layers = args.layers  
    width = args.width  
    
    AE_width1 = args.AE_width1
    AE_width2 = args.AE_width2
    
    activation = args.activation
    activation_AE = args.activation_AE

    dataset = load_dataset('viscoelastic','data',device,dtype)
    
    weight_decay_AE = args.weight_decay_AE
    weight_decay_GFINNs = args.weight_decay_GFINNs    
    
    miles_lr = args.miles_lr
    gamma_lr = args.gamma_lr 
    
    lr = args.lr 
    lr_AE = args.lr_AE

    #-----------------------------------------------------------------------------
    latent_dim = args.latent_dim
    iterations = args.iterations
    extraD_L = args.extraD_L
    extraD_M = args.extraD_M
    xi_scale = args.xi_scale

    load_model = args.load_model
    load_iterations = args.load_iterations

    lam = args.lambda_deg #degeneracy penalty
    lambda_r_AE = args.lambda_r_AE
    lambda_jac_AE = args.lambda_jac_AE
    lambda_dx = args.lambda_dx
    lambda_dz = args.lambda_dz

    layer_vec_AE = [100*4, AE_width1 ,AE_width2, latent_dim]

    #--------------------------------------------------------------------------------
    
    if args.load_model:
        AE_name = '_REC'+"{:.0e}".format(lambda_r_AE) + '_JAC'+ "{:.0e}".format(lambda_jac_AE) + '_MOD'+"{:.0e}".format(lambda_dx) + '_DEG' + "{:.0e}".format(lam)  + '_iter'+str(iterations+load_iterations)
    else:
        AE_name = '_REC'+"{:.0e}".format(lambda_r_AE) + '_JAC'+ "{:.0e}".format(lambda_jac_AE) + '_MOD'+"{:.0e}".format(lambda_dx) + '_DEG' + "{:.0e}".format(lam)+ '_iter'+str(iterations)

    load_path =  problem + '_tLaSDI-' + args.net +'_REC'+"{:.0e}".format(lambda_r_AE) + '_JAC'+ "{:.0e}".format(lambda_jac_AE) + '_MOD'+"{:.0e}".format(lambda_dx) + '_DEG' + "{:.0e}".format(lam)+ '_iter'+str(load_iterations)

    path = problem + '_tLaSDI-' + args.net + AE_name


    if args.net == 'GFINNs':
        netS = LNN(latent_dim,extraD_L,layers=layers, width=width, activation=activation,xi_scale=xi_scale)
        netE = MNN(latent_dim,extraD_M,layers=layers, width=width, activation=activation,xi_scale=xi_scale)
        lam = 0

    elif args.net == 'SPNN':
        netS = LNN_soft(latent_dim,layers=layers, width=width, activation=activation)
        netE = MNN_soft(latent_dim,layers=layers, width=width, activation=activation)
        lam = args.lambda_deg

    else:
        raise NotImplementedError

    net = GFINNs(netS, netE, dataset.dt / iters, order=order, iters=iters, lam=lam)

    print_every = 100
    batch_size = None

    args2 = {
        'ROM_model':ROM_model,
        'net': net,
        'sys_name':'viscoelastic',
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
        'activation_AE': activation_AE,
        'lr_AE': lr_AE,
        'lambda_r_AE': lambda_r_AE,
        'lambda_jac_AE': lambda_jac_AE,
        'lambda_dx':lambda_dx,
        'lambda_dz':lambda_dz,
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

    parser = argparse.ArgumentParser(description='Deep learning of thermodynamics-aware reduced-order models from data')

    parser.add_argument('--seed', default=0, type=int,
                        help='random seed')

    parser.add_argument('--device', type=str, choices=["gpu", "cpu"], default="cpu",
                        help='device used')

    # architecture / AE

    parser.add_argument('--activation_AE', type=str, choices=["tanh", "relu", "linear", "sin", "gelu", "elu", "silu"], default="relu",
                        help='activation function for AE')

    parser.add_argument('--AE_width1', type=int, default=160,
                        help='width for the first layer of AE')

    parser.add_argument('--AE_width2', type=int, default=160,
                        help='width for the second layer of AE')

    parser.add_argument('--latent_dim', type=int, default=8,
                        help='Latent space dimension')


    # architecture / DI model

    parser.add_argument('--net', type=str, choices=["GFINNs", "SPNN"], default="GFINNs",
                        help='DI model choice')

    parser.add_argument('--activation', type=str, default="tanh",
                        help='activation functions for DI model')

    parser.add_argument('--layers', type=int, default=5,
                        help='number of layers for DI model')

    parser.add_argument('--width', type=int, default=100,
                        help='width of DI model')

    parser.add_argument('--extraD_L', type=int, default=8,
                        help='# of skew-symmetric matrices generated to construct L in GFINNs')

    parser.add_argument('--extraD_M', type=int, default=8,
                        help='# of skew-symmetric matrices generated to construct M in GFINNs')

    parser.add_argument('--xi_scale', type=float, default=.3779,
                        help='scale for the initialization of skew-symmetric matrices (GFINNs)')


    # Training parameters

    parser.add_argument('--load_model', default=False, type=str2bool,
                        help='load previously trained model')

    parser.add_argument('--iterations', type=int, default=101,
                        help='number of iterations')
    
    parser.add_argument('--load_iterations', type=int, default=101,
                        help='previous number of iterations for loaded networks')

    parser.add_argument('--lambda_r_AE', type=float, default=1e-1,
                        help='weight for reconstruction loss.')

    parser.add_argument('--lambda_jac_AE', type=float, default=1e-2,
                        help='weight for Jacobian loss.')

    parser.add_argument('--lambda_dx', type=float, default=1e-8,
                        help='weight for consistency part of model loss')

    parser.add_argument('--lambda_dz', type=float, default=1e-8,
                        help='weight for model approximation part of model loss.')

    parser.add_argument('--lambda_deg', default=0, type=float,
                        help='weight for degeneracy loss')

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate for training DI model')
    
    parser.add_argument('--lr_AE', type=float, default=1e-4,
                        help='learning rate for training AE')
    
    parser.add_argument('--miles_lr',  type=int, default= 1000,
                        help='learning rate decay frequency')

    parser.add_argument('--gamma_lr', type=float, default=.99,
                        help='rate of learning rate decay')


    
    parser.add_argument('--weight_decay_AE', type=float, default=0,
                        help='weight decay for AE')

    parser.add_argument('--weight_decay_GFINNs', type=float, default=1e-8,
                        help='weight decay for DI model')


    parser.add_argument('--order', type=int, default=4,
                        help='DI model time integrator 1:Euler, 2:RK23, 4:RK45')

    
    args = parser.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    main(args)

