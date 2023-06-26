"""main.py"""


#1D Burgers
import argparse

import numpy as np
import torch
import learner as ln
from learner import data


#from data import Data
#from learner import Data
from data2 import Data
from nn_GFINNs_hyper import *
#from postprocess_dp import plot_DP
from learner.utils import grad
from dataset_sim_hyper import load_dataset, split_dataset
from utilities.utils import str2bool




device = 'gpu'  # 'cpu' or 'gpu'
dtype = 'double'

#------------------------------------------------- parameters changed frequently
#latent_dim = 10
#DINN = 'ESP3'  # 'ESP3' (GFINNs) or 'ESP3_soft' (SPNN)
#iterations = 10000   # may be 20000 should work better

# loss weights  (Integrator loss weight: 1)
# lambda_r_SAE = 1e-1  # reconstruction
# lambda_jac_SAE = 1e-6  # Jacobian
# lambda_dx = 1e-4 # Consistency
# lambda_dz = 1e-4 # Model approximation


def main(args):


    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)



    problem = 'BG'


    order = 2
    iters = 1
    trunc_period = 3


    depth_trunk = 3
    width_trunk = 40

    depth_hyper = 3   #hypernet structure for G (Entropy, Energy functions)
    width_hyper = 40

    depth_hyper2 = 2   #hypernet sturucture for skew-sym matrices
    width_hyper2 = 20


    act_trunk = 'tanh'
    act_hyper = 'tanh'
    act_hyper2 = 'tanh'
    #activation = 'relu'
    num_sensor = 2 # dimension of parameters
    
    
    
    
    if args.net == 'ESP3':
        DI_str = ''
    else:
        DI_str = 'soft'

    
    
    #-----------------------------------------------------------------------------
    latent_dim = args.latent_dim
    iterations = args.iterations
    
    load_model = args.load_model
    load_iterations = args.load_iterations
    
    
    lambda_r_SAE = args.lambda_r_SAE
    lambda_jac_SAE = args.lambda_jac_SAE
    lambda_dx = args.lambda_dx
    lambda_dz = args.lambda_dz
    layer_vec_SAE = [301,100,latent_dim]
    layer_vec_SAE_q = [4140*3, 40, 40, latent_dim]
    layer_vec_SAE_v = [4140*3, 40, 40, latent_dim]
    layer_vec_SAE_sigma = [4140*6, 40*2, 40*2, 2*latent_dim]
    #--------------------------------------------------------------------------------

    if args.load_model:
        AE_name = 'AE'+ str(latent_dim) +DI_str+ '_REC'+"{:.0e}".format(lambda_r_SAE)  + '_JAC'+ "{:.0e}".format(lambda_jac_SAE) + '_CON'+"{:.0e}".format(lambda_dx) + '_APP' + "{:.0e}".format(lambda_dz) + '_iter'+str(iterations+load_iterations)
    else:
        AE_name = 'AE'+ str(latent_dim) +DI_str+ '_REC'+"{:.0e}".format(lambda_r_SAE)  + '_JAC'+ "{:.0e}".format(lambda_jac_SAE) + '_CON'+"{:.0e}".format(lambda_dx) + '_APP' + "{:.0e}".format(lambda_dz) + '_iter'+str(iterations)








    dataset = load_dataset('1DBurgers','data',device)

    #train_snaps, test_snaps = split_dataset(dataset.z.shape[0] - 1)

#(self,num_sensor, ind, extraD, depth_trunk=2, width_trunk=50, depth_hyper=2, width_hyper=50, act_trunk='relu', act_hyper='relu', initializer='default',softmax=False):
    if args.net == 'ESP3':

        netS = VC_LNN3(num_sensor,latent_dim,10,depth_trunk=depth_trunk, width_trunk=width_trunk, depth_hyper=depth_hyper, width_hyper=width_hyper, depth_hyper2=depth_hyper2, width_hyper2=width_hyper2, act_trunk=act_trunk, act_hyper=act_hyper, act_hyper2=act_hyper2, initializer='default',softmax=False)
        netE = VC_MNN3(num_sensor,latent_dim,8,depth_trunk=depth_trunk, width_trunk=width_trunk, depth_hyper=depth_hyper, width_hyper=width_hyper, depth_hyper2=depth_hyper2, width_hyper2=width_hyper2, act_trunk=act_trunk, act_hyper=act_hyper, act_hyper2=act_hyper2, initializer='default',softmax=False)
        lam = 0
    elif args.net == 'ESP3_soft':
        netS = VC_LNN3_soft(num_sensor,latent_dim,depth_trunk=depth_trunk, width_trunk=width_trunk, depth_hyper=depth_hyper, width_hyper=width_hyper, act_trunk=act_trunk, act_hyper=act_hyper, initializer='default',softmax=False)
        netE = VC_MNN3_soft(num_sensor,latent_dim,depth_trunk=depth_trunk, width_trunk=width_trunk, depth_hyper=depth_hyper, width_hyper=width_hyper, act_trunk=act_trunk, act_hyper=act_hyper, initializer='default',softmax=False)
        lam = args.lam
    else:
        raise NotImplementedError

    #print(dataset.dt)  #0.006666666666666667
    net = ESPNN(netS, netE, dataset.dt / iters, order=order, iters=iters, lam=lam)

    # print(sum(p.numel() for p in netS.parameters() if p.requires_grad))
    # print(sum(p.numel() for p in netE.parameters() if p.requires_grad))
    #print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    
    #print(train_snaps.shape)
    #print #150 400
    # training
    lr = 1e-4  #1e-5 VC, 1e-5    0.001 good with relu, 1e-4 good with tanh
    #lr = 1e-3
    lbfgs_steps = 0
    print_every = 200
    #batch_size = train_snaps.shape[0]
    batch_size = None # only None is available for now.
    #batch_size = 20
    #batch_size =
    load_path = problem + args.net+'AE' + str(latent_dim) + DI_str + '_REC' + "{:.0e}".format(lambda_r_SAE) + '_JAC' + "{:.0e}".format( lambda_jac_SAE) + '_CON' + "{:.0e}".format(lambda_dx) + '_APP' + "{:.0e}".format(lambda_dz) + '_iter' + str(load_iterations)
    path = problem + args.net + AE_name    # net = torch.load('outputs/'+path+'/model_best.pkl')

    args2 = {
       # 'data': data,
        'net': net,
        # 'x_trunc': x_trunc,
        # 'latent_idx': latent_idx,
        'dt': dataset.dt,
        #'z_gt': dataset.z,
        'sys_name':'1DBurgers',
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
        'activation_SAE': 'relu',
        'lr_SAE': 1e-4,
        'miles_SAE': [1e9],
        'gamma_SAE': 1e-1,
        'lambda_r_SAE': lambda_r_SAE,
        'lambda_jac_SAE': lambda_jac_SAE,
        'lambda_dx': lambda_dx,
        'lambda_dz': lambda_dz,
        'path': path,
        'load_path': load_path,
        'batch_size': batch_size,
        'print_every': print_every,
        'save': True,
        'load':load_model,
        'callback': None,
        'dtype': dtype,
        'device': device,
        'tol': 1e-3,
        'tol2': 2,
        'adaptive': 'reg_max',
        'n_train_max': 25,
        'subset_size_max': 80,
        'trunc_period': trunc_period
    }

    ln.Brain_tLaSDI_greedy.Init(**args2)
    ln.Brain_tLaSDI_greedy.Run()
    ln.Brain_tLaSDI_greedy.Restore()
    ln.Brain_tLaSDI_greedy.Output()
    ln.Brain_tLaSDI_greedy.Test()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep learning of thermodynamics-aware reduced-order models from data')


    # # Dataset Parameters
    # parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    #

    # ## Sparse Autoencoder
    # # Net Parameters
    #parser.add_argument('--layer_vec_SAE', default=[100*4, 40*4,40*4, 10], nargs='+', type=int, help='full layer vector of the viscolastic SAE')
#     parser.add_argument('--layer_vec_SAE_q', default=[4140*3, 40, 40, 10], nargs='+', type=int, help='full layer vector (position) of the rolling tire SAE')
#     parser.add_argument('--layer_vec_SAE_v', default=[4140*3, 40, 40, 10], nargs='+', type=int, help='full layer vector (velocity) of the rolling tire SAE')
#     parser.add_argument('--layer_vec_SAE_sigma', default=[4140*6, 40*2, 40*2, 2*10], nargs='+', type=int, help='full layer vector (stress tensor) of the rolling tire SAE')
#     parser.add_argument('--activation_SAE', default='relu', type=str, help='activation function')

    #1DBurgers all data
#     parser.add_argument('--layer_vec_SAE', default=[101, 100, latent_dim], nargs='+', type=int, help='full layer vector of the viscolastic SAE')
    #1DBurgers half data
    #parser.add_argument('--layer_vec_SAE', default=[101, 100, 10], nargs='+', type=int, help='full layer vector of the BG SAE')


    #parser = argparse.ArgumentParser(description='Generic Neural Networks')
    #parser.add_argument('--net', default= DINN, type=str, help='ESP or ESP2 or ESP3')
    parser.add_argument('--lam', default=1, type=float, help='lambda as the weight for consistency penalty')
    #parser.add_argument('--seed2', default=0, type=int, help='random seed')
    
    
 
    parser.add_argument('--latent_dim', type=int, default=10,
                        help='Latent dimension.')

    parser.add_argument('--net', type=str, choices=["ESP3", "ESP3_soft"], default="ESP3",
                        help='ESP3 for GFINN and ESP3_soft for SPNN')

    parser.add_argument('--iterations', type=int, default=1000,
                        help='number of iterations')
    
    parser.add_argument('--load_iterations', type=int, default=1000,
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

    
    
    
    
    args = parser.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    main(args)


    #args = parser.parse_args()



