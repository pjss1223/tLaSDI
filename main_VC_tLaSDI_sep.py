"""main.py"""

import argparse

import numpy as np
import torch
import learner as ln
from learner import data


from nn_GFINNs import *
from AE_solver_jac import AE_Solver_jac
from dataset_sim import load_dataset, split_dataset
from utilities.utils import str2bool



device = 'gpu'  # 'cpu' or 'gpu'
dtype = 'double'
batch_size = None
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
    


    sys_name = 'viscoelastic'

    # data
    p = 0.8
    problem = 'VC'
    t_terminal = 40
    dt = 0.1
    trajs = 100
    order = 2
    iters = 1 #fixed to be 1
    trunc_period = 1


    if args.net == 'ESP3':
        DI_str = 'sep'
    else:
        DI_str = 'soft_sep'



    #print(data)
    # NN
    layers = 4  #5 5   #5 5   5
    width = 20  #24 198 #45 30  50
#     activation = 'tanh'
    #activation = 'relu'
    dataset = load_dataset('viscoelastic','data',device,dtype)
    
    activation = args.activation
    
        
    #-----------------------------------------------------------------------------
    latent_dim = args.latent_dim
    iterations = args.iterations
    
    load_model = args.load_model
    load_iterations = args.load_iterations
    
    
    lambda_r_SAE = args.lambda_r_SAE
    lambda_jac_SAE = args.lambda_jac_SAE
    lambda_dx = args.lambda_dx
    lambda_dz = args.lambda_dz
    layer_vec_SAE = [100*4, 40*4,40*4, latent_dim]
    layer_vec_SAE_q = [4140*3, 40, 40, latent_dim]
    layer_vec_SAE_v = [4140*3, 40, 40, latent_dim]
    layer_vec_SAE_sigma = [4140*6, 40*2, 40*2, 2*latent_dim]
    #--------------------------------------------------------------------------------

    
    if args.load_model:
        AE_name = 'AE'+ str(latent_dim) +DI_str+ '_REC'+"{:.0e}".format(lambda_r_SAE)  + '_JAC'+ "{:.0e}".format(lambda_jac_SAE) + '_CON'+"{:.0e}".format(lambda_dx) + '_APP' + "{:.0e}".format(lambda_dz) + '_iter'+str(iterations+load_iterations)
    else:
        AE_name = 'AE'+ str(latent_dim) +DI_str+ '_REC'+"{:.0e}".format(lambda_r_SAE)  + '_JAC'+ "{:.0e}".format(lambda_jac_SAE) + '_CON'+"{:.0e}".format(lambda_dx) + '_APP' + "{:.0e}".format(lambda_dz) + '_iter'+str(iterations)

   
        
    AE_solver = AE_Solver_jac(args,AE_name,layer_vec_SAE,layer_vec_SAE_q,layer_vec_SAE_v,layer_vec_SAE_sigma)
    if args.train_SAE:
        AE_solver.train()
    AE_solver.test()
    
    

    if args.net == 'ESP3':
        # netS = VC_LNN3(x_trunc.shape[1],5,layers=layers, width=width, activation=activation)
        # netE = VC_MNN3(x_trunc.shape[1],4,layers=layers, width=width, activation=activation)
        netS = VC_LNN3(latent_dim,extraD_L,layers=layers, width=width, activation=activation)
        netE = VC_MNN3(latent_dim,extraD_M,layers=layers, width=width, activation=activation)
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
    lr = 1e-4 #1e-5 VC, 1e-5    0.001 good with relu, 1e-4 good with tanh
    lbfgs_steps = 0
    print_every = 100
    

    load_path = problem + args.net+'AE' + str(latent_dim) + DI_str + '_REC' + "{:.0e}".format(lambda_r_SAE) + '_JAC' + "{:.0e}".format( lambda_jac_SAE) + '_CON' + "{:.0e}".format(lambda_dx) + '_APP' + "{:.0e}".format(lambda_dz) + '_iter' + str(load_iterations)
    path = problem + args.net + AE_name       # net = torch.load('outputs/'+path+'/model_best.pkl')

    args2 = {
        'AE': AE_solver.SAE,
        'net': net,
        # 'x_trunc': x_trunc,
        # 'latent_idx': latent_idx,
        'dt': dataset.dt,
        'z_gt': dataset.z,
        'sys_name': sys_name,
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
#         'layer_vec_SAE': layer_vec_SAE,
#         'layer_vec_SAE_q': layer_vec_SAE_q,
#         'layer_vec_SAE_v': layer_vec_SAE_v,
#         'layer_vec_SAE_sigma': layer_vec_SAE_sigma,
        'lambda_dx':lambda_dx,
        'lambda_dz':lambda_dz,
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

    ln.Brain_tLaSDI_sep.Init(**args2)
    ln.Brain_tLaSDI_sep.Run()
    ln.Brain_tLaSDI_sep.Restore()
    ln.Brain_tLaSDI_sep.Output()
    ln.Brain_tLaSDI_sep.Test()



if __name__ == "__main__":


    # Training Parameters


    # GFINNs
    #parser = argparse.ArgumentParser(description='Generic Neural Networks')
    parser = argparse.ArgumentParser(description='Deep learning of thermodynamics-aware reduced-order models from data')


    # # Dataset Parameters
    # parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    #parser.add_argument('--lambda_jac_SAE', default=5e2, type=float, help='Jacobian (regularization) weight SAE')#1e-4 VC, 1e-2 RT
    parser.add_argument('--seed', default=0, type=int, help='random seed')

    # GFINNs

    parser.add_argument('--lam', default=1e-2, type=float, help='lambda as the weight for consistency penalty')
    #parser.add_argument('--seed2', default=0, type=int, help='random seed')
    

    
    parser.add_argument('--latent_dim', type=int, default=10,
                        help='Latent dimension.')

    parser.add_argument('--net', type=str, choices=["ESP3", "ESP3_soft"], default="ESP3",
                        help='ESP3 for GFINN and ESP3_soft for SPNN')

    parser.add_argument('--iterations', type=int, default=100,
                        help='number of iterations')
    
    parser.add_argument('--load_iterations', type=int, default=1000,
                        help='number of iterations of loaded network')

    parser.add_argument('--lambda_r_SAE', type=float, default=1e-1,
                        help='Penalty for reconstruction loss, AE part')

    parser.add_argument('--lambda_jac_SAE', type=float, default=1e-6,
                        help='Penalty for Jacobian loss, AE part')

    parser.add_argument('--lambda_dx', type=float, default=1e-4,
                        help='Penalty for Consistency loss.')

    parser.add_argument('--lambda_dz', type=float, default=1e-4,
                        help='Penalty for Model approximation loss.')
    
    parser.add_argument('--load_model', default=False, type=str2bool, 
                        help='load previously trained model')
    
#     parser.add_argument('--layer_vec_SAE', default=[100*4, 40*4,40*4, latent_dim], nargs='+', type=int, help='full layer vector of the viscolastic SAE')
#     parser.add_argument('--layer_vec_SAE_q', default=[4140*3, 40, 40, 10], nargs='+', type=int, help='full layer vector (position) of the rolling tire SAE')
#     parser.add_argument('--layer_vec_SAE_v', default=[4140*3, 40, 40, 10], nargs='+', type=int, help='full layer vector (velocity) of the rolling tire SAE')
#     parser.add_argument('--layer_vec_SAE_sigma', default=[4140*6, 40*2, 40*2, 2*10], nargs='+', type=int, help='full layer vector (stress tensor) of the rolling tire SAE')
    
    parser.add_argument('--activation', type=str, choices=["tanh", "relu","linear","sin","gelu"], default="tanh",
                        help='ESP3 for GFINN and ESP3_soft for SPNN')
    

    parser.add_argument('--activation_SAE', default='relu', type=str, help='activation function')
    parser.add_argument('--lr_SAE', default=1e-4, type=float, help='learning rate SAE')#1e-4 VC, #1e-4 RT
    parser.add_argument('--miles_SAE', default=[1e9], nargs='+', type=int, help='learning rate scheduler milestones SAE')
    parser.add_argument('--gamma_SAE', default=1e-1, type=float, help='learning rate milestone decay SAE')


    parser.add_argument('--sys_name', default='viscoelastic', type=str, help='physic system name') #'viscoelastic''rolling_tire'
    parser.add_argument('--train_SAE', default=True, type=str2bool, help='SAE train or test')
    parser.add_argument('--device', default='gpu', type=str, help='device type')
    parser.add_argument('--dtype', default=dtype, type=str, help='data type')
    parser.add_argument('--trunc_period', default=1, type=int, help='trunc_period for jacobian')



    #------------------------------
    parser.add_argument('--max_epoch_SAE', default=200, type=float, help='maximum training iterations SAE')
    #-------------------------------



    # Dataset Parameters
    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')

    # Save options
    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')
    parser.add_argument('--save_plots', default=True, type=str2bool, help='save results in png file')
    parser.add_argument('--batch_size_AE', default=batch_size, type=float, help='batch size for AE')
    
    
    args = parser.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    main(args)



    #args = parser.parse_args()



