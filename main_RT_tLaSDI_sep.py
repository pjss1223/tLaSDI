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
dtype = 'float'

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
    


    sys_name = 'rolling_tire'

    # data
    p = 0.8
    problem = 'RT'
    t_terminal = 40
    dt = 0.1
    trajs = 100
    order = 2
    iters = 1 #fixed to be 1
    trunc_period = 80 # defined in args


    if args.net == 'ESP3':
        DI_str = 'sep'
    else:
        DI_str = 'soft_sep'



    #print(data)
    # NN
    layers = 5  #5   
    width = 198  #198 
    activation = args.activation
    #activation = 'relu'
    dataset = load_dataset('rolling_tire','data',device,dtype)
    
    
    extraD_L = args.extraD_L
    extraD_M = args.extraD_M
    xi_scale = args.xi_scale
    
        
    #-----------------------------------------------------------------------------
    latent_dim = args.latent_dim
    latent_dim_q = args.latent_dim_q
    latent_dim_v = args.latent_dim_v
    latent_dim_sigma = args.latent_dim_sigma
    iterations = args.iterations    iterations = args.iterations
    
    load_model = args.load_model
    load_iterations = args.load_iterations
    
    weight_decay_GFINNs = args.weight_decay_GFINNs
    
    
    lambda_r_SAE = args.lambda_r_SAE
    lambda_jac_SAE = args.lambda_jac_SAE
    lambda_dx = args.lambda_dx
    lambda_dz = args.lambda_dz
    layer_vec_SAE = [100*4, 40*4,40*4, latent_dim]
#     layer_vec_SAE_q = [4140*3, 40, 40, latent_dim]
#     layer_vec_SAE_v = [4140*3, 40, 40, latent_dim]
#     layer_vec_SAE_sigma = [4140*6, 40*2, 40*2, 2*latent_dim]
    
#     layer_vec_SAE_q = [2070*3, 40, 40, latent_dim]
#     layer_vec_SAE_v = [2070*3, 40, 40, latent_dim]
#     layer_vec_SAE_sigma = [2070*6, 40*2, 40*2, 2*latent_dim]
    
#     layer_vec_SAE_q = [1035*3, 40, 40, latent_dim]
#     layer_vec_SAE_v = [1035*3, 40, 40, latent_dim]
#     layer_vec_SAE_sigma = [1035*6, 40*2, 40*2, 2*latent_dim]
    layer_vec_SAE_q = [4140*3, 40, 40, latent_dim_q]
    layer_vec_SAE_v = [4140*3, 40, 40, latent_dim_v]
    layer_vec_SAE_sigma = [4140*6, 40*2, 40*2, latent_dim_sigma]
    #--------------------------------------------------------------------------------

    
    if args.load_model:
        AE_name = 'AE_'+ str(latent_dim_q)+'_'+ str(latent_dim_v)+'_'+ str(latent_dim_sigma) +DI_str+ str(extraD_L)+ '_REC'+"{:.0e}".format(lambda_r_SAE)  + '_JAC'+ "{:.0e}".format(lambda_jac_SAE) + '_CON'+"{:.0e}".format(lambda_dx) + '_APP' + "{:.0e}".format(lambda_dz)+ '_DEG' + "{:.0e}".format(lam)  + '_iter'+str(iterations+load_iterations)
    else:
        AE_name = 'AE_'+ str(latent_dim_q)+'_'+ str(latent_dim_v)+'_'+ str(latent_dim_sigma) +DI_str+ str(extraD_L)+ '_REC'+"{:.0e}".format(lambda_r_SAE)  + '_JAC'+ "{:.0e}".format(lambda_jac_SAE) + '_CON'+"{:.0e}".format(lambda_dx) + '_APP' + "{:.0e}".format(lambda_dz)+ '_DEG' + "{:.0e}".format(lam)  + '_iter'+str(iterations)

   
        
    AE_solver = AE_Solver_jac(args,AE_name,layer_vec_SAE,layer_vec_SAE_q,layer_vec_SAE_v,layer_vec_SAE_sigma)
    if args.train_SAE:
        AE_solver.train()
    AE_solver.test()
    
    

    if args.net == 'ESP3':
        # netS = VC_LNN3(x_trunc.shape[1],5,layers=layers, width=width, activation=activation)
        # netE = VC_MNN3(x_trunc.shape[1],4,layers=layers, width=width, activation=activation)
        netS = VC_LNN3(latent_dim_q+latent_dim_v+latent_dim_sigma,extraD_L,layers=layers, width=width, activation=activation,xi_scale=xi_scale)
        netE = VC_MNN3(latent_dim_q+latent_dim_v+latent_dim_sigma,extraD_M,layers=layers, width=width, activation=activation,xi_scale=xi_scale)
        lam = 0
    elif args.net == 'ESP3_soft':
        netS = VC_LNN3_soft(latent_dim_q+latent_dim_v+latent_dim_sigma,layers=layers, width=width, activation=activation)
        netE = VC_MNN3_soft(latent_dim_q+latent_dim_v+latent_dim_sigma,layers=layers, width=width, activation=activation)
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
    batch_size = None

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
        'weight_decay_GFINNs':weight_decay_GFINNs,
        'miles_lr': args.miles_lr,
        'gamma_lr':args.gamma_lr,
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

    parser.add_argument('--lam', default=1, type=float, help='lambda as the weight for consistency penalty')
    #parser.add_argument('--seed2', default=0, type=int, help='random seed')
    
    
    parser.add_argument('--latent_dim', type=int, default=10,
                        help='Latent dimension.')
    
    parser.add_argument('--latent_dim_q', type=int, default=4,
                        help='Latent dimension.')
    parser.add_argument('--latent_dim_v', type=int, default=3,
                        help='Latent dimension.')
    parser.add_argument('--latent_dim_sigma', type=int, default=2,
                        help='Latent dimension.')
    
    
    parser.add_argument('--extraD_L', type=int, default=7,
                        help='extraD for L.')
    parser.add_argument('--extraD_M', type=int, default=7,
                        help='extraD for M.')
    
    parser.add_argument('--xi_scale', type=float, default=1e-1,
                        help='scale for initialized skew-symmetric matrices')
    
    
    

    parser.add_argument('--net', type=str, choices=["ESP3", "ESP3_soft"], default="ESP3",
                        help='ESP3 for GFINN and ESP3_soft for SPNN')

    parser.add_argument('--iterations', type=int, default=10,
                        help='number of iterations')
    
    parser.add_argument('--load_iterations', type=int, default=10,
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


    parser.add_argument('--sys_name', default='rolling_tire', type=str, help='physic system name') #'viscoelastic''rolling_tire'
    parser.add_argument('--train_SAE', default=True, type=str2bool, help='SAE train or test')
    parser.add_argument('--device', default=device, type=str, help='device type')
    parser.add_argument('--dtype', default=dtype, type=str, help='data type')

    parser.add_argument('--trunc_period', default=80, type=int, help='trunc_period for jacobian')

    parser.add_argument('--weight_decay_GFINNs', type=float, default=0,
                        help='rate of learning rate decay for GFINNs')
    
    parser.add_argument('--weight_decay_AE', type=float, default=0,
                        help='rate of learning rate decay for AE')

    #------------------------------
    parser.add_argument('--max_epoch_SAE', default=20, type=float, help='maximum training iterations SAE')
    #-------------------------------

    parser.add_argument('--miles_lr',  type=int, default=[70000],
                        help='iteration steps for learning rate decay ')

    parser.add_argument('--gamma_lr', type=float, default=1e-1,
                        help='rate of learning rate decay.')

    # Dataset Parameters
    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')

    # Save options
    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')
    parser.add_argument('--save_plots', default=True, type=str2bool, help='save results in png file')
    
    
    
    args = parser.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    main(args)



    #args = parser.parse_args()



