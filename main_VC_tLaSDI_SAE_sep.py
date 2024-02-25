"""main.py"""

import argparse

import numpy as np
import torch
import learner as ln
from learner import data

from nn_GFINNs import *
from SAE_solver_jac import SAE_Solver_jac
from dataset_sim import load_dataset, split_dataset
from utilities.utils import str2bool


#------------------------------------------------- parameters changed frequently




def main(args):
    device = args.device  # 'cpu' or 'gpu'
    dtype = args.dtype
    batch_size = args.batch_size
    batch_size_AE = args.batch_size_AE

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    sys_name = 'viscoelastic'

    # data
    p = 0.8
    problem = 'VC'
#     t_terminal = 40
#     dt = 0.1
    trajs = 100
    order = 4
    iters = 1 #fixed to be 1
    trunc_period = 1
    data_type = args.data_type
    
    ROM_model = 'TA-ROM'

    if args.net == 'ESP3':
        DI_str = 'sep'
    else:
        DI_str = 'soft_sep'

    #print(data)
    # NN
    layers = args.layers  #4
    width = args.width  #20   #5 190 worked well
    
    AE_width1 = args.AE_width1
    AE_width2 = args.AE_width2
    
    dataset = load_dataset('viscoelastic','data',device,dtype)
    weight_decay_GFINNs = args.weight_decay_GFINNs

    activation = args.activation
    
    lr = args.lr
    lr_SAE = args.lr_SAE
    
    miles_lr = args.miles_lr
    gamma_lr = args.gamma_lr
    
    extraD_L = args.extraD_L
    extraD_M = args.extraD_M
    xi_scale = args.xi_scale
        
    #-----------------------------------------------------------------------------
    latent_dim_max = args.latent_dim
    iterations = args.iterations
    
    load_model = args.load_model
    load_iterations = args.load_iterations
    
    lambda_r_SAE = args.lambda_r_SAE
    lambda_jac_SAE = args.lambda_jac_SAE
    lambda_dx = args.lambda_dx
    lambda_dz = args.lambda_dz
    lambda_int = args.lambda_int
    
    layer_vec_SAE = [100*4, AE_width1, AE_width2, latent_dim_max]
    layer_vec_SAE_q = [4140*3, 40, 40, latent_dim_max]
    layer_vec_SAE_v = [4140*3, 40, 40, latent_dim_max]
    layer_vec_SAE_sigma = [4140*6, 40*2, 40*2, 2*latent_dim_max]
    #--------------------------------------------------------------------------------
    
    if args.load_model:
        AE_name = 'AE_SAE_sep'+ str(latent_dim_max) +DI_str+ '_REC'+"{:.0e}".format(lambda_r_SAE)  + '_JAC'+ "{:.0e}".format(lambda_jac_SAE) + '_CON'+"{:.0e}".format(lambda_dx) + '_APP' + "{:.0e}".format(lambda_dz)+ '_INT' + "{:.0e}".format(lambda_int) + '_Lr'+ "{:.0e}".format(lr)+ '_Lrae'+ "{:.0e}".format(lr_SAE)  + '_Gam'+ str(int(gamma_lr * 100))+ '_WDG'+ "{:.0e}".format(weight_decay_GFINNs)+ '_' +str(data_type) +'_'+str(seed)+'_iter'+str(iterations+load_iterations)
    else:
        AE_name = 'AE_SAE_sep'+ str(latent_dim_max) +DI_str+ '_REC'+"{:.0e}".format(lambda_r_SAE)  + '_JAC'+ "{:.0e}".format(lambda_jac_SAE) + '_CON'+"{:.0e}".format(lambda_dx) + '_APP' + "{:.0e}".format(lambda_dz)+ '_INT' + "{:.0e}".format(lambda_int) + '_Lr'+ "{:.0e}".format(lr)+ '_Lrae'+ "{:.0e}".format(lr_SAE)+ '_Gam'+ str(int(gamma_lr * 100))+ '_WDG'+ "{:.0e}".format(weight_decay_GFINNs) + '_' +str(data_type) +'_'+str(seed)+'_iter'+str(iterations)
   
        
    AE_solver = SAE_Solver_jac(args,AE_name,layer_vec_SAE,layer_vec_SAE_q,layer_vec_SAE_v,layer_vec_SAE_sigma)
    if args.train_SAE:
        AE_solver.train()
    AE_solver.test() 
    
    x_trunc, latent_idx = AE_solver.detect_dimensionality()
    
    _, latent_dim = x_trunc.shape
        

    if args.net == 'ESP3':

        netS = LNN(latent_dim,extraD_L,layers=layers, width=width, activation=activation,xi_scale=xi_scale)
        netE = MNN(latent_dim,extraD_M,layers=layers, width=width, activation=activation,xi_scale=xi_scale)
        lam = 0
    elif args.net == 'ESP3_soft':
        netS = LNN_soft(latent_dim,layers=layers, width=width, activation=activation)
        netE = MNN_soft(latent_dim,layers=layers, width=width, activation=activation)
        lam = args.lam
    else:
        raise NotImplementedError

    #print(dataset.dt)  #0.006666666666666667
    net = GFINNs(netS, netE, dataset.dt / iters, order=order, iters=iters, lam=lam)

    #print(sum(p.numel() for p in net.parameters() if p.requires_grad))

    # training
    #1e-5 VC, 1e-5    0.001 good with relu, 1e-4 good with tanh
    lbfgs_steps = 0
    print_every = 100
    

    load_path = problem + args.net+'AE_SAE_sep'+ str(latent_dim_max) +DI_str+ '_REC'+"{:.0e}".format(lambda_r_SAE)  + '_JAC'+ "{:.0e}".format(lambda_jac_SAE) + '_CON'+"{:.0e}".format(lambda_dx) + '_APP' + "{:.0e}".format(lambda_dz)+ '_INT' + "{:.0e}".format(lambda_int) + '_Lr'+ "{:.0e}".format(lr)+ '_Lrae'+ "{:.0e}".format(lr_SAE)+ '_Gam'+ str(int(gamma_lr * 100))+ '_WDG'+ "{:.0e}".format(weight_decay_GFINNs) + '_' +str(data_type) +'_'+str(seed)+'_iter'+str(load_iterations)
    
#     load_path = problem + args.net+'AE_SAE_sep' + str(latent_dim_max) + DI_str + '_REC' + "{:.0e}".format(lambda_r_SAE) + '_JAC' + "{:.0e}".format( lambda_jac_SAE) + '_CON' + "{:.0e}".format(lambda_dx) + '_APP' + "{:.0e}".format(lambda_dz)+ '_INT' + "{:.0e}".format(lambda_int)+ '_Lr'+ "{:.0e}".format(lr)+ '_Lrae'+ "{:.0e}".format(lr_SAE) + '_Gam'+str(int(gamma_lr * 100))+ '_WDG'+ "{:.0e}".format(weight_decay_GFINNs) +'_' +str(data_type)+'_'+str(seed)+ '_iter' + str(load_iterations)
    path = problem + args.net + AE_name       # net = torch.load('outputs/'+path+'/model_best.pkl')

    args2 = {
        'ROM_model':ROM_model,
        'AE': AE_solver.SAE,
        'net': net,
        'data_type':data_type,
        'x_trunc': x_trunc,
        'latent_idx': latent_idx,
        'latent_dim_max':latent_dim_max,
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
        'miles_lr': miles_lr,
        'gamma_lr': gamma_lr,
        'weight_decay_GFINNs': weight_decay_GFINNs,
        'lambda_int':lambda_int,
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

    ln.Brain_tLaSDI_SAE_sep.Init(**args2)
    ln.Brain_tLaSDI_SAE_sep.Run()
    ln.Brain_tLaSDI_SAE_sep.Restore()
    ln.Brain_tLaSDI_SAE_sep.Output()
    ln.Brain_tLaSDI_SAE_sep.Test()



if __name__ == "__main__":


    # Training Parameters


    # GFINNs
    #parser = argparse.ArgumentParser(description='Generic Neural Networks')
    parser = argparse.ArgumentParser(description='Deep learning of thermodynamics-aware reduced-order models from data')



    parser.add_argument('--seed', default=0, type=int, help='random seed')


    parser.add_argument('--lam', default=1e-2, type=float, help='lambda as the weight for consistency penalty')

    parser.add_argument('--extraD_L', type=int, default=7,help='extraD for L.')
    parser.add_argument('--extraD_M', type=int, default=7,help='extraD for M.')

    parser.add_argument('--xi_scale', type=float, default=1e-1,
                        help='scale for initialized skew-symmetric matrices')
    
    #####
    parser.add_argument('--device', type=str, choices=["gpu", "cpu"], default="gpu",
                        help='device used')
    
    parser.add_argument('--layers', type=int, default=5,
                        help='number of layers for GFINNs.')
    parser.add_argument('--width', type=int, default=100,
                        help='width of GFINNs.')
    
    parser.add_argument('--AE_width1', type=int, default=160,
                        help='first width for AE.')
    
    parser.add_argument('--AE_width2', type=int, default=160,
                        help='second width for AE.')
    
    parser.add_argument('--data_type', type=str, default="last",
                        help='Test data type')
    
    parser.add_argument('--miles_lr',  type=int, default= 1000,
                        help='iteration steps for learning rate decay ')

    parser.add_argument('--gamma_lr', type=float, default=1,
                        help='rate of learning rate decay.')
    
    #####
    
    parser.add_argument('--latent_dim', type=int, default=8,
                        help='Latent dimension.')

    parser.add_argument('--net', type=str, choices=["ESP3", "ESP3_soft"], default="ESP3_soft",
                        help='ESP3 for GFINN and ESP3_soft for SPNN')

    parser.add_argument('--iterations', type=int, default=0,
                        help='number of iterations')
    
    parser.add_argument('--load_iterations', type=int, default=40011,
                        help='number of iterations of loaded network')

    parser.add_argument('--lambda_r_SAE', type=float, default=1,
                        help='Penalty for reconstruction loss, AE part')
    
    parser.add_argument('--lambda_r_sparse', type=float, default=1e-4,
                        help='Penalty for sparsity loss, AE part')
    
    parser.add_argument('--lambda_int', type=float, default=1e3,
                        help='Penalty for sparsity loss, AE part')

    parser.add_argument('--lambda_jac_SAE', type=float, default=0,
                        help='Penalty for Jacobian loss, AE part')
    
    parser.add_argument('--weight_decay_AE', type=float, default=0,
                        help='weight decay for AE')
    parser.add_argument('--weight_decay_GFINNs', type=float, default=1e-5,
                        help='weight decay for GFINNs')

    parser.add_argument('--lambda_dx', type=float, default=0,
                        help='Penalty for Consistency loss.')

    parser.add_argument('--lambda_dz', type=float, default=0,
                        help='Penalty for Model approximation loss.')
    
    parser.add_argument('--load_model', default=True, type=str2bool, 
                        help='load previously trained model')
    

    
    parser.add_argument('--activation', type=str, choices=["tanh", "relu","linear","sin","gelu"], default="tanh",
                        help='ESP3 for GFINN and ESP3_soft for SPNN')
    

    parser.add_argument('--activation_SAE', default='relu', type=str, help='activation function')
    parser.add_argument('--lr_SAE', default=1e-4, type=float, help='learning rate SAE')#1e-4 VC, #1e-4 RT
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate SPNN')#1e-4 VC, #1e-4 RT

    parser.add_argument('--miles_SAE', default=1000, nargs='+', type=int, help='learning rate scheduler milestones SAE')
    parser.add_argument('--gamma_SAE', default=1, type=float, help='learning rate milestone decay SAE')


    parser.add_argument('--sys_name', default='viscoelastic', type=str, help='physic system name') #'viscoelastic''rolling_tire'
    parser.add_argument('--train_SAE', default=True, type=str2bool, help='SAE train or test')
    parser.add_argument('--dtype', default="double", type=str, help='data type')
    parser.add_argument('--trunc_period', default=1, type=int, help='trunc_period for jacobian')



    #------------------------------
    parser.add_argument('--max_epoch_SAE', default=10, type=float, help='maximum training iterations SAE')
    #-------------------------------



    # Dataset Parameters
    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')

    # Save options
    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')
    parser.add_argument('--save_plots', default=True, type=str2bool, help='save results in png file')
    parser.add_argument('--batch_size_AE', default=None, type=float, help='batch size for AE')
    parser.add_argument('--batch_size', default=None, type=float, help='batch size for AE')

    
    
    args = parser.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    main(args)



    #args = parser.parse_args()



