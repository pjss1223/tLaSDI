"""main.py"""

import argparse



from nn_FNN import *
from dataset_sim import load_dataset, split_dataset
from utilities.utils import str2bool





def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    

    device = args.device  # 'cpu' or 'gpu'
    dtype = 'double'



    # data
    p = 0.8
    problem = 'VC'
    t_terminal = 40
    dt = 0.1
    trajs = 100
    order = 4
    iters = 1 #fixed to be 1
    trunc_period = 1
    
    data_type = args.data_type

    ROM_model = 'Vanilla-FNN'

    DI_str = 'FNN'

    #print(data)
    # NN

    activation = args.activation
    activation_AE = args.activation_AE
    #activation = 'relu'
    dataset = load_dataset('viscoelastic','data',device,dtype)
    miles_lr = args.miles_lr
    gamma_lr = args.gamma_lr
    
    weight_decay_AE = args.weight_decay_AE
    weight_decay_GFINNs = args.weight_decay_GFINNs 
    
    layers = args.layers  
    width = args.width  
    
    AE_width1 = args.AE_width1
    AE_width2 = args.AE_width2
        
    #-----------------------------------------------------------------------------
    latent_dim = args.latent_dim
    iterations = args.iterations
    
    load_model = args.load_model
    load_iterations = args.load_iterations
    
    
    lambda_r_AE = args.lambda_r_AE
    lambda_jac_AE = args.lambda_jac_AE
    lambda_dx = args.lambda_dx
    lambda_dz = args.lambda_dz
    layer_vec_AE = [100*4, AE_width1, AE_width2, latent_dim]

    layer_vec_AE_q = [4140*3, 40, 40, latent_dim]
    layer_vec_AE_v = [4140*3, 40, 40, latent_dim]
    layer_vec_AE_sigma = [4140*6, 40*2, 40*2, 2*latent_dim]
    #--------------------------------------------------------------------------------
    
    
    if args.load_model:
        AE_name = 'AE'+ str(latent_dim)+'_width_'+str(width)+DI_str+ '_REC'+"{:.0e}".format(lambda_r_AE)  + '_JAC'+ "{:.0e}".format(lambda_jac_AE) + '_CON'+"{:.0e}".format(lambda_dx) + '_APP' + "{:.0e}".format(lambda_dz) + '_iter'+str(iterations+load_iterations)
    else:
        AE_name = 'AE'+ str(latent_dim)+'_width_'+str(width) +DI_str+ '_REC'+"{:.0e}".format(lambda_r_AE)  + '_JAC'+ "{:.0e}".format(lambda_jac_AE) + '_CON'+"{:.0e}".format(lambda_dx) + '_APP' + "{:.0e}".format(lambda_dz) + '_iter'+str(iterations)

        
   
#     if args.load_model:
#         AE_name = 'AE'+ str(latent_dim)+'_extraD_'+str(extraD_L) +DI_str+ '_REC'+"{:.0e}".format(lambda_r_AE)  + '_JAC'+ "{:.0e}".format(lambda_jac_AE) + '_CON'+"{:.0e}".format(lambda_dx) + '_APP' + "{:.0e}".format(lambda_dz)+ '_DEG' + "{:.0e}".format(lam)+activation+activation_AE+ '_Gam'+ str(int(gamma_lr * 100))+ '_WDG'+ "{:.0e}".format(weight_decay_GFINNs)+ '_WDA'+ "{:.0e}".format(weight_decay_AE)+'_' +str(data_type) +'_'+str(seed) + '_iter'+str(iterations+load_iterations)

#     else:
#         AE_name = 'AE'+ str(latent_dim)+'_extraD_'+str(extraD_L) +DI_str+ '_REC'+"{:.0e}".format(lambda_r_AE)  + '_JAC'+ "{:.0e}".format(lambda_jac_AE) + '_CON'+"{:.0e}".format(lambda_dx) + '_APP' + "{:.0e}".format(lambda_dz)+ '_DEG' + "{:.0e}".format(lam)+activation +activation_AE+ '_Gam'+ str(int(gamma_lr * 100))+ '_WDG'+ "{:.0e}".format(weight_decay_GFINNs)+'_WDA'+ "{:.0e}".format(weight_decay_AE)+'_' +str(data_type) +'_'+str(seed)+ '_iter'+str(iterations)


    load_path =  problem + args.net +'AE'+ str(latent_dim)+'_width_'+str(width) +DI_str+ '_REC'+"{:.0e}".format(lambda_r_AE)  + '_JAC'+ "{:.0e}".format(lambda_jac_AE) + '_CON'+"{:.0e}".format(lambda_dx) + '_APP' + "{:.0e}".format(lambda_dz) + '_iter'+str(load_iterations)



    net = FNN_latent(latent_dim, dataset.dt, layers=layers, width=width, order=order, iters=iters,
                 activation=activation)
    
    
    
    
    # training
    lr = args.lr #1e-5 VC, 1e-5    0.001 good with relu, 1e-4 good with tanh
    lbfgs_steps = 0
    print_every = 100
    batch_size = None

#     load_path = problem + args.net+'AE' + str(latent_dim) + DI_str + '_REC' + "{:.0e}".format(lambda_r_AE) + '_JAC' + "{:.0e}".format( lambda_jac_AE) + '_CON' + "{:.0e}".format(lambda_dx) + '_APP' + "{:.0e}".format(lambda_dz) + '_iter' + str(load_iterations)
    
    path = problem + args.net + AE_name       # net = torch.load('outputs/'+path+'/model_best.pkl')

    args2 = {
        'ROM_model':ROM_model,
        'net': net,
        # 'x_trunc': x_trunc,
        # 'latent_idx': latent_idx,
        'data_type':data_type,
        'dt': dataset.dt,
        'z_gt': dataset.z,
        'sys_name':'viscoelastic',
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
        'layer_vec_AE': layer_vec_AE,
        'layer_vec_AE_q': layer_vec_AE_q,
        'layer_vec_AE_v': layer_vec_AE_v,
        'layer_vec_AE_sigma': layer_vec_AE_sigma,
        'activation_AE': activation_AE,
        'lr_AE': 1e-4,
#         'miles_AE': [1e9],
#         'gamma_AE': 1e-1,
        'miles_lr': miles_lr, 
        'gamma_lr': gamma_lr,
        'weight_decay_AE':weight_decay_AE,
        'weight_decay_GFINNs':weight_decay_GFINNs,
        'lambda_r_AE': lambda_r_AE,
        'lambda_jac_AE': lambda_jac_AE,
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

    ln.Brain_FNN.Init(**args2)
    ln.Brain_FNN.Run()
    ln.Brain_FNN.Restore()
    ln.Brain_FNN.Output()
    ln.Brain_FNN.Test()



if __name__ == "__main__":

    # GFINNs
    #parser = argparse.ArgumentParser(description='Generic Neural Networks')
    parser = argparse.ArgumentParser(description='Deep learning of thermodynamics-aware reduced-order models from data')


    # # Dataset Parameters
    # parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    #parser.add_argument('--lambda_jac_AE', default=5e2, type=float, help='Jacobian (regularization) weight AE')#1e-4 VC, 1e-2 RT
    parser.add_argument('--seed', default=0, type=int, help='random seed')


    
    
    parser.add_argument('--latent_dim', type=int, default=8,
                        help='Latent dimension.')

    parser.add_argument('--net', type=str, choices=["ESP3", "ESP3_soft", "FNN"], default="FNN",
                        help='ESP3 for GFINN and ESP3_soft for SPNN')

    parser.add_argument('--iterations', type=int, default=0,
                        help='number of iterations')
    
    parser.add_argument('--load_iterations', type=int, default=40011,  #40011
                        help='number of iterations of loaded network')

    parser.add_argument('--lambda_r_AE', type=float, default=1e-1,
                        help='Penalty for reconstruction loss.')

    parser.add_argument('--lambda_jac_AE', type=float, default=0,
                        help='Penalty for Jacobian loss.')

    parser.add_argument('--lambda_dx', type=float, default=0,
                        help='Penalty for Consistency loss.')

    parser.add_argument('--lambda_dz', type=float, default=0,
                        help='Penalty for Model approximation loss.')
    
    parser.add_argument('--load_model', default=True, type=str2bool, 
                        help='load previously trained model')
    
    
    parser.add_argument('--layers', type=int, default=5,
                        help='number of layers for GFINNs.')
    parser.add_argument('--width', type=int, default=215,
                        help='width of GFINNs.')
    
    parser.add_argument('--AE_width1', type=int, default=160,
                        help='first width for AE.')
    
    parser.add_argument('--AE_width2', type=int, default=160,
                        help='second width for AE.')
    
    
    parser.add_argument('--activation', type=str, choices=["tanh", "relu","linear","sin","gelu"], default="tanh",
                        help='activation functions for GFINNs or SPNN')
    
    parser.add_argument('--device', type=str, choices=["gpu", "cpu"], default="gpu",
                        help='device used')
    
    parser.add_argument('--activation_AE', type=str, choices=["tanh", "relu","linear","sin","gelu"], default="relu",
                        help='activation for AE')
    

    
    parser.add_argument('--data_type', type=str,choices=["last","middle"], default="last",
                        help='Test data type')
    
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='rate of learning rate decay.')
    
    parser.add_argument('--miles_lr',  type=int, default= 1000,
                        help='iteration steps for learning rate decay ')

    parser.add_argument('--gamma_lr', type=float, default=.99,
                        help='rate of learning rate decay.')
    
    parser.add_argument('--weight_decay_AE', type=float, default=0,
                        help='weight decay for AE')
    parser.add_argument('--weight_decay_GFINNs', type=float, default=1e-8,
                        help='weight decay for GFINNs')
    
    
    args = parser.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    main(args)



    #args = parser.parse_args()



