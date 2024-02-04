"""main.py"""


#1D Burgers
import argparse

from nn_GFINNs import *

from dataset_sim_hyper import load_dataset
from utilities.utils import str2bool





def main(args):

    load_epochs = args.load_epochs
    load_model = args.load_model  # load model with exactly same set up

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)



    problem = 'BG'
    
    device = args.device  # 'cpu' or 'gpu'
    dtype = args.dtype


    order = args.order
    iters = 1
    trunc_period = args.trunc_period


    layers = args.layers  #GFINNs structure
    width = args.width

    depth_hyper = args.depth_hyper   
    width_hyper = args.width_hyper



    activation = 'tanh' #GFINNs activation func
    act_hyper = 'tanh'
    num_sensor = 2 # dimension of parameters
    
    lbfgs_steps = 0
    batch_num = None # not necessarily defined 
    print_every = 200 # this means that batch size = int(z_gt_tr.shape[0]/batch_num)
    batch_size = args.batch_size # 1~N_t
    
    update_epochs = args.update_epochs


    if args.net == 'GFINNs':
        DI_str = ''
    else:
        DI_str = 'soft'
        
        
    #-----------------------------------------------------------------------------
    latent_dim = args.latent_dim
    epochs = args.epochs
    extraD_L = args.extraD_L
    extraD_M = args.extraD_M
    xi_scale = args.xi_scale
    
    load_model = args.load_model
    load_epochs = args.load_epochs
    
    gamma_lr = args.gamma_lr
    miles_lr = args.miles_lr
    
    weight_decay_GFINNs = args.weight_decay_GFINNs
    weight_decay_AE = args.weight_decay_AE
    
    lambda_r_AE = args.lambda_r_AE
    lambda_jac_AE = args.lambda_jac_AE
    lambda_dx = args.lambda_dx
    lambda_dz = args.lambda_dz
    layer_vec_AE = [201,100,latent_dim]
    #--------------------------------------------------------------------------------



    if load_model:
        AE_name = 'AE_hyper'+ str(latent_dim)+'_extraD_'+str( extraD_L) +DI_str+ '_REC'+"{:.0e}".format(lambda_r_AE)  + '_JAC'+ "{:.0e}".format(lambda_jac_AE) + '_CON'+"{:.0e}".format(lambda_dx) + '_APP' + "{:.0e}".format(lambda_dz) + '_od'+ str(order)+  '_iter'+str(epochs+load_epochs)
    else:
        AE_name = 'AE_hyper'+ str(latent_dim)+'_extraD_'+str( extraD_L) +DI_str+ '_REC'+"{:.0e}".format(lambda_r_AE)  + '_JAC'+ "{:.0e}".format(lambda_jac_AE) + '_CON'+"{:.0e}".format(lambda_dx) + '_APP' + "{:.0e}".format(lambda_dz) + '_od'+ str(order)+    '_iter'+str(epochs)



    load_path = problem + args.net + 'AE_hyper'+ str(latent_dim)+'_extraD_'+str( extraD_L) +DI_str+ '_REC'+"{:.0e}".format(lambda_r_AE)  + '_JAC'+ "{:.0e}".format(lambda_jac_AE) + '_CON'+"{:.0e}".format(lambda_dx) + '_APP' + "{:.0e}".format(lambda_dz)+ '_od'+ str(order)  + '_iter'+str(load_epochs)
    

    
    path = problem + args.net + AE_name    # net = torch.load('outputs/'+path+'/model_best.pkl')



    dataset = load_dataset('1DBurgers','data',device,dtype)


    if args.net == 'GFINNs':
        netS = LNN(latent_dim,extraD_L,layers=layers, width=width, activation=activation, xi_scale=xi_scale)
        netE = MNN(latent_dim,extraD_M,layers=layers, width=width, activation=activation, xi_scale=xi_scale)
        lam = 0
    elif args.net == 'SPNN':
        netS = LNN_soft(latent_dim,layers=layers, width=width, activation=activation)
        netE = MNN_soft(latent_dim,layers=layers, width=width, activation=activation)
        lam = args.lam
    else:
        raise NotImplementedError

    net = GFINNs(netS, netE, dataset.dt / iters, order=order, iters=iters, lam=lam)

    #print(sum(p.numel() for p in net.parameters() if p.requires_grad))


    # training
    lr = 1e-4  #1e-5 VC, 1e-5    0.001 good with relu, 1e-4 good with tanh


    args2 = {
        'net': net,
        'sys_name':'1DBurgers',
        'output_dir': 'outputs',
        'save_plots': True,
        'criterion': None,
        'optimizer': 'adam',
        'lr': lr,
        'epochs': epochs,
        'AE_name': AE_name,
        'dset_dir': 'data',
        'output_dir_AE': 'outputs',
        'save_plots_AE': True,
        'layer_vec_AE': layer_vec_AE,
        'activation_AE': 'relu',
        'depth_hyper': depth_hyper,
        'width_hyper': width_hyper,
        'act_hyper': act_hyper,
        'num_sensor': num_sensor,
        'lr_AE': 1e-4,
        'lambda_r_AE': lambda_r_AE,
        'lambda_jac_AE': lambda_jac_AE,
        'lambda_dx': lambda_dx,
        'lambda_dz': lambda_dz,
        'miles_lr': miles_lr,
        'gamma_lr': gamma_lr,
        'weight_decay':weight_decay_GFINNs,
        'weight_decay_AE':weight_decay_AE,
        'path': path,
        'load_path': load_path,
        'batch_size': batch_size,
        'update_epochs':update_epochs,
        'print_every': print_every,
        'save': True,
        'load': load_model,
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

    ln.Brain_tLaSDI_GAEhyper.Init(**args2)

    ln.Brain_tLaSDI_GAEhyper.Run()

    ln.Brain_tLaSDI_GAEhyper.Restore()

    ln.Brain_tLaSDI_GAEhyper.Output()

    ln.Brain_tLaSDI_GAEhyper.Test()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep learning of thermodynamics-aware reduced-order models from data')


    # # Dataset Parameters
    # parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
  

    parser.add_argument('--lam', default=1, type=float, help='lambda as the weight for consistency penalty')
    
    parser.add_argument('--extraD_L', type=int, default=9,
                        help='extraD for L.')
    parser.add_argument('--extraD_M', type=int, default=9,
                        help='extraD for M.')

 
    parser.add_argument('--latent_dim', type=int, default=10,
                        help='Latent dimension.')

    parser.add_argument('--net', type=str, choices=["GFINNs", "SPNN"], default="GFINNs",
                        help='DI model choices')

    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs')
    
    parser.add_argument('--load_epochs', type=int, default=43901,
                        help='number of epochs of loaded network')

    parser.add_argument('--lambda_r_AE', type=float, default=1e-1,
                        help='Penalty for reconstruction loss.')

    parser.add_argument('--lambda_jac_AE', type=float, default=1e-9,
                        help='Penalty for Jacobian loss.')

    parser.add_argument('--lambda_dx', type=float, default=1e-7,
                        help='Penalty for Consistency loss.')

    parser.add_argument('--lambda_dz', type=float, default=1e-7,
                        help='Penalty for Model approximation loss.')
    
    parser.add_argument('--load_model', default=False, type=str2bool, 
                        help='load previously trained model')
    
    parser.add_argument('--miles_lr',  type=int, default=1000,
                        help='epoch steps for learning rate decay ')

    parser.add_argument('--gamma_lr', type=float, default=.99,
                        help='rate of learning rate decay.')
    
    
    
    parser.add_argument('--weight_decay_GFINNs', type=float, default=0,
                        help='rate of learning rate decay for GFINNs.')
    
    parser.add_argument('--weight_decay_AE', type=float, default=0,
                        help='rate of learning rate decay for AE.')
    
    
    
    parser.add_argument('--device', type=str, choices=["gpu", "cpu"], default="gpu",
                        help='deviced used')
    
    parser.add_argument('--dtype', type=str, choices=["float", "double"], default="float",
                        help='data type used')
    
    
    parser.add_argument('--batch_size', default=50, type=int, help='batch size for  GFINNs')


    parser.add_argument('--layers', type=int, default=5,
                        help='layers for GFINNs.')
    parser.add_argument('--width', type=int, default=40,
                        help='width for GFINNs.')
    parser.add_argument('--depth_hyper', type=int, default=3,
                        help='depth for hypernet.')
    parser.add_argument('--width_hyper', type=int, default=20,
                        help='width for hypernet.')
    
    parser.add_argument('--activation', default='tanh', type=str, help='activation function for GFINNs')
    parser.add_argument('--act_hyper', default='tanh', type=str, help='activation function for hypernet')
    parser.add_argument('--update_epochs', type=int, default=1000,
                        help='update epochs for greeedy sampling')
    parser.add_argument('--order', type=int, default=1,
                        help='order for integrator')
    parser.add_argument('--xi_scale', type=float, default=.3333,
                        help='scale for initialized skew-symmetric matrices')
    parser.add_argument('--trunc_period', type=int, default=2,
                        help='truncate indices for Jacobian computations')
    
    
#     parser.add_argument('--data_type', type=str, choices=["para10", "para13", "para21"], default="para21",
#                         help='number of parameters in data')
    
        
    
    
    args = parser.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    main(args)


    #args = parser.parse_args()



