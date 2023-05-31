"""main.py"""

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





def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)



    latent_dim = 10
    device = 'cpu'  # 'cpu' or 'gpu'
    dtype = 'double'
    # data
    # p = 0.8
    problem = 'BG'
    AE_name = 'AE10Hgreedy_sim10000'
    # t_terminal = 40
    # dt = 0.1
    # trajs = 100
    order = 2
    iters = 1


    #print(data)
    # NN
    # layers = 5  #5 5   #5 5   5
    # width = 100  #24 198 #45 30  50

    depth_trunk = 3
    width_trunk = 40

    depth_hyper = 3
    width_hyper = 40

    depth_hyper2 = 2
    width_hyper2 = 20


    act_trunk = 'tanh'
    act_hyper = 'tanh'
    act_hyper2 = 'tanh'
    #activation = 'relu'
    num_sensor = 2 # dimension of parameters

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
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    #print(train_snaps.shape)
    #print #150 400
    # training
    lr = 1e-4  #1e-5 VC, 1e-5    0.001 good with relu, 1e-4 good with tanh
    #lr = 1e-3
    iterations = 10
    lbfgs_steps = 0
    print_every = 200
    #batch_size = train_snaps.shape[0]
    batch_size = None
    #batch_size = 20
    #batch_size =
    path = problem + args.net + str(args.lam) + str(latent_dim)+AE_name + '_' + str(args.seed)
    # net = torch.load('outputs/'+path+'/model_best.pkl')

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
        'layer_vec_SAE': args.layer_vec_SAE,
        'layer_vec_SAE_q': args.layer_vec_SAE_q,
        'layer_vec_SAE_v': args.layer_vec_SAE_v,
        'layer_vec_SAE_sigma': args.layer_vec_SAE_sigma,
        'activation_SAE': 'relu',
        'lr_SAE': 1e-4,
        'miles_SAE': [1e9],
        'gamma_SAE': 1e-1,
        'lambda_r_SAE': 1e-1,
        'batch_size': batch_size,
        'print_every': print_every,
        'save': True,
        'path': path,
        'callback': None,
        'dtype': dtype,
        'device': device,
        'tol': 1e-3,
        'tol2': 2,
        'adaptive': 'reg_max',
        'n_train_max': 25,
        'subset_size_max': 80
    }

    ln.Brain_hyper_sim_greedy.Init(**args2)
    ln.Brain_hyper_sim_greedy.Run()
    ln.Brain_hyper_sim_greedy.Restore()
    ln.Brain_hyper_sim_greedy.Output()
    ln.Brain_hyper_sim_greedy.Test()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep learning of thermodynamics-aware reduced-order models from data')


    # # Dataset Parameters
    # parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    #

    # ## Sparse Autoencoder
    # # Net Parameters
    #parser.add_argument('--layer_vec_SAE', default=[100*4, 40*4,40*4, 10], nargs='+', type=int, help='full layer vector of the viscolastic SAE')
    parser.add_argument('--layer_vec_SAE_q', default=[4140*3, 40, 40, 10], nargs='+', type=int, help='full layer vector (position) of the rolling tire SAE')
    parser.add_argument('--layer_vec_SAE_v', default=[4140*3, 40, 40, 10], nargs='+', type=int, help='full layer vector (velocity) of the rolling tire SAE')
    parser.add_argument('--layer_vec_SAE_sigma', default=[4140*6, 40*2, 40*2, 2*10], nargs='+', type=int, help='full layer vector (stress tensor) of the rolling tire SAE')
    parser.add_argument('--activation_SAE', default='relu', type=str, help='activation function')

    #1DBurgers all data
    parser.add_argument('--layer_vec_SAE', default=[101, 100, 10], nargs='+', type=int, help='full layer vector of the viscolastic SAE')
    #1DBurgers half data
    #parser.add_argument('--layer_vec_SAE', default=[501, 100, 10], nargs='+', type=int, help='full layer vector of the viscolastic SAE')

    # GFINNs
    #parser = argparse.ArgumentParser(description='Generic Neural Networks')
    parser.add_argument('--net', default='ESP3', type=str, help='ESP or ESP2 or ESP3')
    parser.add_argument('--lam', default=1, type=float, help='lambda as the weight for consistency penalty')
    #parser.add_argument('--seed2', default=0, type=int, help='random seed')
    args = parser.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    main(args)


    #args = parser.parse_args()



