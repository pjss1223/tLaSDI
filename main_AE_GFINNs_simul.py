"""main.py"""

import argparse


from nn_GFINNs import *
#from postprocess_dp import plot_DP
from learner.utils import grad
from AE_solver import AE_Solver
from dataset_sim import load_dataset, split_dataset


from utilities.utils import str2bool


def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)



    latent_dim = 10
    device = 'cpu'  # 'cpu' or 'gpu'
    #dtype = 'float'
    dtype = 'double'
    # data
    p = 0.8
    problem = 'VC'
    t_terminal = 40
    dt = 0.1
    trajs = 100
    order = 2
    iters = 1


    #print(data)
    # NN
    layers = 4  #5 5   #5 5   5
    width = 20  #24 198 #45 30  50
    activation = 'tanh'
    #activation = 'relu'
    dataset = load_dataset('viscoelastic','data',device)

    #train_snaps, test_snaps = split_dataset(dataset.z.shape[0] - 1)

    if args.net == 'ESP3':
        # netS = VC_LNN3(x_trunc.shape[1],5,layers=layers, width=width, activation=activation)
        # netE = VC_MNN3(x_trunc.shape[1],4,layers=layers, width=width, activation=activation)
        netS = VC_LNN3(latent_dim,10,layers=layers, width=width, activation=activation)
        netE = VC_MNN3(latent_dim,8,layers=layers, width=width, activation=activation)
        lam = 0
    elif args.net == 'ESP3_soft':
        netS = VC_LNN3_soft(latent_dim,layers=layers, width=width, activation=activation)
        netE = VC_MNN3_soft(latent_dim,layers=layers, width=width, activation=activation)
        lam = args.lam
    else:
        raise NotImplementedError

    #print(dataset.dt)  #0.006666666666666667
    net = ESPNN(netS, netE, dataset.dt / iters, order=order, iters=iters, lam=lam)

    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    #print(train_snaps.shape)
    #print #150 400
    # training
    lr = 1e-4  #1e-5 VC, 1e-5    0.001 good with relu, 1e-4 good with tanh
    iterations = 10000
    lbfgs_steps = 0
    print_every = 100
    #batch_size = train_snaps.shape[0]
    batch_size = None
    #batch_size = 20
    #batch_size =
    path = problem +args.net + str(args.lam) + '_' + str(args.seed)
    # net = torch.load('outputs/'+path+'/model_best.pkl')

    args2 = {
       # 'data': data,
        'net': net,
        # 'x_trunc': x_trunc,
        # 'latent_idx': latent_idx,
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
        'AE_name': 'AE2reg_sim50000',
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
        'device': device

    }

    ln.Brain_test_sim.Init(**args2)
    ln.Brain_test_sim.Run()
    ln.Brain_test_sim.Restore()
    ln.Brain_test_sim.Output()
    ln.Brain_test_sim.Test()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep learning of thermodynamics-aware reduced-order models from data')


    parser.add_argument('--seed', default=0, type=int, help='random seed')
    #

    # ## Sparse Autoencoder
    # # Net Parameters
    parser.add_argument('--layer_vec_SAE', default=[100*4, 40*4,40*4, 10], nargs='+', type=int, help='full layer vector of the viscolastic SAE')
    parser.add_argument('--layer_vec_SAE_q', default=[4140*3, 40, 40, 10], nargs='+', type=int, help='full layer vector (position) of the rolling tire SAE')
    parser.add_argument('--layer_vec_SAE_v', default=[4140*3, 40, 40, 10], nargs='+', type=int, help='full layer vector (velocity) of the rolling tire SAE')
    parser.add_argument('--layer_vec_SAE_sigma', default=[4140*6, 40*2, 40*2, 2*10], nargs='+', type=int, help='full layer vector (stress tensor) of the rolling tire SAE')
    parser.add_argument('--activation_SAE', default='relu', type=str, help='activation function')



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



