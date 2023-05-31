#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 20:58:43 2021

double pendulum example

@author: zen
"""

import learner as ln
import torch
import numpy as np
import argparse
from data import Data
#from learner import Data
from nn import *
from postprocess_dp import plot_DP
from learner.utils import grad
    
def main(args):
    device = 'cpu' # 'cpu' or 'gpu'
    dtype = 'float'
    # data
    p = 0.8
    problem = 'DP'
    t_terminal = 40
    dt = 0.1
    trajs = 100
    order = 2
    iters = 1

    data = Data(p, problem, t_terminal, dt, trajs, order, iters, new = True)
    print(data)
    # NN
    layers = 5
    width = 30
    activation = 'tanh'

    if args.net == 'ESP':
        netS = DP_LNN(layers = layers, width = width, activation = activation)
        netE = DP_MNN(layers = layers, width = width, activation = activation)
        lam = 0
    elif args.net == 'ESP2':
        netS = DP_LNN2(layers = layers, width = width, activation = activation)
        netE = DP_MNN2(layers = layers, width = width, activation = activation)
        lam = 0
    elif args.net == 'ESP3':
        netS = DP_LNN3(layers = layers, width = width, activation = activation)
        netE = DP_MNN3(layers = layers, width = width, activation = activation)
        lam = 0
    elif args.net == 'ESP_soft':
        netS = DP_LNN_soft(layers = layers, width = width, activation = activation)
        netE = DP_MNN_soft(layers = layers, width = width, activation = activation)
        lam = args.lam
    elif args.net == 'ESP2_soft':
        netS = DP_LNN2_soft(layers = layers, width = width, activation = activation)
        netE = DP_MNN2_soft(layers = layers, width = width, activation = activation)
        lam = args.lam
    elif args.net == 'ESP3_soft':
        netS = DP_LNN3_soft(layers = layers, width = width, activation = activation)
        netE = DP_MNN3_soft(layers = layers, width = width, activation = activation)
        lam = args.lam
    else:
        raise NotImplementedError


    net = ESPNN(netS, netE, data.dt / iters, order = order, iters = iters, lam = lam)

    print(sum(p.numel() for p in net.parameters() if p.requires_grad))

    # training
    lr = 0.001
    iterations = 1000
    lbfgs_steps = 0
    print_every = 100
    batch_size = 100
    path = problem + args.net + str(args.lam) + '_' + str(args.seed)
    #net = torch.load('outputs/'+path+'/model_best.pkl')

    args2 = {
        'data': data,
        'net': net,
        'criterion': None,
        'optimizer': 'adam',
        'lr': lr,
        'iterations': iterations,
        'lbfgs_steps': lbfgs_steps,
        'batch_size': batch_size,
        'print_every': print_every,
        'save': True,
        'path': path,
        'callback': None,
        'dtype': dtype,
        'device': device,
    }


    ln.Brain.Init(**args2)
    ln.Brain.Run()
    ln.Brain.Restore()
    ln.Brain.Output()
    plot_DP(data, net, args)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generic Neural Networks')
    parser.add_argument('--net', default='ESP3', type=str, help='ESP or ESP2 or ESP3')
    parser.add_argument('--lam', default=1, type=float, help='lambda as the weight for consistency penalty')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    args = parser.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    main(args)
