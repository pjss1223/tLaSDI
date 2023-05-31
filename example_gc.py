#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 20:59:32 2021

gas container example

@author: zen
"""

import learner as ln
import torch
import numpy as np
import argparse

from data import Data
from nn import *
from postprocess_gc import plot_GC
        
def main(args):
    device = 'cpu' # 'cpu' or 'gpu'
    dtype = 'float'
    # data
    p = 0.8
    problem = 'GC'
    t_terminal = 8
    dt = 0.02
    iters = 1
    trajs = 100
    order = 2
    
    data = Data(p, problem, t_terminal, dt, trajs, order, iters, new = True, noise = 0)
    # NN
    ind = 4
    layers = 5
    width  = 30
    activation = 'tanh'

    batch_size = 100
    lr = 0.001

    if args.net == 'ESP':
        netS = GC_LNN(layers = layers, width = width, activation = activation)
        netE = GC_MNN(layers = layers, width = width, activation = activation)
        net = ESPNN(netS, netE, data.dt / iters, order = order, iters = iters)
    elif args.net == 'ESP2':
        netS = GC_LNN2(layers = layers, width = width, activation = activation)
        netE = GC_MNN2(layers = layers, width = width, activation = activation)
        net = ESPNN(netS, netE, data.dt / iters, order = order, iters = iters)
    elif args.net == 'ESP3':
        netS = GC_LNN3(layers = layers, width = width, activation = activation)
        netE = GC_MNN3(layers = layers, width = width, activation = activation)
        net = ESPNN(netS, netE, data.dt / iters, order = order, iters = iters)
    elif args.net == 'generic':
        S = ln.nn.FNN(ind, 1, layers=1, width=width, activation=activation)
        E = ln.nn.FNN(ind, 1, layers=layers, width=width, activation=activation)
        netS = generic_LNN(S, ind = ind)
        netE = generic_MNN(E, ind = ind, hidden_dim = ind)
        net = ESPNN(netS, netE, data.dt / iters, order = order, iters = iters)
    elif args.net == 'ON':
        fnn = ln.nn.FNN(data.dims, data.dims, layers, width, activation)
        net = ODENet(fnn, data.dt / iters, order = order, iters = iters)
    else:
        raise NotImplementedError
    
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    
    # training
    iterations  = 0
    lbfgs_steps = 0
    print_every = 100
    path = problem + args.net + str(args.seed) 
    callback = None
    #net = torch.load('outputs/'+path+'/model_best.pkl')
    #net = torch.load('model/'+path+'/model200000.pkl',map_location=torch.device('cpu'))
    
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
        'callback': callback,
        'dtype': dtype,
        'device': device,
    }
    
    ln.Brain.Init(**args2)
    ln.Brain.Run()
    ln.Brain.Restore()
    ln.Brain.Output()
    plot_GC(data, net, args)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generic Neural Networks')
    parser.add_argument('--net', default='ON', type=str, help='ESP or ESP2 or ESP3')
    parser.add_argument('--seed', default=3, type=int, help='random seed')
    args = parser.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    main(args)
