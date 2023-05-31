#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 10:38:39 2021

@author: zen
"""

import learner as ln
import torch
import numpy as np
import argparse
from learner.utils import div, grad

from data import Data
from nn import ESPNN, ESPNN_stochastic
from postprocess_lg import plot_LG
from nn import *
    
class LNN_true(ln.nn.Module):
    def __init__(self):
        super(LNN_true, self).__init__()
        self.A = torch.nn.Parameter((torch.randn([2, 2]) * 0.01).requires_grad_(True))
        
    def forward(self, x):
        L = torch.tensor([[0,1,0], [-1,0,0], [0,0,0]], dtype = self.dtype, device = self.device)
        dS = torch.tensor([[0,0,1]], dtype = self.dtype, device = self.device).repeat(x.shape[0], 1)
        return dS, L
    
class MNN_true(ln.nn.Module):
    def __init__(self, kb = 1):
        super(MNN_true, self).__init__()
        self.kb = kb
        
    def forward(self, x):
        x = x.requires_grad_(True)
        B = self.ns(x)
        M = (B @ torch.transpose(B, 1, 2)) / 2 / self.kb
        p = x[...,1:2]
        dE = torch.cat([torch.zeros_like(p), p, torch.ones_like(p)], dim = -1)
        return dE, M
    
    def B(self, x):
        return self.ns(x)
    
    def dM(self, x):
        x = x.requires_grad_(True)
        B = self.ns(x)
        M = (B @ torch.transpose(B, 1, 2)) / 2 / self.kb
        return div(M, x)
    
    def ns(self, x):
        p = x[...,1:2]
        b1 = self.kb ** 0.5 * torch.cat([torch.zeros_like(p), torch.ones_like(p), -p], dim = -1)
        return b1[...,None]
        
def main(args):
    device = 'cpu' # 'cpu' or 'gpu'
    dtype = 'float'
    # data
    p = 0.5
    problem = 'LG'
    t_terminal = 1
    dt = 0.004
    trajs = args.trajs
    order = 1
    iters = 1
    kb = 1
    data = Data(p, problem, t_terminal, dt, trajs, order, iters, kb = kb, new = True)
    # NN
    layers = 5
    width = 30
    activation = 'tanh'
    b_dim = 1
    fixed = True
    if args.net == 'ESP':
        netS = LG_LNN(layers = layers, width = width, activation = activation)
        netE = LG_MNN(kb = kb, layers = layers, width = width, activation = activation)
    elif args.net == 'ESP2':
        netS = LG_LNN2(layers = layers, width = width, activation = activation)
        netE = LG_MNN2(kb = kb, layers = layers, width = width, activation = activation)
    elif args.net == 'ESP3':
        netS = LG_LNN3(layers = layers, width = width, activation = activation)
        netE = LG_MNN3(kb = kb, layers = layers, width = width, activation = activation)
    elif args.net == 'ON':
        netf = ln.nn.FNN(3, 3, layers, width, activation)
        netg = ln.nn.FNN(3, 3, layers, width, activation)
    else:
        raise NotImplementedError

    if args.net == 'ON':
        net = SDENet(netf, netg, data.dt / iters, order = 1, iters = iters, b_dim = b_dim)
    else:
        net = ESPNN_stochastic(netS, netE, data.dt / iters, kb = kb, order = order, iters = iters, b_dim = b_dim)
    netS_true = LNN_true()
    netE_true = MNN_true(kb = kb)
    net_true = ESPNN_stochastic(netS_true, netE_true, data.dt / iters, kb = kb, order = order, iters = iters, b_dim = b_dim)
    net_true.dtype = dtype
    net_true.device = device
    
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    
    # training
    lr = 0.01
    iterations = 50000
    print_every = 100
    batch_size = None
    path = problem + args.net + str(args.seed) 
    #net = torch.load('outputs/'+path+'/model_best.pkl')

    args2 = {
        'data': data,
        'net': net,
        'criterion': None,
        'optimizer': 'adam',
        'lr': lr,
        'weight_decay': 1e-5,
        'iterations': iterations,
        'lbfgs_steps': 0,
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
    plot_LG(data, ln.Brain.Best_model(), net_true, args)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generic Neural Networks')
    parser.add_argument('--net', default='ESP2', type=str, help='GE or ESP or SP')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--trajs', default=80, type=int, help='number of trajectories')
    args = parser.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    main(args)
