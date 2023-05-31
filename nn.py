#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:40:47 2021

@author: zen
"""
import learner as ln
import torch
from learner.utils import mse, wasserstein, grad, div
from learner.integrator import RK, EM
import numpy as np

# Know L and M, learn E and S    
class GC_LNN(ln.nn.Module):
    '''Fully connected neural networks in the null space of L
    '''
    def __init__(self, layers=2, width=50, activation='relu'):
        super(GC_LNN, self).__init__()
        self.fnn = ln.nn.FNN(2, 1, layers, width, activation)
        
    def forward(self, x):
        ns, L = self.ns()
        x = x.requires_grad_(True)
        S = self.fnn(x @ ns.t())
        dS = grad(S, x)
        return dS, L
    
    def ns(self):
        L = torch.tensor([[0,1,0,0],[-1,0,0,0],[0,0,0,0],[0,0,0,0]], dtype = self.dtype, device = self.device)
        ns = torch.tensor([[0,0,1,0],[0,0,0,1]], dtype = self.dtype, device = self.device)
        return ns, L
    
class GC_MNN(ln.nn.Module):
    def __init__(self, layers=2, width=50, activation='relu'):
        super(GC_MNN, self).__init__()
        self.fnn = ln.nn.FNN(3, 1, layers, width, activation)
        
    def forward(self, x):
        ns, M = self.ns(x)
        x = x.requires_grad_(True)
        F = self.F(x)
        y = torch.cat([x @ ns.t(), F[:,None]], dim = -1)
        E = self.fnn(y)
        dE = grad(E, x)
        return dE, M
    
    def F(self, x):
        q, _, S1, S2 = x[...,0], x[...,1], x[...,2], x[...,3]
        T1 = (torch.exp(S1) / q) ** (2 / 3)
        T2 = (torch.exp(S2) / (2 - q)) ** (2 / 3)
        return T1 + T2
    
    def ns(self, x):
        q, _, S1, S2 = x[...,0], x[...,1], x[...,2], x[...,3]
        T1 = 2/3 * (torch.exp(S1) / q) ** (2 / 3)
        T2 = 2/3 * (torch.exp(S2) / (2 - q)) ** (2 / 3)
        z1 = torch.zeros_like(T1)
        z2 = torch.zeros_like(T1)
        y = torch.stack([z1, z2, np.sqrt(10)/T1, -np.sqrt(10)/T2], dim = -1).unsqueeze(-1)
        M = y @ torch.transpose(y, -1, -2)
        ns = torch.tensor([[1,0,0,0],[0,1,0,0]], dtype = self.dtype, device = self.device)
        return ns, M

# Know E and S, learn L and M
class GC_LNN2(ln.nn.Module):
    def __init__(self, layers=2, width=50, activation='tanh'):
        super(GC_LNN2, self).__init__()
        self.ind = 4
        self.extraD = 5
        self.sigComp = ln.nn.FNN(self.ind, self.extraD * self.extraD , layers, width, activation)
        
        self.Xi1 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.01).requires_grad_(True))
        self.Xi2 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.1).requires_grad_(True))
        self.Xi3 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 1.0).requires_grad_(True))
        self.Xi4 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.1).requires_grad_(True))
        self.Xi5 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.01).requires_grad_(True))
    
    def forward(self, x):
        dS = torch.tensor([[0,0,1,1]], dtype = self.dtype, device = self.device).repeat(x.shape[0], 1)
        ddS = dS.unsqueeze(-2)
        sigComp = self.sigComp(x).reshape(-1, self.extraD, self.extraD)
        sigma = sigComp - torch.transpose(sigComp, -1, -2)
        
        Xi1 = self.Xi1 
        Xi1 = Xi1 - torch.transpose(Xi1, -1,-2)
        Xi2 = self.Xi2
        Xi2 = Xi2 - torch.transpose(Xi2, -1,-2)
        Xi3 = self.Xi3  
        Xi3 = Xi3 - torch.transpose(Xi3, -1,-2)
        Xi4 = self.Xi4  
        Xi4 = Xi4 - torch.transpose(Xi4, -1,-2)
        Xi5 = self.Xi5 
        Xi5 = Xi5 - torch.transpose(Xi5, -1,-2)
        B = torch.cat([ddS @ Xi1, ddS @ Xi2, ddS @ Xi3, ddS @ Xi4, ddS @ Xi5], dim = -2)
        L = torch.transpose(B,-1,-2) @ sigma @ B
        return dS, L
    
class GC_MNN2(ln.nn.Module):
    def __init__(self, layers=2, width=50, activation='tanh'):
        super(GC_MNN2, self).__init__()
        self.ind = 4
        self.extraD = 4
        self.fnnB = ln.nn.FNN(self.ind, self.extraD * self.extraD , layers, width, activation)
        self.Xi1 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.01).requires_grad_(True))
        self.Xi2 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.01).requires_grad_(True))
        self.Xi3 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.01).requires_grad_(True))
        self.Xi4 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.01).requires_grad_(True))
        
    def forward(self, x):
        dE = self.dE(x)
        ddE = dE.unsqueeze(-2) 
        Xi1 = self.Xi1
        Xi1 = (Xi1 - torch.transpose(Xi1, -1,-2))
        Xi2 = self.Xi2
        Xi2 = (Xi2 - torch.transpose(Xi2, -1,-2))
        Xi3 = self.Xi3  
        Xi3 = Xi3 - torch.transpose(Xi3, -1,-2)
        Xi4 = self.Xi4  
        Xi4 = Xi4 - torch.transpose(Xi4, -1,-2)
        
        B = torch.cat([ddE @ Xi1, ddE @ Xi2, ddE @ Xi3, ddE @ Xi4], dim = -2)
        sigComp = self.fnnB(x).reshape(-1, self.extraD, self.extraD) 
        sigma = sigComp @ torch.transpose(sigComp, -1,-2)
        M = torch.transpose(B,-1,-2) @ sigma @ B
        return dE, M
    
    def dE(self, x):
        q, p, S1, S2 = x[...,:1], x[...,1:2], x[...,2:3], x[...,3:]
        T1 = (torch.exp(S1) / q) ** (2 / 3) * (2 / 3)
        T2 = (torch.exp(S2) / (2 - q)) ** (2 / 3) * (2 / 3)
        dE = torch.cat([T2/(2-q) -T1/q, p, T1, T2], dim = -1)
        return dE

# Know nothing, learn L, M, E, S
class GC_LNN3(ln.nn.Module):
    def __init__(self, layers=2, width=50, activation='tanh'):
        super(GC_LNN3, self).__init__()
        self.ind = 4
        self.extraD = 5
        self.fnn = ln.nn.FNN(self.ind, 1, layers, width, activation)
        self.sigComp = ln.nn.FNN(self.ind, self.extraD * self.extraD , layers, width, activation)
        
        self.Xi1 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.01).requires_grad_(True))
        self.Xi2 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.1).requires_grad_(True))
        self.Xi3 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 1.0).requires_grad_(True))
        self.Xi4 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.1).requires_grad_(True))
        self.Xi5 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.01).requires_grad_(True))
    
    def forward(self, x):
        sigComp = self.sigComp(x).reshape(-1, self.extraD, self.extraD)
        sigma = sigComp - torch.transpose(sigComp, -1, -2)
        
        Xi1 = self.Xi1 
        Xi1 = Xi1 - torch.transpose(Xi1, -1,-2)
        Xi2 = self.Xi2
        Xi2 = Xi2 - torch.transpose(Xi2, -1,-2)
        Xi3 = self.Xi3  
        Xi3 = Xi3 - torch.transpose(Xi3, -1,-2)
        Xi4 = self.Xi4  
        Xi4 = Xi4 - torch.transpose(Xi4, -1,-2)
        Xi5 = self.Xi5 
        Xi5 = Xi5 - torch.transpose(Xi5, -1,-2)
        dS = self.ns(x)
        ddS = dS.unsqueeze(-2)
        B = torch.cat([ddS @ Xi1, ddS @ Xi2, ddS @ Xi3, ddS @ Xi4, ddS @ Xi5], dim = -2)
        L = torch.transpose(B,-1,-2) @ sigma @ B
        return dS, L
    
    def ns(self, x):
        x = x.requires_grad_(True)
        S = self.fnn(x)
        dS = grad(S, x) 
        return dS
    
class GC_MNN3(ln.nn.Module):
    def __init__(self, layers=2, width=50, activation='tanh'):
        super(GC_MNN3, self).__init__()
        self.ind = 4
        self.extraD = 4
        self.fnnB = ln.nn.FNN(self.ind, self.extraD * self.extraD , layers, width, activation)
        self.fnn = ln.nn.FNN(self.ind, 1, layers, width, activation)
        self.Xi1 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.01).requires_grad_(True))
        self.Xi2 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.01).requires_grad_(True))
        self.Xi3 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.01).requires_grad_(True))
        self.Xi4 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.01).requires_grad_(True))
        
    def forward(self, x):
        Xi1 = self.Xi1
        Xi1 = (Xi1 - torch.transpose(Xi1, -1,-2))
        Xi2 = self.Xi2
        Xi2 = (Xi2 - torch.transpose(Xi2, -1,-2))
        Xi3 = self.Xi3  
        Xi3 = Xi3 - torch.transpose(Xi3, -1,-2)
        Xi4 = self.Xi4  
        Xi4 = Xi4 - torch.transpose(Xi4, -1,-2)
        
        dE = self.ns(x)
        ddE = dE.unsqueeze(-2) 
        B = torch.cat([ddE @ Xi1, ddE @ Xi2, ddE @ Xi3, ddE @ Xi4], dim = -2)
        sigComp = self.fnnB(x).reshape(-1, self.extraD, self.extraD) 
        sigma = sigComp @ torch.transpose(sigComp, -1,-2)
        M = torch.transpose(B,-1,-2) @ sigma @ B
        return dE, M
    
    def ns(self, x):
        x = x.requires_grad_(True)
        E = self.fnn(x)
        dE = grad(E, x)
        return dE
    
# Know L and M, learn E and S    
class DP_LNN(ln.nn.Module):
    '''Fully connected neural networks in the null space of L
    '''
    def __init__(self, layers=2, width=50, activation='relu'):
        super(DP_LNN, self).__init__()
        self.fnn = ln.nn.FNN(2, 1, layers, width, activation)
        
    def forward(self, x):
        ns, L = self.ns()
        x = x.requires_grad_(True)
        S = self.fnn(x @ ns.t())
        dS = grad(S, x)
        return dS, L
    
    def ns(self):
        L = torch.tensor([[0,0,0,0,1,0,0,0,0,0],
                          [0,0,0,0,0,1,0,0,0,0],
                          [0,0,0,0,0,0,1,0,0,0],
                          [0,0,0,0,0,0,0,1,0,0],
                          [-1,0,0,0,0,0,0,0,0,0],
                          [0,-1,0,0,0,0,0,0,0,0],
                          [0,0,-1,0,0,0,0,0,0,0],
                          [0,0,0,-1,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0,0,0,0]], dtype = self.dtype, device = self.device)
        ns = torch.tensor([[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1]], dtype = self.dtype, device = self.device)
        return ns, L
    
class DP_LNN_soft(ln.nn.Module):
    def __init__(self, layers=2, width=50, activation='relu'):
        super(DP_LNN_soft, self).__init__()
        self.fnn = ln.nn.FNN(10, 1, layers, width, activation)
        
    def forward(self, x):
        L = self.L()
        x = x.requires_grad_(True)
        S = self.fnn(x)
        dS = grad(S, x)
        return dS, L
    
    def L(self):
        L = torch.tensor([[0,0,0,0,1,0,0,0,0,0],
                          [0,0,0,0,0,1,0,0,0,0],
                          [0,0,0,0,0,0,1,0,0,0],
                          [0,0,0,0,0,0,0,1,0,0],
                          [-1,0,0,0,0,0,0,0,0,0],
                          [0,-1,0,0,0,0,0,0,0,0],
                          [0,0,-1,0,0,0,0,0,0,0],
                          [0,0,0,-1,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0,0,0,0]], dtype = self.dtype, device = self.device)
        return L
    
class DP_MNN(ln.nn.Module):
    def __init__(self, layers=2, width=50, activation='relu'):
        super(DP_MNN, self).__init__()
        self.ind = 10
        self.fnn = ln.nn.FNN(9, 1, layers, width, activation)
        
    def forward(self, x):
        ns, M = self.ns(x)
        x = x.requires_grad_(True)
        F = self.F(x)
        y = torch.cat([x @ ns.t(), F], dim = -1)
        E = self.fnn(y)
        dE = grad(E, x)
        return dE, M
    
    def F(self, x):
        q1, q2, _, _, s1, s2 = x[...,:2], x[...,2:4], x[...,4:6], x[...,6:8], x[...,8:9], x[...,9:]
        lambda1 = torch.norm(q1, dim = -1, keepdim = True)
        lambda2 = torch.norm(q1 - q2, dim = -1, keepdim = True)
        theta1 = torch.exp((s1 - torch.log(lambda1)))
        theta2 = torch.exp((s2 - torch.log(lambda2)))
        return theta1 + theta2
    
    def ns(self, x):
        q1, q2, _, _, s1, s2 = x[...,:2], x[...,2:4], x[...,4:6], x[...,6:8], x[...,8:9], x[...,9:]
        lambda1 = torch.norm(q1, dim = -1, keepdim = True)
        lambda2 = torch.norm(q1 - q2, dim = -1, keepdim = True)
        theta1 = torch.exp((s1 - torch.log(lambda1)))
        theta2 = torch.exp((s2 - torch.log(lambda2)))
        z = torch.zeros(x.shape[0], self.ind-2, dtype = self.dtype, device = self.device)
        y = torch.cat([z, (theta2/theta1)**0.5, -(theta1/theta2)**0.5], dim = 1).unsqueeze(-1)
        M = y @ torch.transpose(y, 1, 2)
        ns = torch.tensor([[1,0,0,0,0,0,0,0,0,0],
                           [0,1,0,0,0,0,0,0,0,0],
                           [0,0,1,0,0,0,0,0,0,0],
                           [0,0,0,1,0,0,0,0,0,0],
                           [0,0,0,0,1,0,0,0,0,0],
                           [0,0,0,0,0,1,0,0,0,0],
                           [0,0,0,0,0,0,1,0,0,0],
                           [0,0,0,0,0,0,0,1,0,0]
                           ], dtype = self.dtype, device = self.device)
        return ns, M
    
class DP_MNN_soft(ln.nn.Module):
    def __init__(self, layers=2, width=50, activation='relu'):
        super(DP_MNN_soft, self).__init__()
        self.ind = 10
        self.fnn = ln.nn.FNN(self.ind, 1, layers, width, activation)
        
    def forward(self, x):
        M = self.M(x)
        x = x.requires_grad_(True)
        E = self.fnn(x)
        dE = grad(E, x)
        return dE, M
    
    def M(self, x):
        q1, q2, _, _, s1, s2 = x[...,:2], x[...,2:4], x[...,4:6], x[...,6:8], x[...,8:9], x[...,9:]
        lambda1 = torch.norm(q1, dim = -1, keepdim = True)
        lambda2 = torch.norm(q1 - q2, dim = -1, keepdim = True)
        theta1 = torch.exp((s1 - torch.log(lambda1)))
        theta2 = torch.exp((s2 - torch.log(lambda2)))
        z = torch.zeros(x.shape[0], self.ind - 2, dtype = self.dtype, device = self.device)
        y = torch.cat([z, (theta2/theta1)**0.5, -(theta1/theta2)**0.5], dim = 1).unsqueeze(-1)
        M = y @ torch.transpose(y, 1, 2)
        return M

# Know E and S, learn L and M
class DP_LNN2(ln.nn.Module):
    def __init__(self, layers=2, width=50, activation='tanh'):
        super(DP_LNN2, self).__init__()
        self.ind = 10
        self.extraD = 5
        self.sigComp = ln.nn.FNN(self.ind, self.extraD * self.extraD, layers, width, activation)
        
        self.Xi1 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.01).requires_grad_(True))
        self.Xi2 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.1).requires_grad_(True))
        self.Xi3 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 1.0).requires_grad_(True))
        self.Xi4 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.1).requires_grad_(True))
        self.Xi5 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.01).requires_grad_(True))
    
    def forward(self, x):
        dS = torch.tensor([[0,0,0,0,0,0,0,0,1,1]], dtype = self.dtype, device = self.device).repeat(x.shape[0], 1)
        ddS = dS.unsqueeze(-2)
        sigComp = self.sigComp(x).reshape(-1, self.extraD, self.extraD)
        sigma = sigComp - torch.transpose(sigComp, -1, -2)
        
        Xi1 = self.Xi1 
        Xi1 = Xi1 - torch.transpose(Xi1, -1,-2)
        Xi2 = self.Xi2
        Xi2 = Xi2 - torch.transpose(Xi2, -1,-2)
        Xi3 = self.Xi3  
        Xi3 = Xi3 - torch.transpose(Xi3, -1,-2)
        Xi4 = self.Xi4  
        Xi4 = Xi4 - torch.transpose(Xi4, -1,-2)
        Xi5 = self.Xi5 
        Xi5 = Xi5 - torch.transpose(Xi5, -1,-2)
        B = torch.cat([ddS @ Xi1, ddS @ Xi2, ddS @ Xi3, ddS @ Xi4, ddS @ Xi5], dim = -2)
        L = torch.transpose(B,-1,-2) @ sigma @ B
        return dS, L
    
class DP_LNN2_soft(ln.nn.Module):
    def __init__(self, layers=2, width=50, activation='tanh'):
        super(DP_LNN2_soft, self).__init__()
        self.ind = 10
        self.L = ln.nn.FNN(self.ind, self.ind**2 , layers, width, activation)
    
    def forward(self, x):
        dS = torch.tensor([[0,0,0,0,0,0,0,0,1,1]], dtype = self.dtype, device = self.device).repeat(x.shape[0], 1)
        L = self.L(x).reshape([-1, self.ind, self.ind])
        L = L - torch.transpose(L, -1, -2)
        return dS, L
    
class DP_MNN2(ln.nn.Module):
    def __init__(self, layers=2, width=50, activation='tanh'):
        super(DP_MNN2, self).__init__()
        self.ind = 10
        self.extraD = 4
        self.fnnB = ln.nn.FNN(self.ind, self.extraD * self.extraD , layers, width, activation)
        self.Xi1 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.01).requires_grad_(True))
        self.Xi2 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.01).requires_grad_(True))
        self.Xi3 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.01).requires_grad_(True))
        self.Xi4 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.01).requires_grad_(True))
        
    def forward(self, x):
        dE = self.dE(x)
        ddE = dE.unsqueeze(-2) 
        Xi1 = self.Xi1
        Xi1 = (Xi1 - torch.transpose(Xi1, -1,-2))
        Xi2 = self.Xi2
        Xi2 = (Xi2 - torch.transpose(Xi2, -1,-2))
        Xi3 = self.Xi3  
        Xi3 = Xi3 - torch.transpose(Xi3, -1,-2)
        Xi4 = self.Xi4  
        Xi4 = Xi4 - torch.transpose(Xi4, -1,-2)
        
        B = torch.cat([ddE @ Xi1, ddE @ Xi2, ddE @ Xi3, ddE @ Xi4], dim = -2)
        sigComp = self.fnnB(x).reshape(-1, self.extraD, self.extraD) 
        sigma = sigComp @ torch.transpose(sigComp, -1,-2)
        M = torch.transpose(B,-1,-2) @ sigma @ B
        return dE, M
    
    def dE(self, x):
        q1, q2, p1, p2, s1, s2 = x[...,:2], x[...,2:4], x[...,4:6], x[...,6:8], x[...,8:9], x[...,9:]
        lambda1 = torch.norm(q1, dim = -1, keepdim = True)
        lambda2 = torch.norm(q1 - q2, dim = -1, keepdim = True)
        theta1 = torch.exp((s1 - torch.log(lambda1)))
        theta2 = torch.exp((s2 - torch.log(lambda2)))
        dq1 = torch.log(lambda1) / (lambda1 **2) * q1 + q1 / (lambda1**2) - torch.exp((s1 - torch.log(lambda1))) * q1 / lambda1 ** 2
        dq1 += torch.log(lambda2) / (lambda2 **2) * (q1-q2) + (q1 - q2)/ (lambda2**2) - torch.exp((s2 - torch.log(lambda2))) * (q1 -q2) / lambda2 ** 2
        dq2 = torch.log(lambda2) / (lambda2 **2) * (q2-q1) + (q2 - q1)/ (lambda2**2) - torch.exp((s2 - torch.log(lambda2))) * (q2 -q1) / lambda2 ** 2
        dp1 = p1 
        dp2 = p2
        ds1 = theta1
        ds2 = theta2
        dE = torch.cat([dq1, dq2, dp1, dp2, ds1, ds2], dim = -1)
        return dE
    
class DP_MNN2_soft(ln.nn.Module):
    def __init__(self, layers=2, width=50, activation='tanh'):
        super(DP_MNN2_soft, self).__init__()
        self.ind = 10
        self.M = ln.nn.FNN(self.ind, self.ind**2 , layers, width, activation)
    
    def forward(self, x):
        dE = self.dE(x)
        M = self.M(x).reshape([-1, self.ind, self.ind])
        M = M @ torch.transpose(M, -1, -2)
        return dE, M
    
    def dE(self, x):
        q1, q2, p1, p2, s1, s2 = x[...,:2], x[...,2:4], x[...,4:6], x[...,6:8], x[...,8:9], x[...,9:]
        lambda1 = torch.norm(q1, dim = -1, keepdim = True)
        lambda2 = torch.norm(q1 - q2, dim = -1, keepdim = True)
        theta1 = torch.exp((s1 - torch.log(lambda1)))
        theta2 = torch.exp((s2 - torch.log(lambda2)))
        dq1 = torch.log(lambda1) / (lambda1 **2) * q1 + q1 / (lambda1**2) - torch.exp((s1 - torch.log(lambda1))) * q1 / lambda1 ** 2
        dq1 += torch.log(lambda2) / (lambda2 **2) * (q1-q2) + (q1 - q2)/ (lambda2**2) - torch.exp((s2 - torch.log(lambda2))) * (q1 -q2) / lambda2 ** 2
        dq2 = torch.log(lambda2) / (lambda2 **2) * (q2-q1) + (q2 - q1)/ (lambda2**2) - torch.exp((s2 - torch.log(lambda2))) * (q2 -q1) / lambda2 ** 2
        dp1 = p1 
        dp2 = p2
        ds1 = theta1
        ds2 = theta2
        dE = torch.cat([dq1, dq2, dp1, dp2, ds1, ds2], dim = -1)
        return dE
    
# Know nothing, learn L, M, E, S
class DP_LNN3(ln.nn.Module):
    def __init__(self, layers=2, width=50, activation='tanh'):
        super(DP_LNN3, self).__init__()
        self.ind = 10
        self.extraD = 5
        self.fnn = ln.nn.FNN(self.ind, 1, layers, width, activation)
        self.sigComp = ln.nn.FNN(self.ind, self.extraD * self.extraD , layers, width, activation)
        
        self.Xi1 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.01).requires_grad_(True))
        self.Xi2 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.1).requires_grad_(True))
        self.Xi3 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 1.0).requires_grad_(True))
        self.Xi4 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.1).requires_grad_(True))
        self.Xi5 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.01).requires_grad_(True))
    
    def forward(self, x):
        sigComp = self.sigComp(x).reshape(-1, self.extraD, self.extraD)
        sigma = sigComp - torch.transpose(sigComp, -1, -2)
        
        Xi1 = self.Xi1 
        Xi1 = Xi1 - torch.transpose(Xi1, -1,-2)
        Xi2 = self.Xi2
        Xi2 = Xi2 - torch.transpose(Xi2, -1,-2)
        Xi3 = self.Xi3  
        Xi3 = Xi3 - torch.transpose(Xi3, -1,-2)
        Xi4 = self.Xi4  
        Xi4 = Xi4 - torch.transpose(Xi4, -1,-2)
        Xi5 = self.Xi5 
        Xi5 = Xi5 - torch.transpose(Xi5, -1,-2)
        dS = self.ns(x)
        ddS = dS.unsqueeze(-2)
        B = torch.cat([ddS @ Xi1, ddS @ Xi2, ddS @ Xi3, ddS @ Xi4, ddS @ Xi5], dim = -2)
        L = torch.transpose(B,-1,-2) @ sigma @ B
        return dS, L
    
    def ns(self, x):
        x = x.requires_grad_(True)
        S = self.fnn(x)
        dS = grad(S, x) 
        return dS
    
class DP_LNN3_soft(ln.nn.Module):
    def __init__(self, layers=2, width=50, activation='tanh'):
        super(DP_LNN3_soft, self).__init__()
        self.ind = 10
        self.L = ln.nn.FNN(self.ind, self.ind**2 , layers, width, activation)
        self.S = ln.nn.FNN(self.ind, 1, layers, width, activation)
    
    def forward(self, x):
        L = self.L(x).reshape([-1, self.ind, self.ind])
        L = L - torch.transpose(L, -1, -2)
        x = x.requires_grad_(True)
        S = self.S(x)
        dS = grad(S, x)
        return dS, L
    
class DP_MNN3(ln.nn.Module):
    def __init__(self, layers=2, width=50, activation='tanh'):
        super(DP_MNN3, self).__init__()
        self.ind = 10
        self.extraD = 4
        self.fnnB = ln.nn.FNN(self.ind, self.extraD * self.extraD , layers, width, activation)
        self.fnn = ln.nn.FNN(self.ind, 1, layers, width, activation)
        self.Xi1 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.01).requires_grad_(True))
        self.Xi2 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.01).requires_grad_(True))
        self.Xi3 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.01).requires_grad_(True))
        self.Xi4 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.01).requires_grad_(True))
        
    def forward(self, x):
        Xi1 = self.Xi1
        Xi1 = (Xi1 - torch.transpose(Xi1, -1,-2))
        Xi2 = self.Xi2
        Xi2 = (Xi2 - torch.transpose(Xi2, -1,-2))
        Xi3 = self.Xi3  
        Xi3 = Xi3 - torch.transpose(Xi3, -1,-2)
        Xi4 = self.Xi4  
        Xi4 = Xi4 - torch.transpose(Xi4, -1,-2)
        dE = self.ns(x)
        ddE   = dE.unsqueeze(-2)
        B = torch.cat([ddE @ Xi1, ddE @ Xi2, ddE @ Xi3, ddE @ Xi4], dim = -2)
        sigComp = self.fnnB(x).reshape(-1, self.extraD, self.extraD) 
        sigma = sigComp @ torch.transpose(sigComp, -1,-2)
        M = torch.transpose(B,-1,-2) @ sigma @ B
        return dE, M
    
    def ns(self, x):
        x = x.requires_grad_(True)
        E = self.fnn(x)
        dE = grad(E, x)
        return dE
    
class DP_MNN3_soft(ln.nn.Module):
    def __init__(self, layers=2, width=50, activation='tanh'):
        super(DP_MNN3_soft, self).__init__()
        self.ind = 10
        self.M = ln.nn.FNN(self.ind, self.ind**2, layers, width, activation)
        self.E = ln.nn.FNN(self.ind, 1, layers, width, activation)
    
    def forward(self, x):
        M = self.M(x).reshape([-1, self.ind, self.ind])
        M = M @ torch.transpose(M, -1, -2)
        x = x.requires_grad_(True)
        E = self.E(x)
        dE = grad(E, x)
        return dE, M

class LG_LNN(ln.nn.Module):
    def __init__(self, layers, width, activation):
        super(LG_LNN, self).__init__()
        self.fnn = ln.nn.FNN(1, 1, layers, width, activation)
        
    def forward(self, x):
        L = torch.tensor([[0,1,0], [-1,0,0], [0,0,0]], dtype = self.dtype, device = self.device)
        ns = torch.tensor([[0,0,1]], dtype = self.dtype, device = self.device)
        x = x.requires_grad_(True)
        S = self.fnn(x @ ns.t())
        dS = grad(S, x)
        return dS, L
    
class LG_MNN(ln.nn.Module):
    def __init__(self, layers, width, activation, kb = 1):
        super(LG_MNN, self).__init__()
        self.kb = kb
        self.fnn = ln.nn.FNN(2, 1, layers, width, activation)
        
    def forward(self, x):
        B = self.ns(x)
        M = (B @ torch.transpose(B, 1, 2)) / 2 / self.kb
        x = x.requires_grad_(True)
        F = self.F(x)
        ns = torch.tensor([[1,0,0]], dtype = self.dtype, device = self.device)
        y = torch.cat([x @ ns.t(), F[:,None]], dim = -1)
        E = self.fnn(y)
        dE = grad(E, x)
        return dE, M
    
    def F(self, x):
        _, p, S = x[...,0], x[...,1], x[...,2]
        return p**2/2 + S
    
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
    
class LG_LNN2(ln.nn.Module):
    def __init__(self, layers=2, width=50, activation='tanh'):
        super(LG_LNN2, self).__init__()
        self.ind = 3
        self.extraD = 5
        self.sigComp = ln.nn.FNN(self.ind, self.extraD * self.extraD , layers, width, activation)
        self.Xi1 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.01).requires_grad_(True))
        self.Xi2 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.1).requires_grad_(True))
        self.Xi3 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 1.0).requires_grad_(True))
        self.Xi4 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.1).requires_grad_(True))
        self.Xi5 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.01).requires_grad_(True))
        
    def forward(self, x):
        dS = torch.tensor([[0,0,1]], dtype = self.dtype, device = self.device).repeat(x.shape[0], 1)
        ddS = dS.unsqueeze(-2)
        sigComp = self.sigComp(x).reshape(-1, self.extraD, self.extraD)
        sigma = sigComp - torch.transpose(sigComp, -1, -2)
        Xi1 = self.Xi1
        Xi1 = Xi1 - torch.transpose(Xi1, -1,-2)
        Xi2 = self.Xi2
        Xi2 = Xi2 - torch.transpose(Xi2, -1,-2)
        Xi3 = self.Xi3
        Xi3 = Xi3 - torch.transpose(Xi3, -1,-2)
        Xi4 = self.Xi4
        Xi4 = Xi4 - torch.transpose(Xi4, -1,-2)
        Xi5 = self.Xi5
        Xi5 = Xi5 - torch.transpose(Xi5, -1,-2)
        B = torch.cat([ddS @ Xi1, ddS @ Xi2, ddS @ Xi3, ddS @ Xi4, ddS @ Xi5], dim = -2)
        L = torch.transpose(B,-1,-2) @ sigma @ B
        return dS, L
    
class LG_MNN2(ln.nn.Module):
    def __init__(self, kb = 1, layers=2, width=50, activation='relu'):
        super(LG_MNN2, self).__init__()
        self.ind = 3
        self.extraD = 1
        self.kb = kb
        self.sigComp = torch.nn.Parameter((torch.randn([self.extraD, self.extraD])).requires_grad_(True))
        self.Xi1 = torch.nn.Parameter((torch.randn([self.ind, self.ind])).requires_grad_(True))
        self.fnnB = ln.nn.FNN(self.ind, self.extraD ** 2, layers, width, activation)
        
    def forward(self, x):
        dE = self.dE(x)
        ddE = dE.unsqueeze(-2) 
        B = ddE @ (self.Xi1 - torch.transpose(self.Xi1, -1,-2))
        sigComp = self.fnnB(x).reshape(-1, self.extraD, self.extraD) 
        B = torch.transpose(B,-1,-2) @ sigComp
        M = (B @ torch.transpose(B, -1, -2)) / 2 / self.kb
        return dE, M
    
    def B(self, x):
        dE = self.dE(x)
        ddE = dE.unsqueeze(-2) 
        B = ddE @ (self.Xi1 - torch.transpose(self.Xi1, -1,-2))
        sigComp = self.fnnB(x).reshape(-1, self.extraD, self.extraD) 
        B = torch.transpose(B,-1,-2) @ sigComp
        return B
    
    def dE(self, x):
        p = x[...,1:2]
        dE = torch.cat([torch.zeros_like(p), p, torch.ones_like(p)], dim = -1)
        return dE
    
    def dM(self, x):
        x = x.requires_grad_(True)
        dE = self.dE(x)
        ddE = dE.unsqueeze(-2) 
        B = ddE @ (self.Xi1 - torch.transpose(self.Xi1, -1,-2))
        sigComp = self.fnnB(x).reshape(-1, self.extraD, self.extraD) 
        B = torch.transpose(B,-1,-2) @ sigComp
        M = (B @ torch.transpose(B, -1, -2)) / 2 / self.kb
        return div(M, x)
       
class LG_LNN3(ln.nn.Module):
    def __init__(self, layers=2, width=50, activation='tanh'):
        super(LG_LNN3, self).__init__()
        self.ind = 3
        self.extraD = 5
        self.sigComp = ln.nn.FNN(self.ind, self.extraD * self.extraD, layers, width, activation)
        self.Xi1 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.01).requires_grad_(True))
        self.Xi2 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.1).requires_grad_(True))
        self.Xi3 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 1.0).requires_grad_(True))
        self.Xi4 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.1).requires_grad_(True))
        self.Xi5 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.01).requires_grad_(True))
        self.fnn = ln.nn.FNN(self.ind, 1, 1, width, activation)

    def forward(self, x):
        sigComp = self.sigComp(x).reshape(-1, self.extraD, self.extraD)
        sigma = sigComp - torch.transpose(sigComp, -1, -2)
        Xi1 = self.Xi1
        Xi1 = Xi1 - torch.transpose(Xi1, -1,-2)
        Xi2 = self.Xi2
        Xi2 = Xi2 - torch.transpose(Xi2, -1,-2)
        Xi3 = self.Xi3
        Xi3 = Xi3 - torch.transpose(Xi3, -1,-2)
        Xi4 = self.Xi4
        Xi4 = Xi4 - torch.transpose(Xi4, -1,-2)
        Xi5 = self.Xi5
        Xi5 = Xi5 - torch.transpose(Xi5, -1,-2)
        dS = self.dS(x)
        ddS = dS.unsqueeze(-2)
        B = torch.cat([ddS @ Xi1, ddS @ Xi2, ddS @ Xi3, ddS @ Xi4, ddS @ Xi5], dim = -2)
        L = torch.transpose(B,-1,-2) @ sigma @ B
        return dS, L

    def dS(self, x):
        x = x.requires_grad_(True)
        S = self.fnn(x)
        dS = grad(S, x) 
        return dS

class LG_MNN3(ln.nn.Module):
    def __init__(self, kb = 1, layers=2, width=50, activation='relu'):
        super(LG_MNN3, self).__init__()
        self.ind = 3
        self.extraD = 1
        self.kb = kb
        self.sigComp = torch.nn.Parameter((torch.randn([self.extraD, self.extraD])).requires_grad_(True))
        self.Xi1 = torch.nn.Parameter((torch.randn([self.ind, self.ind])).requires_grad_(True))
        self.fnnB = ln.nn.FNN(self.ind, self.extraD ** 2, 1, width, activation)
        self.fnn = ln.nn.FNN(self.ind, 1, layers, width, activation)

    def forward(self, x):
        dE = self.dE(x)
        ddE = dE.unsqueeze(-2)
        B = ddE @ (self.Xi1 - torch.transpose(self.Xi1, -1,-2))
        sigComp = self.fnnB(x).reshape(-1, self.extraD, self.extraD)
        B = torch.transpose(B,-1,-2) @ sigComp
        M = (B @ torch.transpose(B, -1, -2)) / 2 / self.kb
        return dE, M

    def B(self, x):
        dE = self.dE(x)
        ddE = dE.unsqueeze(-2)
        B = ddE @ (self.Xi1 - torch.transpose(self.Xi1, -1,-2))
        sigComp = self.fnnB(x).reshape(-1, self.extraD, self.extraD)
        B = torch.transpose(B,-1,-2) @ sigComp
        return B

    def dE(self, x):
        x = x.requires_grad_(True)
        E = self.fnn(x)
        dE = grad(E, x)
        return dE

    def dM(self, x):
        x = x.requires_grad_(True)
        dE = self.dE(x)
        ddE = dE.unsqueeze(-2)
        B = ddE @ (self.Xi1 - torch.transpose(self.Xi1, -1,-2))
        sigComp = self.fnnB(x).reshape(-1, self.extraD, self.extraD)
        B = torch.transpose(B,-1,-2) @ sigComp
        M = (B @ torch.transpose(B, -1, -2)) / 2 / self.kb
        return div(M, x)

class LG_LNN_true(ln.nn.Module):
    def __init__(self):
        super(LG_LNN_true, self).__init__()
        self.A = torch.nn.Parameter((torch.randn([2, 2]) * 0.01).requires_grad_(True))
        
    def forward(self, x):
        L = torch.tensor([[0,1,0], [-1,0,0], [0,0,0]], dtype = self.dtype, device = self.device)
        dS = torch.tensor([[0,0,1]], dtype = self.dtype, device = self.device).repeat(x.shape[0], 1)
        return dS, L
    
class LG_MNN_true(ln.nn.Module):
    def __init__(self, kb = 1):
        super(LG_MNN_true, self).__init__()
        self.kb = kb
        
    def forward(self, x):
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

class generic_LNN(ln.nn.Module):
    def __init__(self, S, ind):
        super(generic_LNN, self).__init__()
        self.S = S
        self.ind = ind
        self.__init_params()
        
    def forward(self, x):
        xi1 = self.xi_tilde
        xi2 = torch.transpose(xi1, 1, 2)
        xi3 = torch.transpose(xi2, 0, 2)
        xi4 = torch.transpose(xi3, 1, 2)
        xi5 = torch.transpose(xi4, 0, 2)
        xi6 = torch.transpose(xi5, 1, 2)
        xi = (xi1 - xi2 + xi3 - xi4 + xi5 - xi6) / 6 
        x = x.requires_grad_(True)
        S = self.S(x)
        dS = grad(S, x)
        L = torch.tensordot(dS, xi, dims = ([-1], [-1]))
        return dS, L.squeeze()
        
    def __init_params(self):
        self.xi_tilde = torch.nn.Parameter(torch.randn([self.ind, self.ind, self.ind]).requires_grad_(True))
        
class generic_MNN(ln.nn.Module):
    def __init__(self, E, ind, hidden_dim):
        super(generic_MNN, self).__init__()
        self.E = E
        self.ind = ind
        self.hidden_dim = hidden_dim
        self.__init_params()
        
    def forward(self, x):
        lam = (self.lam_tilde - torch.transpose(self.lam_tilde, -1, -2)) / 2
        D = self.D_tilde @ self.D_tilde.t()
        zeta = torch.tensordot(torch.tensordot(lam, D, dims = ([0], [0])), lam, dims = ([2], [0]))
        x = x.requires_grad_(True)
        E = self.E(x)
        dE = grad(E, x)
        dE2 = dE[...,None] @ dE[:,None]
        M = torch.tensordot(dE2, zeta, dims = ([1,2], [1,3]))
        return dE, M.squeeze()
        
    def __init_params(self):
        self.lam_tilde = torch.nn.Parameter(torch.randn([self.hidden_dim, self.ind, self.ind]).requires_grad_(True))
        self.D_tilde = torch.nn.Parameter(torch.randn([self.hidden_dim, self.hidden_dim]).requires_grad_(True))
        
class ESPNN(ln.nn.LossNN):
    ''' Most useful one: 
    netE: state variable x -> dE, M, orthogonal to each other
    netS: state variable x -> dS, L, orthogonal to each other
    the loss is defined in 'criterion' function
    '''
    def __init__(self, netS, netE, dt, order = 1, iters = 1, kb = 1, b_dim = 1, fluc = False, lam = 0):
        super(ESPNN, self).__init__()
        self.netS = netS
        self.netE = netE
        self.dt = dt
        self.iters = iters
        self.fluc = fluc
        self.b_dim = b_dim
        self.lam = lam
        self.integrator = RK(self.f, order = order, iters = iters)
        self.loss = mse
            
    def f(self, x):
        dE, M = self.netE(x)
        dS, L = self.netS(x)
        dE = dE.unsqueeze(1)
        dS = dS.unsqueeze(1)
        return -(dE @ L).squeeze() + (dS @ M).squeeze() 
    
    def g(self, x):
        return self.netE.B(x)
    
    def consistency_loss(self, x):
        dE, M = self.netE(x)
        dS, L = self.netS(x)
        dEM = dE @ M
        dSL = dS @ L
        return self.lam * (torch.mean(dEM**2) + torch.mean(dSL**2))
        
    def criterion(self, X, y):
        X_next = self.integrator.solve(X, self.dt)
        #print(X_next.shape)
        # print(y.shape)
        #print(X_next.shape) #torch.Size([100, 10]) torch.Size([8000, 10])
        # print(y.shape) torch.Size([100, 10]) torch.Size([8000, 10])
        loss = self.loss(X_next, y) 
        if self.lam > 0:
            loss += self.consistency_loss(X)
        return loss
    
    def predict(self, x0, k, return_np = False):
        x = torch.transpose(self.integrator.flow(x0, self.dt, k - 1), 0, 1)
        if return_np:
            x = x.detach().cpu().numpy()
        return x
    
class ESPNN_stochastic(ln.nn.LossNN):
    def __init__(self, netS, netE, dt, order = 1, iters = 1, kb = 1, b_dim = 1):
        super(ESPNN_stochastic, self).__init__()
        self.netS = netS
        self.netE = netE
        self.dt = dt
        self.iters = iters
        self.b_dim = b_dim
        self.kb = kb
        self.integrator = EM(self.f, self.g, order = order, iters = iters, b_dim = b_dim)
        
    def f(self, x):
        dE, M = self.netE(x)
        dS, L = self.netS(x)
        dE = dE.unsqueeze(1)
        dS = dS.unsqueeze(1)
        return -(dE @ L).squeeze() + (dS @ M).squeeze() + self.kb * self.netE.dM(x)
    
    def g(self, x):
        return self.netE.B(x)
    
    def criterion(self, X, y):
        mean = (self.f(X) * self.dt)[...,None]
        eps = 1e-6 * torch.tensor([1,0,0], device = self.device, dtype = self.dtype).reshape([1,-1,1]).repeat(X.shape[0],1,1)
        var = (self.g(X) ** 2 * self.dt+eps)[...,None] ** (1/2)
        m = torch.distributions.multivariate_normal.MultivariateNormal(mean, scale_tril = var)
        return -m.log_prob((y - X)[...,None]).mean()
    
    def predict(self, x0, k, noise = None, return_np = False):
        x = torch.transpose(self.integrator.flow(x0, self.dt, k - 1, noise), 0, 1)
        if return_np:
            x = x.detach().cpu().numpy()
        return x
 
class SDENet(ln.nn.LossNN):
    def __init__(self, netf, netg, dt, order = 1, iters = 1, b_dim = 1):
        super(SDENet, self).__init__()
        self.f = netf
        self.g = netg
        self.dt = dt
        self.iters = iters
        self.b_dim = b_dim
        self.integrator = EM(self.f, self.g, order = order, iters = iters, b_dim = b_dim)
 
    def criterion(self, X, y):
        mean = (self.f(X) * self.dt)[...,None]
        eps = 1e-6 * torch.tensor([1,0,0], device = self.device, dtype = self.dtype).reshape([1,-1,1]).repeat(X.shape[0],1,1)
        var = (self.g(X).unsqueeze(-1) ** 2 * self.dt+eps)[...,None] ** (1/2)
        m = torch.distributions.multivariate_normal.MultivariateNormal(mean, scale_tril = var)
        return -m.log_prob((y - X)[...,None]).mean()

    def predict(self, x0, k, noise = None, return_np = False):
        x = torch.transpose(self.integrator.flow(x0, self.dt, k - 1, noise), 0, 1)
        if return_np:
            x = x.detach().cpu().numpy()
        return x
 
class ODENet(ln.nn.LossNN):
    ''' ODENet (EulerNet when order = 1)
    '''
    def __init__(self, net, dt, order = 1, iters = 1):
        super(ODENet, self).__init__()
        self.net = net
        self.dt = dt
        self.iters = iters
        self.integrator = RK(self.net, order = order, iters = iters)
        
    def criterion(self, X, y):
        X = self.integrator.solve(X, self.dt)
        return mse(X, y)
    
    def predict(self, x0, k, return_np = False):
        x = [x0]
        for i in range(k - 1):
            xx = x[-1]
            xx = self.integrator.solve(xx, self.dt)
            x.append(xx)
        x = torch.stack(x)
        if return_np:
            x = x.detach().cpu().numpy()
        return x
            
