#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:40:47 2021

@author: zen
"""
import learner as ln
import torch
from learner.utils import mse, wasserstein, div, grad
from learner.integrator import RK, EM
import numpy as np



# learn L, M, E, S
class LNN(ln.nn.Module):
    def __init__(self, ind, K, layers=2, width=50, activation='relu', xi_scale=0.01):

        super(LNN, self).__init__()
        #self.S = S
        self.S = ln.nn.FNN(ind, 1, layers, width, activation)
        self.ind = ind
        self.K = K
        self.sigComp = ln.nn.FNN(ind, K**2 , layers, width, activation)
        self.xi_scale = xi_scale
        self.__init_params()
        
        
    def forward(self, x):
        sigComp = self.sigComp(x).reshape(-1, self.K, self.K)
        sigma = sigComp - torch.transpose(sigComp, -1, -2)

        x = x.requires_grad_(True)
        S = self.S(x)
        dS = grad(S, x).reshape([-1,self.ind])
        ddS = dS.unsqueeze(-2)
        B = []
        for i in range(self.K):
            xi = torch.triu(self.xi[i], diagonal = 1)
            xi = xi - torch.transpose(xi, -1,-2)
            B.append(ddS@xi)
        B = torch.cat(B, dim = -2)
        L = torch.transpose(B,-1,-2) @ sigma @ B
        if len(dS.size()) == 1:
            dS = dS.unsqueeze(0)
        return dS, L
        
    def __init_params(self):
        self.xi = torch.nn.Parameter((torch.randn([self.K, self.ind, self.ind])*self.xi_scale).requires_grad_(True)) 

class LNN_soft(ln.nn.Module):
    def __init__(self,ind, layers=2, width=50, activation='relu'):
        super(LNN_soft, self).__init__()
        self.ind = ind
        self.L = ln.nn.FNN(self.ind, self.ind ** 2, layers, width, activation)
        self.S = ln.nn.FNN(self.ind, 1, layers, width, activation)

    # def grad_S(self, x):
    #     x = np.squeeze(x)
    #     return grad(self.S, argnums=0)(x)

    def forward(self, x):
        L = self.L(x).reshape([-1, self.ind, self.ind])
        L = L - torch.transpose(L, -1, -2)
        x = x.requires_grad_(True)
        S = self.S(x)
        dS = grad(S, x)

        #dS = vmap(self.grad_S,in_dims=0)(x)

        #dS = vmap(jacrev(self.S, argnums=0), in_dims=0)(x)
        #dS = dS.squeeze(1)
        if len(dS.size()) == 1:
            dS = dS.unsqueeze(0)
        return dS, L


class MNN(ln.nn.Module):
    def __init__(self, ind, K, layers=2, width=50, activation='relu', xi_scale=0.01):
        super(MNN, self).__init__()
        self.E = ln.nn.FNN(ind, 1, layers, width, activation)
        self.ind = ind
        self.K = K
        self.sigComp = ln.nn.FNN(ind, K**2 , layers, width, activation)
        self.xi_scale = xi_scale
        self.__init_params()
        
    def forward(self, x):
        sigComp = self.sigComp(x).reshape(-1, self.K, self.K)
        sigma = sigComp @ torch.transpose(sigComp, -1, -2)

        x = x.requires_grad_(True)
        E = self.E(x)
        dE = grad(E, x).reshape([-1,self.ind])
        ddE = dE.unsqueeze(-2)
        B = []
        for i in range(self.K):
            xi = torch.triu(self.xi[i], diagonal = 1)
            xi = xi - torch.transpose(xi, -1,-2)
            B.append(ddE@xi)
        
        B = torch.cat(B, dim = -2)
        M = torch.transpose(B,-1,-2) @ sigma @ B
        

        if len(dE.size()) == 1:
            dE = dE.unsqueeze(0)
        return dE, M
        
    def __init_params(self):
        self.xi = torch.nn.Parameter((torch.randn([self.K, self.ind, self.ind])*self.xi_scale).requires_grad_(True))


class MNN_soft(ln.nn.Module):
    def __init__(self,ind, layers=2, width=50, activation='relu'):
        super(MNN_soft, self).__init__()
        self.ind = ind
        self.M = ln.nn.FNN(self.ind, self.ind ** 2, layers, width, activation)
        self.E = ln.nn.FNN(self.ind, 1, layers, width, activation)

    def forward(self, x):
        M = self.M(x).reshape([-1, self.ind, self.ind])
        M = M @ torch.transpose(M, -1, -2)
        x = x.requires_grad_(True)
        E = self.E(x)
        dE = grad(E, x)

        if len(dE.size()) == 1:
            dE = dE.unsqueeze(0)
        return dE, M


class GFINNs(ln.nn.LossNN):

    def __init__(self, netS, netE, dt, order=1, iters=1, kb=1, b_dim=1, fluc=False, lam=0):
        super(GFINNs, self).__init__()
        self.netS = netS
        self.netE = netE
        self.dt = dt
        self.iters = iters
        self.fluc = fluc
        self.b_dim = b_dim
        self.lam = lam
        self.integrator = RK(self.f, order=order, iters=iters)
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
        
        dE = dE.unsqueeze(1)
        dS = dS.unsqueeze(1)     

        
        dEM = dE @ M
        dSL = dS @ L

        
        return self.lam * (torch.mean(dEM ** 2) + torch.mean(dSL ** 2))

    def criterion(self, X, y):      

        X_next = self.integrator.solve(X, self.dt)

        loss = self.loss(X_next, y)

        if self.lam > 0:
            loss += self.consistency_loss(X)
        return loss
    
    def criterion_rel(self, X, y):      

        X_next = self.integrator.solve(X, self.dt)

#         loss = self.loss(X_next, y)
        loss = torch.mean((X_next - y) ** 2/(1e-6+(y)**2))

        if self.lam > 0:
            loss += self.consistency_loss(X)
        return loss
    
    def criterion2(self, X, y):      

        X_next = self.integrator.solve(X, self.dt)

        loss = self.loss(X_next, y)

        return loss

    def integrator2(self, X):
        X_next = self.integrator.solve(X, self.dt)
        return X_next

    def predict(self, x0, k, return_np=False):
        x = torch.transpose(self.integrator.flow(x0, self.dt, k - 1), 0, 1)
        if return_np:
            x = x.detach().cpu().numpy()
        return x
