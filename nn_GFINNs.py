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
from functorch import vmap, jacrev
#from functorch import grad, vmap, jacrev


# Know L and M, learn E and S    
class GC_LNN(ln.nn.Module):
    '''Fully connected neural networks in the null space of L
    '''
    def __init__(self, layers=2,width=50, activation='relu'):
        super(GC_LNN, self).__init__()
        self.fnn = ln.nn.FNN(2, 1, layers, width, activation)
        
    def forward(self, x):
        ns, L = self.ns()
        x = x.requires_grad_(True)
        S = self.fnn(x @ ns.t())
        dS = grad(S, x)
        if len(dS.size()) == 1:
            dS = dS.unsqueeze(0)
        return dS, L
    
    def ns(self):
        L = torch.tensor([[0,1,0,0],[-1,0,0,0],[0,0,0,0],[0,0,0,0]], dtype = self.dtype, device = self.device)
        ns = torch.tensor([[0,0,1,0],[0,0,0,1]], dtype = self.dtype, device = self.device)
        return ns, L


class GC_LNN_soft(ln.nn.Module):
    '''Fully connected neural networks in the null space of L
    '''

    def __init__(self, layers=2, width=50, activation='relu'):
        super(GC_LNN_soft, self).__init__()
        self.fnn = ln.nn.FNN(4, 1, layers, width, activation)

    def forward(self, x):
        L = self.L()
        x = x.requires_grad_(True)
        S = self.fnn(x)
        dS = grad(S, x)
        if len(dS.size()) == 1:
            dS = dS.unsqueeze(0)
        
        return dS, L

    def L(self):
        L = torch.tensor([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=self.dtype,
                         device=self.device)
        return L
    
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
        if len(dE.size()) == 1:
            dE = dE.unsqueeze(0)

        return dE, M
    
    def F(self, x):
        q, _, S1, S2 = x[...,0], x[...,1], x[...,2], x[...,3]
#         T1 = torch.abs(torch.exp(S1) / q) ** (2 / 3)
#         T2 = torch.abs(torch.exp(S2) / (2 - q)) ** (2 / 3)
#         T1 = (torch.exp(S1) / q) ** (2 / 3)
#         T2 = (torch.exp(S2) / (2 - q)) ** (2 / 3)
        T1 = ((torch.exp(S1) / q) **2 )**(1 / 3)
        T2 = ((torch.exp(S2) / (2 - q)) ** 2)**(1 / 3)
        return T1 + T2
    
    def ns(self, x):
        q, _, S1, S2 = x[...,0], x[...,1], x[...,2], x[...,3]
#         print(torch.min(S1))
#         print(torch.min(S2))
#         print('q')
#         print(q) #3.8204
#         print('S2')
#         print(S2) #-8.5198e+03
#         print('abs')
#         print(torch.abs(torch.exp(S2) / (2 - q)))
#         T1 = 2/3 * (torch.exp(S1) / q) ** (2 / 3)
#         T2 = 2/3 * (torch.exp(S2) / (2 - q)) ** (2 / 3)
#         T1 = 2/3 * torch.abs(torch.exp(S1) / q) ** (2 / 3)
#         T2 = 2/3 * torch.abs(torch.exp(S2) / (2 - q)) ** (2 / 3)
        T1 = 2/3 *((torch.exp(S1) / q) **2 )**(1 / 3)
        T2 = 2/3 *((torch.exp(S2) / (2 - q)) ** 2)**(1 / 3)
        #print(T2)
       

        z1 = torch.zeros_like(T1)
        z2 = torch.zeros_like(T1)

        y = torch.stack([z1, z2, np.sqrt(10)/T1, -np.sqrt(10)/T2], dim = -1).unsqueeze(-1)
        M = y @ torch.transpose(y, -1, -2)
#         print(M.shape)
        ns = torch.tensor([[1,0,0,0],[0,1,0,0]], dtype = self.dtype, device = self.device)
        return ns, M



class GC_MNN_soft(ln.nn.Module):
    def __init__(self, layers=2, width=50, activation='relu'):
        super(GC_MNN_soft, self).__init__()
        self.fnn = ln.nn.FNN(4, 1, layers, width, activation)

    def forward(self, x):
        M = self.M(x)
        x = x.requires_grad_(True)
        E = self.fnn(x)
        dE = grad(E, x)
        if len(dE.size()) == 1:
            dE = dE.unsqueeze(0)
        return dE, M

    def M(self, x):
        q, _, S1, S2 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
#         T1 = 2 / 3 * torch.abs(torch.exp(Sß1) / q) ** (2 / 3)
#         T2 = 2 / 3 * torch.abs(torch.exp(S2) / (2 - q)) ** (2 / 3)
#         T1 = 2 / 3 * (torch.exp(S1) / q) ** (2 / 3)
#         T2 = 2 / 3 * (torch.exp(S2) / (2 - q)) ** (2 / 3)
        T1 = 2/3 *((torch.exp(S1) / q) **2 )**(1 / 3)
        T2 = 2/3 *((torch.exp(S2) / (2 - q)) ** 2)**(1 / 3)
        z1 = torch.zeros_like(T1)
        z2 = torch.zeros_like(T1)
        y = torch.stack([z1, z2, np.sqrt(10) / T1, -np.sqrt(10) / T2], dim=-1).unsqueeze(-1)
        M = y @ torch.transpose(y, -1, -2)
        return M
    
    
### Know L and M, learn E and S with transformation

class GC_LNN_trans(ln.nn.Module):
    '''Fully connected neural networks in the null space of L
    '''
    def __init__(self, QtU, QtU_inv, layers=2,width=50, activation='relu'):
        super(GC_LNN_trans, self).__init__()
        self.fnn = ln.nn.FNN(2, 1, layers, width, activation)
        self.QtU = QtU
        self.QtU_inv = QtU_inv
        
    def forward(self, x):
        ns, L = self.ns()
        x = x.requires_grad_(True)
        S = self.fnn(x @ ns.t())
        dS = grad(S, x)
        if len(dS.size()) == 1:
            dS = dS.unsqueeze(0)
        return dS, L
    
    def ns(self):
        L = torch.tensor([[0,1,0,0],[-1,0,0,0],[0,0,0,0],[0,0,0,0]], dtype = self.dtype, device = self.device)
        ns = torch.tensor([[0,0,1,0],[0,0,0,1]], dtype = self.dtype, device = self.device)
        L = self.QtU_inv @ L @self.QtU_inv.t()
        ns = ns @ self.QtU
        return ns, L


class GC_LNN_trans_soft(ln.nn.Module):
    '''Fully connected neural networks in the null space of L
    '''

    def __init__(self, QtU, QtU_inv, layers=2, width=50, activation='relu'):
        super(GC_LNN_trans_soft, self).__init__()
        self.fnn = ln.nn.FNN(4, 1, layers, width, activation)
        self.QtU = QtU
        self.QtU_inv = QtU_inv

    def forward(self, x):
        L = self.L()
        x = x.requires_grad_(True)
        S = self.fnn(x)
        dS = grad(S, x)
        if len(dS.size()) == 1:
            dS = dS.unsqueeze(0)
        return dS, L

    def L(self):
        L = torch.tensor([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=self.dtype,
                         device=self.device)
        L = self.QtU_inv @ L @ self.QtU_inv.t()
        return L
    
class GC_MNN_trans(ln.nn.Module):
    def __init__(self, QtU, QtU_inv, layers=2, width=50, activation='relu'):
        super(GC_MNN_trans, self).__init__()
        self.fnn = ln.nn.FNN(3, 1, layers, width, activation)
        self.QtU = QtU
        self.QtU_inv = QtU_inv
        
        
        
    def forward(self, x):
        
        
        ns, M = self.ns(x)
        x = x.requires_grad_(True)
        F = self.F(x)

        y = torch.cat([x @ ns.t(), F[:,None]], dim = -1)
        E = self.fnn(y)
        dE = grad(E, x)
        if len(dE.size()) == 1:
            dE = dE.unsqueeze(0)

        return dE, M
    
    def F(self, x):
        x = x @ self.QtU.t()
        q, _, S1, S2 = x[...,0], x[...,1], x[...,2], x[...,3]

        T1 = ((torch.exp(S1) / q) **2 )**(1 / 3)
        T2 = ((torch.exp(S2) / (2 - q)) ** 2)**(1 / 3)
        return T1 + T2
    
    def ns(self, x):
        x = x @ self.QtU.t()
        q, _, S1, S2 = x[...,0], x[...,1], x[...,2], x[...,3]

        T1 = 2/3 *((torch.exp(S1) / q) **2 )**(1 / 3)
        T2 = 2/3 *((torch.exp(S2) / (2 - q)) ** 2)**(1 / 3)
        #print(T2)
       

        z1 = torch.zeros_like(T1)
        z2 = torch.zeros_like(T1)

        y = torch.stack([z1, z2, np.sqrt(10)/T1, -np.sqrt(10)/T2], dim = -1).unsqueeze(-1)
        M = y @ torch.transpose(y, -1, -2)
        ns = torch.tensor([[1,0,0,0],[0,1,0,0]], dtype = self.dtype, device = self.device)
        M = self.QtU_inv @ M @ self.QtU_inv.t()
        ns = ns @ self.QtU
        
        return ns, M



class GC_MNN_trans_soft(ln.nn.Module):
    def __init__(self, QtU, QtU_inv, layers=2, width=50, activation='relu'):
        super(GC_MNN_trans_soft, self).__init__()
        self.fnn = ln.nn.FNN(4, 1, layers, width, activation)
        self.QtU = QtU
        self.QtU_inv = QtU_inv

    def forward(self, x):
        M = self.M(x)
        x = x.requires_grad_(True)
        E = self.fnn(x)
        dE = grad(E, x)
        if len(dE.size()) == 1:
            dE = dE.unsqueeze(0)
        return dE, M

    def M(self, x):
        q, _, S1, S2 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]

        T1 = 2/3 *((torch.exp(S1) / q) **2 )**(1 / 3)
        T2 = 2/3 *((torch.exp(S2) / (2 - q)) ** 2)**(1 / 3)
        z1 = torch.zeros_like(T1)
        z2 = torch.zeros_like(T1)
        y = torch.stack([z1, z2, np.sqrt(10) / T1, -np.sqrt(10) / T2], dim=-1).unsqueeze(-1)
        M = y @ torch.transpose(y, -1, -2)
        M = self.QtU_inv @ M @ self.QtU_inv.t()
        return M


# class GC_MNN(ln.nn.Module):
#     def __init__(self, layers=2, width=50, activation='relu'):
#         super(GC_MNN, self).__init__()
#         self.fnn = ln.nn.FNN(2, 1, layers, width, activation)
        
#     def forward(self, x):

#         ns, M = self.ns()
#         x = x.requires_grad_(True)
#         y = torch.cat([x @ ns.t()], dim = -1)
#         E = self.fnn(y)
#         dE = grad(E, x)

#         return dE, M

#     def ns(self):
#         M = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype = self.dtype, device = self.device)
#         ns = torch.tensor([[1,0, 0,0],[0,1, 0, 0]], dtype = self.dtype, device = self.device)
#         return ns, M
    
    
# class GC_MNN_soft(ln.nn.Module):
#     def __init__(self, layers=2, width=50, activation='relu'):
#         super(GC_MNN_soft, self).__init__()
#         self.fnn = ln.nn.FNN(4, 1, layers, width, activation)

#     def forward(self, x):
#         M = self.M(x)
#         x = x.requires_grad_(True)
#         E = self.fnn(x)
#         dE = grad(E, x)
#         return dE, M

#     def M(self, x):
#         M = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype = self.dtype, device = self.device)
#         return M


# Know L and M, learn E and S    
class VC_LNN(ln.nn.Module):
    '''Fully connected neural networks in the null space of L
    '''
    def __init__(self, layers=2,width=50, activation='relu'):
        super(VC_LNN, self).__init__()
        self.fnn = ln.nn.FNN(3, 1, layers, width, activation)
        
    def forward(self, x):
        ns, L = self.ns()
        x = x.requires_grad_(True)
        S = self.fnn(x @ ns.t())
        dS = grad(S, x)
        if len(dS.size()) == 1:
            dS = dS.unsqueeze(0)
        return dS, L
    
    def ns(self):
        L = torch.tensor([[0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [-1, 0, 0, 1, -1], [0, 0, -1, 0, 0],[0, 0, 1, 0, 0]], dtype = self.dtype, device = self.device)
        ns = torch.tensor([[0,1, 0,0 ,0],[1,0, 0, 1, 0],[-1,0, 0, 0, 1]], dtype = self.dtype, device = self.device)
        return ns, L


class VC_LNN_soft(ln.nn.Module):
    '''Fully connected neural networks in the null space of L
    '''

    def __init__(self, layers=2, width=50, activation='relu'):
        super(VC_LNN_soft, self).__init__()
        self.fnn = ln.nn.FNN(5, 1, layers, width, activation)

    def forward(self, x):
        L = self.L()
        x = x.requires_grad_(True)
        S = self.fnn(x)
        dS = grad(S, x)
        #dS = self.fnn(x)
        if len(dS.size()) == 1:
            dS = dS.unsqueeze(0)
        return dS, L

    def L(self):
        L = torch.tensor([[0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [-1, 0, 0, 1, -1], [0, 0, -1, 0, 0],[0, 0, 1, 0, 0]], dtype=self.dtype, device=self.device)
        return L
    
class VC_MNN(ln.nn.Module):
    '''Fully connected neural networks in the null space of L
    '''
    def __init__(self, layers=2,width=50, activation='relu'):
        super(VC_MNN, self).__init__()
        self.fnn = ln.nn.FNN(3, 1, layers, width, activation)
        
    def forward(self, x):
        ns, M = self.ns()
        x = x.requires_grad_(True)
        E = self.fnn(x @ ns.t())
        dE = grad(E, x)
        if len(dE.size()) == 1:
            dE = dE.unsqueeze(0)
        return dE, M
    
    def ns(self):
        M = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 1, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 1]], dtype = self.dtype, device = self.device)
        ns = torch.tensor([[1,0, 0,0 ,0],[0,1, 0, 0, 0],[0,0, -1, 1, 0]], dtype = self.dtype, device = self.device)
        return ns, M


class VC_MNN_soft(ln.nn.Module):
    '''Fully connected neural networks in the null space of L
    '''

    def __init__(self, layers=2, width=50, activation='relu'):
        super(VC_MNN_soft, self).__init__()
        self.fnn = ln.nn.FNN(5, 1, layers, width, activation)

    def forward(self, x):
        M = self.M()
        x = x.requires_grad_(True)
        E = self.fnn(x)
        dE = grad(E, x)
        #dE = self.fnn(x)
        
        if len(dE.size()) == 1:
            dE = dE.unsqueeze(0)
        
        return dE, M

    def M(self):
        M = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 1, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 1]], dtype=self.dtype,
                         device=self.device)
        return M


# Know nothing, learn L, M, E, S
class VC_LNN3(ln.nn.Module):
#     def __init__(self, S, ind, K, layers, width, activation):
    def __init__(self, ind, K, layers, width, activation, xi_scale):

        super(VC_LNN3, self).__init__()
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

class VC_LNN3_soft(ln.nn.Module):
    def __init__(self,ind, layers=2, width=50, activation='relu'):
        super(VC_LNN3_soft, self).__init__()
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


class VC_MNN3(ln.nn.Module):
    def __init__(self, ind, K, layers, width, activation, xi_scale):
        super(VC_MNN3, self).__init__()
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


class VC_MNN3_soft(ln.nn.Module):
    def __init__(self,ind, layers=2, width=50, activation='relu'):
        super(VC_MNN3_soft, self).__init__()
        self.ind = ind
        self.M = ln.nn.FNN(self.ind, self.ind ** 2, layers, width, activation)
        self.E = ln.nn.FNN(self.ind, 1, layers, width, activation)

    # def grad_E(self, x):
    #     x = np.squeeze(x)
    #     return grad(self.E, argnums=0)(x)

    def forward(self, x):
        M = self.M(x).reshape([-1, self.ind, self.ind])
        M = M @ torch.transpose(M, -1, -2)
        x = x.requires_grad_(True)
        E = self.E(x)
        dE = grad(E, x)

        
        #dE = vmap(self.grad_E,in_dims=0)(x)

        # dE = vmap(jacrev(self.E,argnums=0), in_dims=0)(x)
        # dE = dE.squeeze(1)
        if len(dE.size()) == 1:
            dE = dE.unsqueeze(0)
        return dE, M


class ESPNN(ln.nn.LossNN):
    ''' Most useful one:
    netE: state variable x -> dE, M, orthogonal to each other
    netS: state variable x -> dS, L, orthogonal to each other
    the loss is defined in 'criterion' function
    '''

    def __init__(self, netS, netE, dt, order=1, iters=1, kb=1, b_dim=1, fluc=False, lam=0):
        super(ESPNN, self).__init__()
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
        
        
#         print(x.shape)

        dE = dE.unsqueeze(1)
        
        dS = dS.unsqueeze(1)
#         print(dE.shape)
#         print(L.shape)

        
        

        #return -(dE @ L).squeeze() + (dS @ M).squeeze()
        return -(dE @ L).squeeze() + (dS @ M).squeeze()

    def g(self, x):
        return self.netE.B(x)

    def consistency_loss(self, x):

        
        dE, M = self.netE(x)
        dS, L = self.netS(x)
        
        dE = dE.unsqueeze(1)
        dS = dS.unsqueeze(1)
       
        
#         print(dE.shape)
#         print(dS.shape)
#         print(M.shape)
#         print(L.shape)
        
        
        dEM = dE @ M
        dSL = dS @ L
        
#         print(dEM.shape)
#         print(dSL.shape)
        
        
        return self.lam * (torch.mean(dEM ** 2) + torch.mean(dSL ** 2))

    def criterion(self, X, y):      

        X_next = self.integrator.solve(X, self.dt)

        loss = self.loss(X_next, y)

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
