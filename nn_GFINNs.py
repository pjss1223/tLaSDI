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

# Know nothing, learn L, M, E, S
class VC_LNN3(ln.nn.Module):
    #def __init__(self,ind,extraD, layers=2, width=50, activation='tanh'):
    def __init__(self, ind, extraD, layers=5, width=24, activation='relu'):

        super(VC_LNN3, self).__init__()
        self.ind = ind
        self.extraD = extraD
        self.fnn = ln.nn.FNN(self.ind, 1, layers, width, activation)
        #print(width)
        self.sigComp = ln.nn.FNN(self.ind, self.extraD * self.extraD, layers, width, activation)

        #default


        self.Xi1 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.01).requires_grad_(True))
        self.Xi2 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.1).requires_grad_(True))
        self.Xi3 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 1.0).requires_grad_(True))
        self.Xi4 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.1).requires_grad_(True))
        self.Xi5 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.01).requires_grad_(True))
        self.Xi6 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.01).requires_grad_(True))
        self.Xi7 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.1).requires_grad_(True))
        self.Xi8 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 1.0).requires_grad_(True))
        self.Xi9 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.1).requires_grad_(True))
        self.Xi10 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.01).requires_grad_(True))



        # path = './outputs/'
        # # torch.save({'Xi1':self.Xi1,'Xi2':self.Xi2,'Xi3':self.Xi3,'Xi4':self.Xi4,'Xi5':self.Xi5,'Xi6':self.Xi6,'Xi7':self.Xi7,'Xi8':self.Xi8,'Xi9':self.Xi9,'Xi10':self.Xi10},path + '/Xis_L.p')
        #
        # Xis = torch.load(path + '/Xis_L.p')
        #
        # self.Xi1 = Xis['Xi1']
        # self.Xi2 = Xis['Xi2']
        # self.Xi3 = Xis['Xi3']
        # self.Xi4 = Xis['Xi4']
        # self.Xi5 = Xis['Xi5']
        # self.Xi6 = Xis['Xi6']
        # self.Xi7 = Xis['Xi7']
        # self.Xi8 = Xis['Xi8']
        # self.Xi9 = Xis['Xi9']
        # self.Xi10 = Xis['Xi10']





    def forward(self, x):
        sigComp = self.sigComp(x).reshape(-1, self.extraD, self.extraD)

        sigma = sigComp - torch.transpose(sigComp, -1, -2)


        Xi1 = self.Xi1
        Xi1 = Xi1 - torch.transpose(Xi1, -1, -2)
        Xi2 = self.Xi2
        Xi2 = Xi2 - torch.transpose(Xi2, -1, -2)
        Xi3 = self.Xi3
        Xi3 = Xi3 - torch.transpose(Xi3, -1, -2)
        Xi4 = self.Xi4
        Xi4 = Xi4 - torch.transpose(Xi4, -1, -2)
        Xi5 = self.Xi5
        Xi5 = Xi5 - torch.transpose(Xi5, -1, -2)
        Xi6 = self.Xi6
        Xi6 = Xi6 - torch.transpose(Xi6, -1, -2)
        Xi7 = self.Xi7
        Xi7 = Xi7 - torch.transpose(Xi7, -1, -2)
        Xi8 = self.Xi8
        Xi8 = Xi8 - torch.transpose(Xi8, -1, -2)
        Xi9 = self.Xi9
        Xi9 = Xi9 - torch.transpose(Xi9, -1, -2)
        Xi10 = self.Xi10
        Xi10 = Xi10 - torch.transpose(Xi10, -1, -2)

        dS = self.ns(x)
        ddS = dS.unsqueeze(-2)
        #B = torch.cat([ddS @ Xi1, ddS @ Xi2, ddS @ Xi3, ddS @ Xi4, ddS @ Xi5], dim=-2)
        #B = torch.cat([ddS @ Xi1, ddS @ Xi2, ddS @ Xi3, ddS @ Xi4, ddS @ Xi5, ddS @ Xi6, ddS @ Xi7, ddS @ Xi8], dim=-2)
        #B = torch.cat([ddS @ Xi1, ddS @ Xi2, ddS @ Xi3, ddS @ Xi4, ddS @ Xi5, ddS @ Xi6, ddS @ Xi7, ddS @ Xi8, ddS @ Xi9], dim=-2)
        # print(ddS.shape)
        # print(Xi1.shape)



        B = torch.cat([ddS @ Xi1, ddS @ Xi2, ddS @ Xi3, ddS @ Xi4, ddS @ Xi5, ddS @ Xi6, ddS @ Xi7, ddS @ Xi8, ddS @ Xi9, ddS @ Xi10], dim=-2)
        #print(B.shape)

        L = torch.transpose(B, -1, -2) @ sigma @ B
        return dS, L

    def grad_fnn(self, x):
        x = np.squeeze(x)
        return grad(self.fnn, argnums=0)(x)

    def ns(self, x):
        x = x.requires_grad_(True)
        #print(x.shape) #20x1
        S = self.fnn(x)
        #print(S.shape) #20x1
        dS = grad(S, x)
        #print(dS.shape)#119 10

        #dS = vmap(self.grad_fnn, in_dims=0)(x)

        #dS = vmap(jacrev(self.fnn, argnums=0), in_dims=0)(x)
        #dS = dS.squeeze(1)
        #print(dS.shape)
        if len(dS.size()) == 1:
            dS = dS.unsqueeze(0)
        return dS


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
    #def __init__(self, ind, extraD, layers=2, width=50, activation='tanh'):
    def __init__(self, ind, extraD, layers=5, width=24, activation='relu'):

        super(VC_MNN3, self).__init__()
        self.ind = ind
        self.extraD = extraD
        self.fnnB = ln.nn.FNN(self.ind, self.extraD * self.extraD, layers, width, activation)
        self.fnn = ln.nn.FNN(self.ind, 1, layers, width, activation)



        self.Xi1 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.01).requires_grad_(True))
        self.Xi2 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.1).requires_grad_(True))
        self.Xi3 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 1.0).requires_grad_(True))
        self.Xi4 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.1).requires_grad_(True))
        self.Xi5 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.01).requires_grad_(True))
        self.Xi6 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.01).requires_grad_(True))
        self.Xi7 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.1).requires_grad_(True))
        self.Xi8 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 1.0).requires_grad_(True))
        # # self.Xi9 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.1).requires_grad_(True))
        # # self.Xi10 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.01).requires_grad_(True))
        # #self.Xi11 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.01).requires_grad_(True))
        # # self.Xi12 = torch.nn.Parameter((torch.randn([self.ind, self.ind]) * 0.01).requires_grad_(True))


        # path = './outputs/'
        #
        # #torch.save({'Xi1':self.Xi1,'Xi2':self.Xi2,'Xi3':self.Xi3,'Xi4':self.Xi4,'Xi5':self.Xi5,'Xi6':self.Xi6,'Xi7':self.Xi7,'Xi8':self.Xi8},path + '/Xis_M.p')
        # Xis = torch.load(path + '/Xis_M.p')
        #
        # self.Xi1 = Xis['Xi1']
        # self.Xi2 = Xis['Xi2']
        # self.Xi3 = Xis['Xi3']
        # self.Xi4 = Xis['Xi4']
        # self.Xi5 = Xis['Xi5']
        # self.Xi6 = Xis['Xi6']
        # self.Xi7 = Xis['Xi7']
        # self.Xi8 = Xis['Xi8']



    def forward(self, x):
        Xi1 = self.Xi1
        Xi1 = (Xi1 - torch.transpose(Xi1, -1, -2))
        Xi2 = self.Xi2
        Xi2 = (Xi2 - torch.transpose(Xi2, -1, -2))
        Xi3 = self.Xi3
        Xi3 = Xi3 - torch.transpose(Xi3, -1, -2)
        Xi4 = self.Xi4
        Xi4 = Xi4 - torch.transpose(Xi4, -1, -2)
        Xi5 = self.Xi5
        Xi5 = (Xi5 - torch.transpose(Xi5, -1, -2))
        Xi6 = self.Xi6
        Xi6 = (Xi6 - torch.transpose(Xi6, -1, -2))
        Xi7 = self.Xi7
        Xi7 = Xi7 - torch.transpose(Xi7, -1, -2)
        Xi8 = self.Xi8
        Xi8 = Xi8 - torch.transpose(Xi8, -1, -2)


        dE = self.ns(x)
        ddE = dE.unsqueeze(-2)
        B = torch.cat([ddE @ Xi1, ddE @ Xi2, ddE @ Xi3, ddE @ Xi4, ddE @ Xi5, ddE @ Xi6, ddE @ Xi7, ddE @ Xi8], dim=-2)

        #print(B.shape) #800 8 10

        sigComp = self.fnnB(x).reshape(-1, self.extraD, self.extraD)

        sigma = sigComp @ torch.transpose(sigComp, -1, -2)
        M = torch.transpose(B, -1, -2) @ sigma @ B
        return dE, M

    # def grad_fnn(self, x):
    #     print(x)
    #
    #     return grad(self.fnn, argnums=0)(x)

    def ns(self, x):
        x = x.requires_grad_(True)
        #print(x.shape) #20x9, 20x1
        E = self.fnn(x)
        #print(E.shape) #20x1, 20x20
        dE = grad(E, x)

        #dE = vmap(self.grad_fnn,in_dims=0)(x)

        # dE = vmap(jacrev(self.fnn), in_dims=0)(x)
        # dE = dE.squeeze(1)
        if len(dE.size()) == 1:
            dE = dE.unsqueeze(0)
        #print(dE.shape)
        return dE


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
   #     print(dE.shape)
        # print(M.shape)
        dS, L = self.netS(x)
        # print('de',dE) blow up is due to M and L
        # print('M',M)
        # print('ds',dS)
        # print('L',L)
        dE = dE.unsqueeze(1)
        #print(dE.shape)
        dS = dS.unsqueeze(1)

        # print(dE @ L)
        # print(dS @ M)


        #return -(dE @ L).squeeze() + (dS @ M).squeeze()
        return (dE @ L).squeeze() + (dS @ M).squeeze()

    def g(self, x):
        return self.netE.B(x)

    def consistency_loss(self, x):
        dE, M = self.netE(x)
        dS, L = self.netS(x)
        dEM = dE @ M
        dSL = dS @ L
        return self.lam * (torch.mean(dEM ** 2) + torch.mean(dSL ** 2))

    def criterion(self, X, y):
        #print(X.shape)
        #print(self.dt)
        X_next = self.integrator.solve(X, self.dt)
        #print(X_next)

        loss = self.loss(X_next, y)
        if self.lam > 0:
            loss += self.consistency_loss(X)
        return loss

    def integrator2(self, X):
        X_next = self.integrator.solve(X, self.dt)
        return X_next

    def predict(self, x0, k, return_np=False):
        x = torch.transpose(self.integrator.flow(x0, self.dt, k - 1), 0, 1)
        if return_np:
            x = x.detach().cpu().numpy()
        return x
