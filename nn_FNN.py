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


class FNN_latent(ln.nn.Module):
    #def __init__(self,ind,extraD, layers=2, width=50, activation='tanh'):
    def __init__(self, ind,dt, layers=5, width=24,order=1, iters=1, activation='relu'):

        super(FNN_latent, self).__init__()
        self.ind = ind
        self.fnn = ln.nn.FNN(self.ind, self.ind, layers, width, activation)
        self.integrator = RK(self.forward, order=order, iters=iters)
        self.loss = mse
        self.dt = dt


    def forward(self, x):

        F = self.fnn(x)

        return F

    def criterion(self, X, y):
        #print(X.shape)
        #print(self.dt)
        X_next = self.integrator.solve(X, self.dt)
        #print(X_next)

        loss = self.loss(X_next, y)

        return loss

    def integrator2(self, X):

        X_next = self.integrator.solve(X, self.dt)
        #print(X_next)
        return X_next

