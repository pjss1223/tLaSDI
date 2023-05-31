#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 17:51:26 2021

@author: zen
"""

import numpy as np
import torch


class RK_parain:
    '''Runge-Kutta scheme.
    '''

    def __init__(self, f, para_dim, order=4, iters=1):
        self.f = f
        self.iters = iters
        self.order = order
        self.dim_mu = para_dim

        if self.order == 1:
            self.solver = self.euler
        elif self.order == 2:
            self.solver = self.rk23
        elif self.order == 4:
            self.solver = self.rk45
        else:
            raise NotImplementedError

    def euler(self, x, h):
        dt = h / self.iters
        for _ in range(self.iters):
            x[:,:-self.dim_mu] = x[:,:-self.dim_mu] + dt * self.f(x)
        return x[:,:-self.dim_mu]

    def rk23(self, x, h):
        dt = h / self.iters
        for _ in range(self.iters):

            # print(x.shape)#1000 12
            #print(self.f(x).shape)  # 1000 10?
            k1 = dt * self.f(x)
            # print(k1.shape)#1000, 10

            # print(torch.zeros(self.f(x).shape[0],self.dim_mu).shape)#2,2
            # print(k1.shape)#2
            k1 = torch.cat((k1, torch.zeros(self.f(x).shape[0],self.dim_mu)),1)
            k2 = dt * self.f(x + k1)
            k2 = torch.cat((k2, torch.zeros(self.f(x).shape[0], self.dim_mu)),1)
            k3 = dt * self.f(x + 0.25 * k1 + 0.25 * k2)
            k3 = torch.cat((k3, torch.zeros(self.f(x).shape[0], self.dim_mu)),1)
            # x[:,:-self.dim_mu] = x[:,:-self.dim_mu] + (k1 + k2 + 4 * k3) / 6
            x = x + (k1 + k2 + 4 * k3) / 6
        return x

    def rk45(self, x, h):
        dt = h / self.iters
        for _ in range(self.iters):
            k1 = dt * self.f(x)
            k2 = dt * self.f(x + 0.25 * k1)
            k3 = dt * self.f(x + 3 / 32 * k1 + 9 / 32 * k2)
            k4 = dt * self.f(x + 1932 / 2197 * k1 - 7200 / 2197 * k2 + 7296 / 2197 * k3)
            k5 = dt * self.f(x + 439 / 216 * k1 - 8 * k2 + 3680 / 513 * k3 - 845 / 4104 * k4)
            k6 = dt * self.f(x - 8 / 27 * k1 + 2 * k2 - 3544 / 2565 * k3 + 1859 / 4104 * k4 - 11 / 40 * k5)
            x[:,:-self.dim_mu] = x[:,:-self.dim_mu] + 16 / 135 * k1 + 6656 / 12825 * k3 + 28561 / 56430 * k4 - 9 / 50 * k5 + 2 / 55 * k6
        return x[:,:-self.dim_mu]

    def solve(self, x, h):
        #print(x.shape) #1000, 12
        x = self.solver(x, h)
        return x

    def flow(self, x, h, steps):
        dim = x.shape[-1] if isinstance(x, np.ndarray) else x.size(-1)
        size = len(x.shape) if isinstance(x, np.ndarray) else len(x.size())
        X = [x]
        for i in range(steps):
            if isinstance(x, torch.Tensor):
                X[-1] = X[-1].detach()
            X.append(self.solve(X[-1], h))
        shape = [steps + 1, dim] if size == 1 else [-1, steps + 1, dim]
        return np.hstack(X).reshape(shape) if isinstance(x, np.ndarray) else torch.cat(X, dim=-1).view(shape)