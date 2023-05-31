#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 20:43:06 2021

@author: zen
"""


import numpy as np
import torch

class EM:
    '''Euler-Maruyama scheme.
    '''
    def __init__(self, f, g, order=1, iters=1, b_dim = 1):
        self.f = f
        self.g = g
        self.iters = iters
        self.order = order
        self.b_dim = b_dim
        if self.order == 1:
            self.solver = self.euler
        else:
            raise NotImplementedError
        
    def euler(self, x, h, noise):
        dt = h / self.iters
        if not isinstance(noise, np.ndarray) and not isinstance(noise, torch.Tensor):
            noise = torch.randn(self.iters, x.shape[0], 1, self.b_dim, device = x.device, dtype = x.dtype) * np.sqrt(dt)
                
        for i in range(self.iters):
            #print(x.shape, self.f(x).shape, self.g(x).shape, noise[i].shape)
            x = x + dt * self.f(x) + (self.g(x) * noise[i]).sum(-1)
        return x

    def solve(self, x, h, noise = None):
        x = self.solver(x, h, noise)
        return x
    
    def flow(self, x, h, steps, noise = None):
        dim = x.shape[-1] if isinstance(x, np.ndarray) else x.size(-1)
        size = len(x.shape) if isinstance(x, np.ndarray) else len(x.size())
        X = [x]
        dt = h / self.iters
        if isinstance(x, np.ndarray):
            noise = np.random.randn(steps, self.iters, x.shape[0], 1, self.b_dim) * np.sqrt(dt)
        for i in range(steps):
            if isinstance(x, np.ndarray):
                X.append(self.solve(X[-1], h, noise[i]))
            else:
                X.append(self.solve(X[-1], h, noise[i]).detach())
        shape = [steps + 1, dim] if size == 1 else [-1, steps + 1, dim]
        return np.hstack(X).reshape(shape) if isinstance(x, np.ndarray) else torch.cat(X, dim=-1).view(shape)