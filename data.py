#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:36:11 2021

@author: zen
"""
import learner as ln
import numpy as np
import os
from learner.integrator import RK # runge kutta for deterministic system
from learner.integrator import EM # euler maruyama for stochastic system
import scipy.io

class Data(ln.Data):
    def __init__(self, split_ratio, problem, t_terminal, dt, trajs, order, iters, noise = 0, new = False, kb = 0, transform = False):
        super(Data, self).__init__()
        if problem == 'DP': # double pendulum
            self.integrator = RK(self.__double_pendulum, order=order, iters=iters)
        elif problem == 'LDP': # double pendulum, with M constant
            self.integrator = RK(self.__linear_double_pendulum, order=order, iters=iters)
        elif problem == 'GC': # gas container
            self.integrator = RK(self.__gas_container, order=order, iters=iters)
        elif problem == 'VC': # gas container
            self.integrator = RK(self.__gas_container, order=order, iters=iters)
        elif problem == 'LG': # stochastic langevin equation
            self.integrator = EM(self.__langevin_f, self.__langevin_g, order=1, iters=iters)
        elif problem == 'SPH':
            pass
        else:
            raise NotImplementedError
            
        # bolzmann constant for stochastic system, 0 for determininstic ones
        self.kb = kb 
        
        # Generate several trajectories for training and testing
        data = self.__generate_flow(problem = problem, file = problem, t_terminal = t_terminal, dt = dt, trajs = trajs, new = new)
        self.dt = data['dt']
        self.t = data['t_vec']
        self.num_t = len(self.t)
        self.dims = data['Z'].shape[-1]
        self.transform = transform
        self.noise = noise
        
        # From the trajectories data, we do the train-test split and group them into
        # data pairs for future training
        self.__generate_data(data['Z'], split_ratio)
    
    def __generate_data(self, X, split):
        # train-test split
        num_train = int(len(X)*split)
        self.train_traj = X[:num_train]
        self.test_traj = X[num_train:]
        std = np.std(self.train_traj, (0,1), keepdims = True)
        
        # add noise to the data
        self.train_traj += self.noise * np.random.randn(*self.train_traj.shape) * std
        
        # This is not used: one can make the problems more difficult by sending the data through
        # an unknown linear transformation
        if self.transform:
            self.A = np.random.randn(self.dims, self.dims)
        else:
            self.A = np.eye(self.dims)
        self.train_traj = self.trans(self.train_traj)
        self.test_traj = self.trans(self.test_traj)
        
        # group the trajectories data into input-target pairs, that can be fed directly into NNs
        X_train, y_train = self.train_traj[:,:-1], self.train_traj[:,1:]
        # print(self.train_traj.shape)(80, 401, 10)
        # print(X_train.shape)(80, 400, 10)
        #
        # print(self.test_traj.shape)(20, 401, 10)
        X_test, y_test = self.test_traj[:,:-1], self.test_traj[:,1:]
        #print(X_test.shape)(20, 400, 10)

        self.X_train = X_train.reshape([-1,self.dims])
        self.X_test = X_test.reshape([-1,self.dims])
        self.y_train = y_train.reshape([-1,self.dims])
        self.y_test = y_test.reshape([-1,self.dims])
        
    def trans(self, x):
        return x @ self.A
        
    def detrans(self, x):
        return x @ np.linalg.inv(self.A)
    
    def __gas_container(self, x):
        q, p, S1, S2 = x[...,0:1], x[...,1:2], x[...,2:3], x[...,3:]
        E1 = (np.exp(S1) / q) ** (2 / 3)
        E2 = (np.exp(S2) / (2 - q)) ** (2 / 3)
        q_dot = p
        p_dot = 2 / 3 * (E1 / q - E2 / (2 - q))
        S1_dot = 9 * 10 / 4 / E1 * (1 / E1 - 1 / E2)
        S2_dot = - 9 * 10 / 4 / E2 * (1 / E1 - 1 / E2)
        return np.concatenate([q_dot, p_dot, S1_dot, S2_dot], axis = -1)
    
    def __langevin_f(self, x):
        _, p, _ = x[...,0:1], x[...,1:2], x[...,2:]
        q_dot = p
        p_dot = - 0.5 * p
        S_dot = 0.5 * p ** 2 - 0.5 * self.kb
        return np.concatenate([q_dot, p_dot, S_dot], axis = -1)
    
    def __langevin_g(self, x):
        _, p, _ = x[...,0:1], x[...,1:2], x[...,2:]
        q_dot = 0 * p 
        p_dot = p / p * (self.kb ** 0.5)
        S_dot = - p * (self.kb ** 0.5)
        return np.concatenate([q_dot, p_dot, S_dot], axis = -1)[...,None]
    
    def __double_pendulum(self, x):
        m1, m2, c1, c2, beta, l10, l20, k, c0, theta_r = 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
        q1, q2, p1, p2, s1, s2 = x[...,:2], x[...,2:4], x[...,4:6], x[...,6:8], x[...,8:9], x[...,9:]
        q1_dot = p1 / m1
        q2_dot = p2 / m2
        lambda1 = np.linalg.norm(q1, axis = -1, keepdims = True)
        lambda2 = np.linalg.norm(q1 - q2, axis = -1, keepdims = True)
        theta1 = theta_r * np.exp((s1 - beta * np.log(lambda1 / l10))/c0)
        theta2 = theta_r * np.exp((s2 - beta * np.log(lambda2 / l20))/c0)
        p1_dot = -c1 * np.log(lambda1/l10) / (lambda1 **2) * q1 - beta * theta_r * q1 / (lambda1 **2) + theta_r * np.exp((s1 - beta * np.log(lambda1 / l10))/c0) * beta * q1 / lambda1 ** 2
        p1_dot += -c2 * np.log(lambda2/l20) / (lambda2 **2) * (q1-q2) - beta * theta_r * (q1 - q2)/ (lambda2 **2) + theta_r * np.exp((s2 - beta * np.log(lambda2 / l20))/c0) * beta * (q1 -q2) / lambda2 ** 2
        p2_dot = -c2 * np.log(lambda2/l20) / (lambda2 **2) * (q2-q1) - beta * theta_r * (q2 - q1)/ (lambda2 **2) + theta_r * np.exp((s2 - beta * np.log(lambda2 / l20))/c0) * beta * (q2 -q1) / lambda2 ** 2
        s1_dot = k *(theta2/theta1-1)
        s2_dot = k *(theta1/theta2-1)
        return np.concatenate([q1_dot, q2_dot, p1_dot, p2_dot, s1_dot, s2_dot], axis = -1)
    
    def __linear_double_pendulum(self, x):
        q1, q2, p1, p2, s1, s2 = x[...,:2], x[...,2:4], x[...,4:6], x[...,6:8], x[...,8:9], x[...,9:]
        q1_dot = p1
        q2_dot = p2
        lambda1 = np.linalg.norm(q1, axis = -1, keepdims = True)
        lambda2 = np.linalg.norm(q1 - q2, axis = -1, keepdims = True)
        p1_dot = - 2 * np.log(lambda1) / (lambda1 **2) * q1 - 2 * np.log(lambda2) / (lambda2 **2) * (q1 - q2)
        p2_dot = - 2 * np.log(lambda2) / (lambda2 **2) * (q2 - q1)
        s1_dot = 0.5 * s1 / s1
        s2_dot = - 0.25 * s2 / s2
        return np.concatenate([q1_dot, q2_dot, p1_dot, p2_dot, s1_dot, s2_dot], axis = -1)
    
    def __generate_flow(self, file, new, trajs = 3, t_terminal = 5, dt = 0.05, problem = 'GC'):
        data = {}
        path = 'data/database_{}.npy'.format(file)
        if os.path.exists(path) and (not new):
            data = np.load(path, allow_pickle=True).item()
            #print(data.shape)
            return data
        elif problem == 'VE':
            path = 'data/database_viscoelastic.mat'
            data = scipy.io.loadmat(path)
            data['Z'] = np.transpose(data['Z'], (0,2,1))[:,:150]
            data['t_vec'] = data['t_vec'][0]
            data['dt'] = data['dt'][0][0]
            np.random.shuffle(data['Z'])
            return data

        t = np.linspace(0, t_terminal, int(t_terminal / dt) + 1)
        
        # specify the initial conditions
        if problem == 'GC':
            x0 = np.array([[1,0,2,2]]) + (2*np.random.rand(trajs, 4) - 1) * np.array([[0.8,1,1,1]])
        elif problem == problem == 'LG':
            x0 = np.array([[0,0,0]]) + np.random.randn(trajs, 3) * 0.2
        elif problem == 'DP':
            x0 = np.array([[1,0,2.2,0,0,2,1,0,1,0.2]]) + (2*np.random.rand(trajs, 10) - 1)  * 0.1
        elif problem == 'LDP':
            x0 = np.array([[1,0,2.2,0,0,2,1,0,1,0.2]]) + (2*np.random.rand(trajs, 10) - 1)  * 0.15
        
        # solve the ODE using predefined numerical integrators
        Z = self.integrator.flow(x0, dt, int(t_terminal / dt))
        data['Z'] = Z
        data['dt'] = dt
        data['t_vec'] = t
        if not os.path.exists('data'): os.mkdir('data')
        if not os.path.exists(path): np.save(path, data)
        return data
        
def test_lg():
    np.random.seed(1)   
    p = 0.5
    problem = 'LG'
    t_terminal = 1
    dt = 0.004
    trajs = 20
    order = 1
    iters = 1
    data = Data(p, problem, t_terminal, dt, trajs, order, iters, new = True, kb = 1)
    import matplotlib.pyplot as plt
    print(data.train_traj.shape)
    x = data.train_traj[2]
    plt.plot(data.t, x[:,0], label = 'q')
    plt.plot(data.t, x[:,1], label = 'p')
    plt.plot(data.t, x[:,2], label = 'S')
    print((data.y_train - data.X_train)[:500,-1])
    plt.plot(data.t, x[:,1] ** 2/2+ x[:,2], label = 'E')
    plt.legend()
    plt.tight_layout()
    plt.xlabel('t')
    plt.ylabel('One sample path')
    plt.savefig('figs/lg_sample.pdf')
    
def test_gc():
    #np.random.seed(1)   
    p = 0.5
    problem = 'GC'
    t_terminal = 8
    dt = 0.02
    trajs = 50
    order = 2
    iters = 1
    data = Data(p, problem, t_terminal, dt, trajs, order, iters, new = True, noise = 0)
    import matplotlib.pyplot as plt
    print(data.train_traj.shape)
    x = data.train_traj[2]
    plt.plot(data.t, x[:,0], label = 'q')
    plt.plot(data.t, x[:,1], label = 'p')
    plt.plot(data.t, x[:,2], label = '$S_1$')
    plt.plot(data.t, x[:,3], label = '$S_2$')
    plt.legend()
    plt.tight_layout()
    plt.xlabel('t')
    plt.ylabel('One sample path')
    plt.savefig('figs/gc_sample.pdf')
  
if __name__ == '__main__':
    #test_lg()
    test_gc()
    
        
