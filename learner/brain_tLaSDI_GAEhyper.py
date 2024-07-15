"""
@author: jpzxshi & zen
"""
import os
import time
import numpy as np
import torch

from .nn import LossNN
from .utils import timing, cross_entropy_loss
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

import seaborn as sns
import matplotlib.patches as patches
from copy import deepcopy

import torch
import torch.optim as optim
import numpy as np
import copy
from scipy import sparse as sp

from model_AEhyper import AutoEncoder 
from dataset_sim_hyper import load_dataset
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

class Brain_tLaSDI_GAEhyper:
    '''Runner based on torch.
    '''
    brain = None

    @classmethod
    def Init(cls,  net, sys_name, output_dir, save_plots, criterion, optimizer, lr,
             epochs, AE_name,dset_dir,output_dir_AE,save_plots_AE,layer_vec_AE,
             activation_AE,depth_hyper, width_hyper, act_hyper, num_sensor,lr_AE,lambda_r_AE,lambda_jac_AE,lambda_dx,lambda_dz,miles_lr = [10000],gamma_lr = 1e-1, path=None, load_path = None, batch_size=None,
             batch_size_test=None, weight_decay=0,weight_decay_AE=0,update_epochs=1000, print_every=1000, save=False, load=False, callback=None, dtype='float',
             device='cpu',tol = 1e-3, tol2 = 2, adaptive = 'reg_max',n_train_max = 30,subset_size_max=80,trunc_period =1):
        cls.brain = cls( net, sys_name, output_dir, save_plots, criterion,
                         optimizer, lr, weight_decay,weight_decay_AE, epochs, AE_name,dset_dir,output_dir_AE,save_plots_AE,layer_vec_AE,
                         activation_AE,depth_hyper, width_hyper, act_hyper, num_sensor,lr_AE,lambda_r_AE,lambda_jac_AE,lambda_dx,lambda_dz,miles_lr,gamma_lr, path,load_path, batch_size,
                         batch_size_test, update_epochs, print_every, save, load, callback, dtype, device, tol, tol2,adaptive,n_train_max,subset_size_max,trunc_period)

    @classmethod
    def Run(cls):
        cls.brain.run()

    @classmethod
    def Restore(cls):
        cls.brain.restore()

    @classmethod
    def Test(cls):
        cls.brain.test()

    @classmethod
    def Output(cls, best_model=True, loss_history=True, info=None, **kwargs):
        cls.brain.output( best_model, loss_history, info, **kwargs)

    @classmethod
    def Loss_history(cls):
        return cls.brain.loss_history

    @classmethod
    def Encounter_nan(cls):
        return cls.brain.encounter_nan

    @classmethod
    def Best_model(cls):
        return cls.brain.best_model

    def __init__(self,  net,sys_name, output_dir,save_plots, criterion, optimizer, lr, weight_decay,weight_decay_AE, epochs, AE_name,dset_dir,output_dir_AE,save_plots_AE,layer_vec_AE,
             activation_AE,depth_hyper, width_hyper, act_hyper, num_sensor,lr_AE,lambda_r_AE,lambda_jac_AE,lambda_dx,lambda_dz,miles_lr,gamma_lr, path, load_path, batch_size,
                 batch_size_test, update_epochs, print_every, save, load, callback, dtype, device, tol, tol2, adaptive,n_train_max,subset_size_max,trunc_period):
        self.net = net
        self.sys_name = sys_name
        self.output_dir = output_dir
        self.save_plots = save_plots
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr
        self.lr_AE = lr_AE
        self.weight_decay = weight_decay
        self.weight_decay_AE = weight_decay_AE

        self.epochs = epochs
        self.path = path
        self.load_path = load_path
        self.batch_size = batch_size
        self.batch_size_test = batch_size_test
        self.print_every = print_every
        self.save = save
        self.load = load
        self.callback = callback
        
        self.dtype = dtype
        self.device = device
        self.dtype_torch = torch.float32 if dtype == 'float' else torch.float64 
        self.device_torch = torch.device("cuda") if device == 'gpu' else torch.device("cpu")

        self.AE_name = AE_name
        self.n_train_max = n_train_max
        self.subset_size_max = subset_size_max
        self.update_epochs = update_epochs
        
        self.miles_lr = miles_lr
        self.gamma_lr = gamma_lr
        
    
        if self.load:
            path = './outputs/' + self.load_path
            loss_history_value= torch.load( path + '/loss_history_value.p')
            self.lr = loss_history_value['lr_final']
            self.lr_AE = loss_history_value['lr_AE_final']

        else:    
            self.lr = lr
            self.lr_AE = lr_AE
        


        #update tol adaptive method
        self.adaptive = adaptive
        #self.dset_dir = dset_dir
        self.output_dir_AE = output_dir_AE
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self.save_plots_AE = save_plots_AE
        self.tol = tol
        self.tol2 = tol2
        self.trunc_period = trunc_period
        if self.load:
            path = './outputs/' + self.load_path
            self.AE = torch.load( path + '/model_best_AE.pkl')
            self.net = torch.load( path + '/model_best.pkl')
            
            self.AE = self.AE.to(dtype=self.dtype_torch, device=self.device_torch)
            self.net = self.net.to(dtype=self.dtype_torch, device=self.device_torch)

        else:
 
            self.AE = AutoEncoder(layer_vec_AE, activation_AE,depth_hyper, width_hyper, act_hyper, num_sensor).to(dtype=self.dtype_torch, device=self.device_torch)

        # ALL parameters --------------------------------------------------------------------------------

        if self.sys_name == '1DBurgers':
               
            self.num_test = 441
            self.num_train = 4 # initial num_train
            self.err_type = 2  # residual error indicator for 1DBurgers

            amp_test = np.linspace(0.7, 0.8, 21)
            width_test = np.linspace(0.9, 1.0, 21)
            amp_train = np.linspace(0.7, 0.8, 2)
            width_train = np.linspace(0.9, 1.0, 2)
            
            self.amp_test = amp_test
            self.width_test = width_test

        if self.sys_name == '1DHeat':

            self.num_test = 441
            self.num_train = 4 # initial num_train
            self.err_type = 3  # residual error indicator for 1DHeat

            amp_test = np.linspace(0.7, 0.8, 21)
            width_test = np.linspace(0.9, 1.0, 21)
            amp_train = np.linspace(0.7, 0.8, 2)
            width_train = np.linspace(0.9, 1.0, 2)

            self.amp_test = amp_test
            self.width_test = width_test

        grid2, grid1 = np.meshgrid(width_train, amp_train)
        train_param = np.hstack((grid1.flatten().reshape(-1, 1), grid2.flatten().reshape(-1, 1)))
        grid2, grid1 = np.meshgrid(width_test, amp_test)
        test_param = np.hstack((grid1.flatten().reshape(-1, 1), grid2.flatten().reshape(-1, 1)))

        train_indices = []
        for i in range(self.num_test):
            for j in range(self.num_train):
                if np.abs(test_param[i, 0] - train_param[j, 0]) < 1e-8 and \
                        np.abs(test_param[i, 1] - train_param[j, 1]) < 1e-8:
                    train_indices.append(i)
        
        self.test_param = test_param

        self.train_indices = train_indices
        self.test_indices = np.arange(self.num_test)
        
        if self.load:
            path = './outputs/' + self.load_path
            tr_indices = torch.load(path + '/train_indices.p')            
            self.train_indices = tr_indices['train_indices']
            print(self.train_indices)
            self.num_train = len(self.train_indices)

        self.dset_dir = dset_dir

        # Dataset Parameters
        self.dataset = load_dataset(self.sys_name, self.dset_dir,self.device,self.dtype)
        self.dt = self.dataset.dt
        self.dx = self.dataset.dx
        self.dim_t = self.dataset.dim_t
        self.dim_z = self.dataset.dim_z
        self.mu1 = self.dataset.mu
        
        self.dim_mu = self.dataset.dim_mu

        self.mu_tr1 = self.mu1[self.train_indices,:]

        self.mu = torch.repeat_interleave(self.mu1, self.dim_t, dim=0)
        self.mu_tr = torch.repeat_interleave(self.mu_tr1,self.dim_t-1,dim=0)

        self.mu_tr_all = torch.repeat_interleave(self.mu_tr1,self.dim_t,dim=0)

        self.z_gt = torch.from_numpy(np.array([]))
        self.z_tr = torch.from_numpy(np.array([]))
        self.z1_tr = torch.from_numpy(np.array([]))
        self.z_tr_all = torch.from_numpy(np.array([]))
        self.dz_tr = torch.from_numpy(np.array([]))

        for j in range(self.mu1.shape[0]):
            self.z_gt = torch.cat((self.z_gt,torch.from_numpy(self.dataset.py_data['data'][j]['x'])),0)

        for j in self.train_indices:
            self.z_tr = torch.cat((self.z_tr,torch.from_numpy(self.dataset.py_data['data'][j]['x'][:-1,:])),0)
            self.z1_tr = torch.cat((self.z1_tr, torch.from_numpy(self.dataset.py_data['data'][j]['x'][1:,:])), 0)
            self.z_tr_all = torch.cat((self.z_tr_all, torch.from_numpy(self.dataset.py_data['data'][j]['x'])), 0)
            self.dz_tr = torch.cat((self.dz_tr, torch.from_numpy(self.dataset.py_data['data'][j]['dx'][:-1, :])), 0)

        self.z_gt = self.z_gt.to(dtype=self.dtype_torch, device=self.device_torch)
        self.z_tr = self.z_tr.to(dtype=self.dtype_torch, device=self.device_torch)
        self.z1_tr = self.z1_tr.to(dtype=self.dtype_torch, device=self.device_torch)
        self.z_tr_all = self.z_tr_all.to(dtype=self.dtype_torch, device=self.device_torch)
        self.dz_tr = self.dz_tr.to(dtype=self.dtype_torch, device=self.device_torch)

        self.lambda_r = lambda_r_AE
        self.lambda_jac = lambda_jac_AE
        self.lambda_dx = lambda_dx
        self.lambda_dz = lambda_dz

        self.loss_history = None
        self.encounter_nan = False
        self.best_model = None

        self.__optimizer = None
        self.__criterion = None

    @timing
    def run(self):
        self.__init_brain()
        print('Training...', flush=True)

        testing_losses = []
        err_array = []
        err_max_para = []
        num_train = self.num_train
        
        if self.load:
            path = './outputs/' + self.load_path
            tr_indices = torch.load(path + '/train_indices.p')
            
            err_max_para = [self.mu1[self.train_indices,:]]
            err_array = tr_indices['err_array']

        if self.load:
            path = './outputs/' + self.load_path
            loss_history_value= torch.load( path + '/loss_history_value.p')
            loss_history = loss_history_value['loss_history']
            loss_GFINNs_history = loss_history_value['loss_GFINNs_history']
            loss_AE_history = loss_history_value['loss_AE_history']
            loss_AE_jac_history = loss_history_value['loss_AE_jac_history']
            loss_dx_history = loss_history_value['loss_dx_history']
            loss_dz_history = loss_history_value['loss_dz_history']
            i_loaded = loss_history[-1][0]
        else:
            loss_history = []
            loss_GFINNs_history = []
            loss_AE_history = []
            loss_AE_jac_history = []
            loss_dx_history = []
            loss_dz_history = []
            i_loaded = 0
        #initial training, testing data (normalized)

        z_gt_tr = self.z_tr
        self.z_tr = None

        z1_gt_tr = self.z1_tr
        self.z1_tr = None

        dz_gt_tr = self.dz_tr
        self.dz_tr = None

        z_gt_tr_all = self.z_tr_all
        self.z_tr_all = None

        mu_tr1 = self.mu_tr1

        mu_tr = self.mu_tr

        if (self.dim_t-1) % self.batch_size ==0:
            self.batch_num = (self.dim_t-1) / self.batch_size
        else:
            self.batch_num = ((self.dim_t-1) // self.batch_size) +1
            
#         self.batch_num = ((self.dim_t-1) // self.batch_size)
        Loss_early = 1e-10
        self.batch_num = int(self.batch_num)

        best_loss = float('inf')  # Initialize the best loss as infinity
        best_model = None
        best_model_AE = None

        w = 1
        prev_lr = self.__optimizer.param_groups[0]['lr']
        for i in tqdm(range(self.epochs + 1)):

            for batch in range(self.batch_num):
                start_idx = batch * self.batch_size
                end_idx = (batch + 1) * self.batch_size
                if batch == self.batch_num-1:
                    end_idx = self.dim_t-1
                
                row_indices_batch = torch.cat([torch.arange(idx_r+start_idx, idx_r + end_idx) for idx_r in range(0, z_gt_tr.size(0), self.dim_t-1)])
            #
                z_gt_tr_batch = z_gt_tr[row_indices_batch,:]
                
                mu_tr_batch = mu_tr[row_indices_batch,:]

                z1_gt_tr_batch = z1_gt_tr[row_indices_batch,:]

                dz_gt_tr_batch = dz_gt_tr[row_indices_batch,:]
            #
                z_ae_tr, X_train = self.AE(z_gt_tr_batch, mu_tr_batch)

                z1_ae_tr, y_train = self.AE(z1_gt_tr_batch, mu_tr_batch)

                mu_train = mu_tr_batch

                loss_GFINNs = self.__criterion(self.net(X_train), y_train)

                # reconstruction loss
                loss_AE = torch.mean((z_ae_tr - z_gt_tr_batch) ** 2)

                if  ((self.lambda_jac == 0 and self.lambda_dx == 0) and self.lambda_dz == 0): 
                    loss_AE_jac = torch.tensor(0, dtype=torch.float64)
                    loss_dx = torch.tensor(0, dtype=torch.float64)
                    loss_dz = torch.tensor(0, dtype=torch.float64)

                else:

                    dx_train = self.net.f(X_train)
        
                    dz_train, dx_data_train, dz_train_dec , idx_trunc = self.AE.JVP(z_gt_tr_batch, X_train, dz_gt_tr_batch, dx_train, mu_train, self.trunc_period)
                    
#                   # consistency loss
                    loss_dx = torch.mean((dx_train - dx_data_train) ** 2)

                    loss_AE_jac = torch.mean((dz_gt_tr_batch[:, idx_trunc] - dz_train) ** 2)

                    loss_dz = torch.mean((dz_gt_tr_batch[:, idx_trunc] - dz_train_dec) ** 2)

                loss = loss_GFINNs+self.lambda_r*loss_AE+ self.lambda_dx*loss_dx +self.lambda_dz*loss_dz+ self.lambda_jac*loss_AE_jac

                if i < self.epochs:
                    self.__optimizer.zero_grad()
                    loss.backward(retain_graph=False)
                    self.__optimizer.step()
                    
            self.__scheduler.step()

            self.N_subset = int(0.4 * self.num_test)

            param_flag = True

            err_max = torch.tensor([float('nan')])
            if i % self.update_epochs == 0 and i != 0:

                # select a random subset for evaluation
                rng = np.random.default_rng()
                a = np.setdiff1d(np.arange(self.num_test), self.train_indices)  # exclude existing training cases
                rng.shuffle(a)
                subset = a[:self.N_subset]

                err_array_tmp = np.zeros([self.num_test, 1])
                for i_test in np.arange(self.num_test):
                    if i_test in subset:
                        z_subset = torch.from_numpy(self.dataset.py_data['data'][i_test]['x']).to(dtype=self.dtype_torch, device=self.device_torch)
                        z0_subset = z_subset[0,:].unsqueeze(0).to(dtype=self.dtype_torch, device=self.device_torch)

                        mu0 = self.mu1[i_test, :].unsqueeze(0)

                        with torch.no_grad():
                            _,x0_subset = self.AE(z0_subset,mu0)

                        x_net_subset = torch.zeros(self.dim_t, x0_subset.shape[1]).to(dtype=self.dtype_torch, device=self.device_torch)

                        x_net_subset[0,:] = x0_subset

                        x0_subset = x0_subset

                        for snapshot in range(self.dim_t - 1):

                            x1_net = self.net.integrator2(self.net(x0_subset))

                            x_net_subset[snapshot + 1, :] = x1_net

                            x0_subset = x1_net
                        
                        with torch.no_grad():
                            z_ae_subset = self.AE.decode(x_net_subset,mu0.squeeze(0).repeat(self.dim_t,1))

                        n_s = int(1*self.dim_t)
                        err_array_tmp[i_test] = self.err_indicator(z_ae_subset[:n_s],z_subset[:n_s],self.err_type)

                    else:
                        err_array_tmp[i_test] =-1

                #maximum residual errors
                err_max = err_array_tmp.max() # maximum relative error measured in 'subset'
                err_idx = np.argmax(err_array_tmp)
                err_max_para_tmp = self.mu1[err_idx, :]

                testing_losses.append(err_max)
                err_array.append(err_array_tmp)

                err_max_para.append(err_max_para_tmp)

                #update tolerance
                tol_old = self.tol

                err_res_training = np.zeros(num_train)  # residual norm
                err_max_training = np.zeros(num_train)  # max relative error

                for i_train in range(num_train):

                    z0_train_tmp = z_gt_tr_all[i_train*(self.dim_t),:]
                    mu_tmp = mu_tr1[i_train].unsqueeze(0)
                    z0_train_tmp = z0_train_tmp.unsqueeze(0)
                    _, x0_train_tmp = self.AE(z0_train_tmp,mu_tmp)

                    x_net_train = torch.zeros([self.dim_t, x0_train_tmp.shape[1]]).to(dtype=self.dtype_torch, device=self.device_torch)

                    x_net_train[0, :] = x0_train_tmp
                    x0_train_tmp = x0_train_tmp

                    for snapshot in range(self.dim_t - 1):
                        x1_train_tmp = self.net.integrator2(self.net(x0_train_tmp))

                        x_net_train[snapshot + 1, :] = x1_train_tmp

                        x0_train_tmp = x1_train_tmp
                        
                    with torch.no_grad():

                        z_ae_train = self.AE.decode(x_net_train,mu_tmp.squeeze(0).repeat(self.dim_t,1))

                    z_gt_tr_all_i = z_gt_tr_all[i_train*self.dim_t:(i_train+1)*self.dim_t,:]
                    err_res_training[i_train] = self.err_indicator(z_ae_train,z_gt_tr_all_i,self.err_type)#residual err
                    err_max_training[i_train] = self.err_indicator(z_ae_train,z_gt_tr_all_i,1)#max relative err

                # update tolerance of error indicator
                if self.adaptive == 'mean':
                    tol_new = (err_res_training / err_max_training).mean() * self.tol2
                elif self.adaptive == 'last':
                    tol_new = (err_res_training[-1] / err_max_training[-1]).mean() * self.tol2
                else:
                    x = err_max_training.reshape(-1, 1)
                    y = err_res_training.reshape(-1, 1)
                    reg = LinearRegression().fit(x, y)
                    if self.adaptive == 'reg_mean':
                        tol_new = max(0, reg.coef_[0][0] * self.tol2 + reg.intercept_[0])
                        print(reg.coef_[0][0], reg.intercept_[0])

                    elif self.adaptive == 'reg_max':
                        y_diff = y - reg.predict(x)
                        tol_new = max(0, reg.coef_[0][0] * self.tol2 + reg.intercept_[0] + y_diff.max())

                    elif self.adaptive == 'reg_min':
                        y_diff = y - reg.predict(x)
                        tol_new = max(0, reg.coef_[0][0] * self.tol2+ reg.intercept_[0] + y_diff.min())

                self.tol = tol_new

                #return tol_new, err2.max()
                #print(err_max_taining.shape)
                print(f"  Max rel. err.: {err_max_training.max():.1f}%, Update tolerance for error indicator from {tol_old:.5f} to {tol_new:.5f}")

                # Update training dataset and parameter set
                for i_trpara in mu_tr1:
                    if np.linalg.norm(i_trpara.detach().cpu().numpy() - err_max_para_tmp.detach().cpu().numpy()) < 1e-8:
                        print(f"  PARAMETERS EXIST, NOT adding it!")
                        param_flag = False
                        break
                if param_flag:
                    print(f'* Update Training set: add case {err_max_para_tmp}')

                    num_train += 1
                    self.train_indices.append(err_idx)
                    err_max_para.append(err_max_para_tmp)

                    z_tr_add = torch.from_numpy(self.dataset.py_data['data'][err_idx]['x'][:-1, :]).to(dtype=self.dtype_torch, device=self.device_torch)
                    z1_tr_add = torch.from_numpy(self.dataset.py_data['data'][err_idx]['x'][1:, :]).to(dtype=self.dtype_torch, device=self.device_torch)
                    z_tr_all_add = torch.from_numpy(self.dataset.py_data['data'][err_idx]['x']).to(dtype=self.dtype_torch, device=self.device_torch)
                    dz_tr_add = torch.from_numpy(self.dataset.py_data['data'][err_idx]['dx'][:-1, :]).to(dtype=self.dtype_torch, device=self.device_torch)
                    

                    z_gt_tr = torch.cat((z_gt_tr, z_tr_add),0)
                    z1_gt_tr = torch.cat((z1_gt_tr, z1_tr_add),0)
                    z_gt_tr_all = torch.cat((z_gt_tr_all, z_tr_all_add),0)
                    dz_gt_tr = torch.cat((dz_gt_tr, dz_tr_add),0)

                    mu_tr1 = torch.cat((mu_tr1,err_max_para_tmp.unsqueeze(0)),0)

                    mu_tr = torch.repeat_interleave(mu_tr1, self.dim_t - 1, dim=0)

                # Update random subset size
                subset_ratio = self.N_subset / self.num_test * 100  # new subset size

                if err_res_training.max() <= self.tol:
                    w += 1
                    if self.N_subset * 2 <= self.num_test:
                        self.N_subset *= 2  # double the random subset size for evaluation
                    else:
                        self.N_subset= self.num_test
                    subset_ratio = self.N_subset / self.num_test* 100  # new subset size
                    print(f"  Max error indicator <= Tol! Current subset ratio {subset_ratio:.1f}%")

                # check termination criterion
                #if 'sindy_max' in params.keys() and params['sindy_max'] != None:  # prescribed number of local DIs
                if self.n_train_max is not None:
                    if num_train == self.n_train_max + 1:
                        print(f"  Max # SINDys {num_train:d} is reached! Training done!")
                        train_flag = False
                elif subset_ratio >= self.subset_size_max:  # prescribed error toerlance
                    print(  f"  Current subset ratio {subset_ratio:.1f}% >= Target subset ratio {self.subset_size_max:.1f}%!")
                    train_flag = False

            if  i == 0 or (i+i_loaded) % self.print_every == 0 or i == self.epochs:

                print(' ADAM || It: %05d, Loss: %.4e, loss_GFINNs: %.4e, loss_AE_recon: %.4e, loss_AE_jac: %.4e, loss_dx: %.4e, loss_dz: %.4e, validation test: %.4e' %
                    (i+i_loaded, loss.item(), loss_GFINNs.item(), loss_AE.item(), loss_AE_jac.item() , loss_dx.item(), loss_dz.item(), err_max))
                if torch.any(torch.isnan(loss)):
                    self.encounter_nan = True
                    print('Encountering nan, stop training', flush=True)
                    return None
                # if self.save:
                #     if not os.path.exists('model'): os.mkdir('model')
                #     if self.path == None:
                #         torch.save(self.net, 'model/model{}.pkl'.format(i))
                #         torch.save(self.AE, 'model/AE_model{}.pkl'.format(i))
                #     else:
                #         if not os.path.isdir('model/' + self.path): os.makedirs('model/' + self.path)
                #         torch.save(self.net, 'model/{}/model{}.pkl'.format(self.path, i+i_loaded))
                #         torch.save(self.AE, 'model/{}/AE_model{}.pkl'.format(self.path, i+i_loaded))
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_model = self.net
                    best_model_AE = self.AE

                if self.callback is not None:
                    output = self.callback(self.data, self.net)
                    loss_history.append([i+i_loaded, loss.item(), err_max.item(), *output])
                    loss_GFINNs_history.append([i, loss_GFINNs.item(), *output])#, loss_GFINNs_test.item()
                    loss_AE_history.append([i+i_loaded, loss_AE.item(), *output])#, loss_AE_test.item()
                    loss_dx_history.append([i+i_loaded, loss_dx.item(), *output])
                    loss_dz_history.append([i+i_loaded, loss_dz.item(), *output])
                    loss_AE_jac_history.append([i+i_loaded, loss_AE_jac.item(), *output])
                else:
                    loss_history.append([i+i_loaded, loss.item(), err_max.item()])
                    loss_GFINNs_history.append([i, loss_GFINNs.item()]) #, loss_GFINNs_test.item()])
                    loss_AE_history.append([i+i_loaded, loss_AE.item()]) #, loss_AE_test.item()])
                    loss_dx_history.append([i+i_loaded, loss_dx.item()])
                    loss_dz_history.append([i+i_loaded, loss_dz.item()])
                    loss_AE_jac_history.append([i+i_loaded, loss_AE_jac.item()])


                if loss <= Loss_early:
                    print('Stop training: Loss under %.2e' % Loss_early)
                    break
                    
                current_lr = self.__optimizer.param_groups[0]['lr']

                # Check if learning rate is updated
                if current_lr != prev_lr:
                    # Print the updated learning rate
                    print(f"Epoch {i+i_loaded + 1}: Learning rate updated to {current_lr}")
                    # Update the previous learning rate
                    prev_lr = current_lr

        print(f"'number of training data': {num_train}'")
        
        lr_final = self.__optimizer.param_groups[0]['lr']
        lr_AE_final = self.__optimizer.param_groups[1]['lr']
        
        path = './outputs/' + self.path
        if not os.path.isdir(path): os.makedirs(path)
        torch.save({'loss_history':loss_history, 'loss_GFINNs_history':loss_GFINNs_history,'loss_AE_history':loss_AE_history,'loss_AE_jac_history':loss_AE_jac_history,'loss_dx_history':loss_dx_history,'loss_dz_history':loss_dz_history, 'lr_final':lr_final,'lr_AE_final':lr_AE_final}, path + '/loss_history_value.p')

        self.loss_history = np.array(loss_history)
        self.loss_GFINNs_history = np.array(loss_GFINNs_history)
        self.loss_AE_history = np.array(loss_AE_history)
        self.loss_AE_jac_history = np.array(loss_AE_jac_history)
        self.loss_dx_history = np.array(loss_dx_history)
        self.loss_dz_history = np.array(loss_dz_history)

        self.err_array = err_array
        self.err_max_para = err_max_para

        self.best_model = best_model
        self.best_model_AE = best_model_AE

        ##clear some memory
        z_gt_tr = None

        z1_gt_tr = None
        
        z1_gt_tt = None   
        dz_gt_tr = None

        z_gt_tr_all = None
        
        z_gt = None
        
        self.mu_tr1 = mu_tr1

        # print('Done!', flush=True)
        return self.loss_history, self.loss_GFINNs_history, self.loss_AE_history, self.loss_AE_jac_history, self.loss_dx_history, self.loss_dz_history

    def restore(self):
        if self.loss_history is not None and self.save == True:
            best_loss_index = np.argmin(self.loss_history[:, 1])
            iteration = int(self.loss_history[best_loss_index, 0])
            loss_train = self.loss_history[best_loss_index, 1]
            loss_test = self.loss_history[best_loss_index, 2]

            print('BestADAM It: %05d, Loss: %.4e, Test: %.4e' %
                  (iteration, loss_train, loss_test))
        else:
            raise RuntimeError('restore before running or without saved models')
        print('Done!', flush=True)
        return self.best_model, self.best_model_AE

    def output(self, best_model, loss_history, info, **kwargs):
        if self.path is None:
            path = './outputs/' + self.AE_name+'_'+ time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        else:
            path = './outputs/' + self.path
        if not os.path.isdir(path): os.makedirs(path)

        if best_model:
            torch.save(self.best_model, path + '/model_best.pkl')
            torch.save(self.best_model_AE, path + '/model_best_AE.pkl')
            torch.save({'train_indices':self.train_indices,'err_array':self.err_array,'err_max_para':self.err_max_para}, path+'/train_indices.p')
            
        if loss_history:
   
            p1,=plt.plot(self.loss_history[:,0], self.loss_history[:,1],'-')
            p2,=plt.plot(self.loss_GFINNs_history[:,0], self.loss_GFINNs_history[:,1],'-')
            p3,=plt.plot(self.loss_AE_history[:,0], self.loss_AE_history[:,1],'-')
            p4,=plt.plot(self.loss_AE_jac_history[:,0], self.loss_AE_jac_history[:,1],'-')
            p5,=plt.plot(self.loss_dx_history[:,0], self.loss_dx_history[:,1],'-')
            p6,=plt.plot(self.loss_dz_history[:,0], self.loss_dz_history[:,1],'-')
            p7,=plt.plot(self.loss_history[:,0], self.loss_history[:,2],'o')
            plt.legend(['$\mathcal{L}$','$\mathcal{L}_{int}$','$\mathcal{L}_{rec}$','$\mathcal{L}_{jac}$','$\mathcal{L}_{con}$', '$\mathcal{L}_{approx}$','rel. l2 error'], loc='best',ncol=3)  # , '$\hat{u}$'])
            plt.yscale('log')
            plt.ylim(1e-10, 1e1)
            plt.savefig(path + '/loss_all_pred_'+self.AE_name+self.sys_name+'.png')
            p1.remove()
            p2.remove()
            p3.remove()
            p4.remove()
            p5.remove()
            p6.remove()
            p7.remove()

        if info is not None:
            with open(path + '/info.txt', 'w') as f:
                for key, arg in info.items():
                    f.write('{}: {}\n'.format(key, str(arg)))
        for key, arg in kwargs.items():
            np.savetxt(path + '/' + key + '.txt', arg)

    def __init_brain(self):
        self.loss_history = None
        self.encounter_nan = False
        self.best_model = None

        self.net.device = self.device
        self.net.dtype = self.dtype
        self.__init_optimizer()
        self.__init_criterion()

    def __init_optimizer(self):

        if self.optimizer == 'adam':
            params = [
                {'params': self.net.parameters(), 'lr': self.lr, 'weight_decay': self.weight_decay},
                {'params': self.AE.parameters(), 'lr': self.lr_AE, 'weight_decay': self.weight_decay_AE}
            ]

            self.__optimizer = torch.optim.AdamW(params)

            self.__scheduler = torch.optim.lr_scheduler.StepLR(self.__optimizer, step_size=self.miles_lr, gamma=self.gamma_lr)
        else:
            raise NotImplementedError

    def __init_criterion(self):
        if isinstance(self.net, LossNN):
            self.__criterion = self.net.criterion
            if self.criterion is not None:
                import warnings
                warnings.warn('loss-oriented neural network has already implemented its loss function')
        elif self.criterion == 'MSE':
            self.__criterion = torch.nn.MSELoss()
        elif self.criterion == 'CrossEntropy':
            self.__criterion = cross_entropy_loss
        else:
            raise NotImplementedError

    def test(self):
        print("\n[tLaSDI Testing Started]\n")

        self.net = self.best_model
        self.AE = self.best_model_AE

        z_tt_all = torch.from_numpy(np.array([]))
    
        pred_indices = np.arange(self.num_test) 
        for j in pred_indices:
            z_tt_all = torch.cat((z_tt_all, torch.from_numpy(self.dataset.py_data['data'][j]['x'])), 0)
            
        z_tt_all = z_tt_all.to(dtype=self.dtype_torch, device=self.device_torch)

        self.mu_tt_all = self.mu1[pred_indices, :]

        mu_pred_all = torch.repeat_interleave(self.mu_tt_all, self.dim_t, dim=0)

        z0 = z_tt_all[::self.dim_t, :]

        mu0 = mu_pred_all[::self.dim_t, :]

        chunk_size = int(z_tt_all.shape[0]/10)
        z_tt_chunks = torch.chunk(z_tt_all, chunk_size, dim=0)
        mu_chunks = torch.chunk(mu_pred_all, chunk_size, dim=0)

        z_ae_chunks = []
        x_all_chunks = []
        for z_tt_chunk, mu_chunk in zip(z_tt_chunks,mu_chunks):
            with torch.no_grad():
                z_ae_chunk, x_all_chunk = self.AE(z_tt_chunk.detach(),mu_chunk)
                z_ae_chunks.append(z_ae_chunk)
                x_all_chunks.append(x_all_chunk)
        x_all_all = torch.cat(x_all_chunks,dim=0)
            
        _, x0 = self.AE(z0,mu0)
        
        x_net = torch.zeros(x_all_all.shape).to(dtype=self.dtype_torch, device=self.device_torch)

        x_net[::self.dim_t,:] = x0

        dSdt_net = torch.zeros(x_all_all.shape[0]).to(dtype=self.dtype_torch)
        dEdt_net = torch.zeros(x_all_all.shape[0]).to(dtype=self.dtype_torch)
        S_net = torch.zeros(x_all_all.shape[0]).to(dtype=self.dtype_torch)

        dE, M = self.net.netE(x0)

        dS, L = self.net.netS(x0)
        S = self.net.netS.S(x0)

        dE = dE.unsqueeze(1)
        dS = dS.unsqueeze(1)

        dEdt = torch.sum(dE.squeeze()* ((dE @ L) + (dS @ M)).squeeze(),1)
        dSdt = torch.sum(dS.squeeze()* ((dE @ L) + (dS @ M)).squeeze(),1)
        S = S.squeeze()

        dEdt_net[::self.dim_t] = dEdt
        dSdt_net[::self.dim_t] = dSdt
        S_net[::self.dim_t] = S

        for snapshot in range(self.dim_t - 1):

            x1_net = self.net.integrator2(x0.detach())

            x_net[snapshot + 1::self.dim_t, :] = x1_net

            x0 = x1_net
            
            dE, M = self.net.netE(x0)

            dS, L = self.net.netS(x0)

            S = self.net.netS.S(x0)
            S = S.squeeze()

            dE = dE.unsqueeze(1)

            dS = dS.unsqueeze(1)

            dEdt = torch.sum(dE.squeeze() * ((dE @ L) + (dS @ M)).squeeze(), 1)
            dSdt = torch.sum(dS.squeeze() * ((dE @ L) + (dS @ M)).squeeze(), 1)

            dEdt_net[snapshot+1::self.dim_t] = dEdt
            dSdt_net[snapshot+1::self.dim_t] = dSdt
            S_net[snapshot+1::self.dim_t] = S

        x_tlasdi_all = x_net
        
        x_net = None
        x_tlasdi_all = x_tlasdi_all.detach()

        chunk_size = int(x_tlasdi_all.shape[0]/10)
        x_tlasdi_chunks = torch.chunk(x_tlasdi_all, chunk_size, dim=0)
        mu_chunks = torch.chunk(mu_pred_all, chunk_size, dim=0)
        
        z_tlasdi_chunks = []
        for x_tlasdi_chunk, mu_chunk in zip(x_tlasdi_chunks,mu_chunks):
            with torch.no_grad():
                z_tlasdi_chunk = self.AE.decode(x_tlasdi_chunk.detach(),mu_chunk)
                z_tlasdi_chunks.append(z_tlasdi_chunk)
        z_tlasdi_all = torch.cat(z_tlasdi_chunks,dim=0)

        a_grid, w_grid = np.meshgrid(self.amp_test, self.width_test)
        param_list = np.hstack([a_grid.flatten().reshape(-1,1), w_grid.flatten().reshape(-1,1)])
        a_grid, w_grid = np.meshgrid(np.arange(self.amp_test.size), np.arange(self.width_test.size))
        idx_list = np.hstack([a_grid.flatten().reshape(-1,1), w_grid.flatten().reshape(-1,1)])

        idx_param = []
        for i,ip in enumerate(self.mu_tr1.cpu().numpy()):
            idx = np.argmin(np.linalg.norm(param_list-ip, axis=1))
            idx_param.append((idx, np.array([param_list[idx,0], param_list[idx,1]])))

        max_err = np.zeros([len(self.amp_test), len(self.width_test)])
        res_norm = np.zeros([len(self.amp_test), len(self.width_test)])

        count = 0
        for i,a in enumerate(self.amp_test):
            for j,w in enumerate(self.width_test):

                # Max error of all time steps
                max_array_tmp = (np.linalg.norm(z_tt_all[count*self.dim_t:(count+1)*self.dim_t].cpu() - z_tlasdi_all[count*self.dim_t:(count+1)*self.dim_t].cpu(), axis=1) / np.linalg.norm(z_tt_all[count*self.dim_t:(count+1)*self.dim_t].cpu(), axis=1)*100)
                max_array = np.expand_dims(max_array_tmp, axis=0)

                max_err[i,j] = max_array.max()
        
                res_norm[i,j] = self.err_indicator(z_tlasdi_all[count*self.dim_t:(count+1)*self.dim_t].cpu(), z_tt_all[count*self.dim_t:(count+1)*self.dim_t].cpu(), self.err_type)

                count += 1
            
        print('training parameters')
        print(self.mu1[self.train_indices])

        data_path = './outputs/' + self.path
        self.max_err_heatmap(max_err, self.amp_test, self.width_test, data_path, idx_list, idx_param,
                xlabel='Width', ylabel='Amplitude', dtype='float')
        
        self.max_err_heatmap(1000*res_norm, self.amp_test, self.width_test, data_path, idx_list, idx_param,
                xlabel='Width', ylabel='Amplitude',label = 'Residual Norm', dtype='float')

        path = './outputs/' + self.path
        torch.save({'z_tlasdi_all':z_tlasdi_all, 'z_tt_all':z_tt_all}, path + '/u_tLaSDI_u_GT.p')
        torch.save({'dSdt_net':dSdt_net, 'dEdt_net':dEdt_net, 'S_net':S_net, 'train_indices':self.train_indices,'mu1':self.mu1}, path + '/Entropy_Energy_tLaSDI.p')
        
        print("\n[tLaSDI Testing Finished]\n")

    def err_indicator(self,z,data, err_type):
        """
        This function computes errors using a speciffied error indicator.
        inputs:
            data: dict, data of the evalution case
            err_type: int, types of error indicator
                    1: max relative error (if test data is available)
                    2: residual norm (mean), 1D Burger's eqn
                    3: residual norm (mean), 1D heat eqn
        outputs:
            err: float, error
        """

        z = z.detach().cpu().numpy()
        data = data.detach().cpu().numpy()
        if err_type == 1:
            err = (np.linalg.norm(data - z, axis=1) / np.linalg.norm(data, axis=1) * 100).max()
            #err = (torch.linalg.norm(data - z, axis=1) / torch.linalg.norm(data, axis=1) * 100).max()
        elif err_type == 2:
            res = []
            for k in range(z.shape[0] - 1):
                res.append(self.residual_1Dburger(z[k, :], z[k + 1, :]))
            err = np.stack(res).mean()
        elif err_type == 3:
            res = []
            for k in range(z.shape[0] - 1):
                res.append(self.residual_1DHeat(z[k, :], z[k + 1, :]))
            err = np.stack(res).mean()

        return err

    def residual_1Dburger(self, u0, u1):

        nx = self.dim_z
        dx = 6 / (nx - 1)
        dt = self.dt
        c = dt / dx

        idxn1 = np.zeros(nx, dtype='int')
        idxn1[1:] = np.arange(nx - 1)
        idxn1[0] = nx - 1

        f = c * (u1 ** 2 - u1 * u1[idxn1])
        r = -u0 + u1 + f

        return np.linalg.norm(r)
    
    def residual_1DHeat(self, u0, u1):

        nx = self.dim_z
        dx = 6 / (nx - 1)
        dt = self.dt
        
        c = dt / (dx**2)

        idxn1 = np.zeros(nx, dtype='int')
        idxn1[1:] = np.arange(nx - 1)
        idxn1[0] = nx - 1
        
        idxw1 = np.zeros(nx,dtype='int')
        idxw1[1:]  = idxn1[1:]+2
        idxw1[0] = 1
        idxw1[-1] = 0

        f = c*(u1[idxw1] - 2*u1 + u1[idxn1])
        r = -u0 + u1 - f

        return np.linalg.norm(r)
    
    def max_err_heatmap(self, max_err, p1_test, p2_test, data_path, idx_list=[], idx_param=[],
                    xlabel='param1', ylabel='param2', label='Max. Relative Error (%)', dtype='int', scale=1):
        sns.set(font_scale=1.3)
        if dtype == 'int':
            max_err = max_err.astype(int)
            fmt1 = 'd'
        else:
            fmt1 = '.1f'
        rect = []
        for i in range(len(idx_param)):
            print(f"idx: {idx_param[i][0]}, param: {idx_param[i][1]}")
            idd = idx_param[i][0]
            rect.append(
                patches.Rectangle((idx_list[idd, 1], idx_list[idd, 0]), 1, 1, linewidth=2, edgecolor='k', facecolor='none'))
        rect2 = deepcopy(rect)

        if max_err.size < 100:
            fig = plt.figure(figsize=(5, 5))
        else:
            fig = plt.figure(figsize=(9, 9))

        fontsize = 14

        ax = fig.add_subplot(111)
        cbar_ax = fig.add_axes([0.99, 0.19, 0.02, 0.7])


        vmax = max_err.max() * scale
        heatmap = sns.heatmap(max_err * scale, ax=ax, square=True,
                    xticklabels=p2_test, yticklabels=p1_test,
                    annot=True, annot_kws={'size': fontsize}, fmt=fmt1,
                    cbar_ax=cbar_ax, cbar=True, cmap='vlag', robust=True, vmin=0, vmax=5.9)

        # Define a formatter function to add the percentage sign
        def percentage_formatter(x, pos):
            return "{:.0f}%".format(x)

        # Apply the formatter to the colorbar
        cbar = heatmap.collections[0].colorbar
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))

        for i in rect2:
            ax.add_patch(i)

        # format text labels
        fmt = '{:0.2f}'
        xticklabels = []
        for item in ax.get_xticklabels():
            item.set_text(fmt.format(float(item.get_text())))
            xticklabels += [item]
        yticklabels = []
        for item in ax.get_yticklabels():
            item.set_text(fmt.format(float(item.get_text())))
            yticklabels += [item]
        ax.set_xticklabels(xticklabels)
        ax.set_yticklabels(yticklabels)
        ax.set_xlabel('Width ($\\omega$)', fontsize=24)
        ax.set_ylabel('Amplitude ($\\alpha$)', fontsize=24)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)

        plt.tight_layout()
        if label == 'Residual Norm':
            plt.savefig(data_path + f'heatmap_resNorm.png', bbox_inches='tight')
        else:
            plt.savefig(data_path + f'heatmap_maxRelErr_tlasdi.png', bbox_inches='tight')
        plt.show()

