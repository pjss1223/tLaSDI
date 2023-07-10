"""
@author: jpzxshi & zen
"""
import os
import time
import numpy as np
import torch

from data2 import Data
from .nn import LossNN
from .utils import timing, cross_entropy_loss
from sklearn.linear_model import LinearRegression

#from utilities.plot_gfinns import plot_results #, plot_latent

import torch
import torch.optim as optim
import numpy as np
import copy
from scipy import sparse as sp

from model_AEhyper import SparseAutoEncoder #, StackedSparseAutoEncoder
from dataset_sim_hyper import load_dataset, split_dataset
from utilities.plot import plot_results, plot_latent_visco, plot_latent_tire, plot_latent
from utilities.utils import print_mse, all_latent
import matplotlib.pyplot as plt


class Brain_tLaSDI_GAEhyper:
    '''Runner based on torch.
    '''
    brain = None

    @classmethod
    def Init(cls,  net, dt, sys_name, output_dir, save_plots, criterion, optimizer, lr,
             iterations, lbfgs_steps, AE_name,dset_dir,output_dir_AE,save_plots_AE,layer_vec_SAE,layer_vec_SAE_q,layer_vec_SAE_v,layer_vec_SAE_sigma,
             activation_SAE,depth_hyper, width_hyper, act_hyper, num_sensor,lr_SAE,lambda_r_SAE,lambda_jac_SAE,lambda_dx,lambda_dz,miles_lr = [10000],gamma_lr = 1e-1, path=None, load_path = None, batch_size=None,
             batch_size_test=None, weight_decay=0, print_every=1000, save=False, load=False, callback=None, dtype='float',
             device='cpu',tol = 1e-3, tol2 = 2, adaptive = 'reg_max',n_train_max = 30,subset_size_max=80,trunc_period =1):
        cls.brain = cls( net, dt, sys_name, output_dir, save_plots, criterion,
                         optimizer, lr, weight_decay, iterations, lbfgs_steps,AE_name,dset_dir,output_dir_AE,save_plots_AE,layer_vec_SAE,
                         layer_vec_SAE_q,layer_vec_SAE_v,layer_vec_SAE_sigma,activation_SAE,depth_hyper, width_hyper, act_hyper, num_sensor,lr_SAE,lambda_r_SAE,lambda_jac_SAE,lambda_dx,lambda_dz,miles_lr,gamma_lr, path,load_path, batch_size,
                         batch_size_test, print_every, save, load, callback, dtype, device, tol, tol2,adaptive,n_train_max,subset_size_max,trunc_period)

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

    def __init__(self,  net, dt,sys_name, output_dir,save_plots, criterion, optimizer, lr, weight_decay, iterations, lbfgs_steps,AE_name,dset_dir,output_dir_AE,save_plots_AE,layer_vec_SAE,layer_vec_SAE_q,layer_vec_SAE_v,layer_vec_SAE_sigma,
             activation_SAE,depth_hyper, width_hyper, act_hyper, num_sensor,lr_SAE,lambda_r_SAE,lambda_jac_SAE,lambda_dx,lambda_dz,miles_lr,gamma_lr, path, load_path, batch_size,
                 batch_size_test, print_every, save, load, callback, dtype, device, tol, tol2, adaptive,n_train_max,subset_size_max,trunc_period):
        #self.data = data
        self.net = net
        #print(self.net.netE.fnnB.modus['LinMout'].weight)
        self.sys_name = sys_name
        self.output_dir = output_dir
        self.save_plots = save_plots
        #self.x_trunc = x_trunc
#        self.latent_idx = latent_idx
        self.dt = dt
        #self.z_gt = z_gt
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.iterations = iterations
        self.lbfgs_steps = lbfgs_steps
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
        self.AE_name = AE_name
        self.n_train_max = n_train_max
        self.subset_size_max = subset_size_max
        
        self.miles_lr = miles_lr
        self.gamma_lr = gamma_lr
        


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
            self.SAE = torch.load( path + '/model_best_AE.pkl')
            self.net = torch.load( path + '/model_best.pkl')
        else:

            if self.sys_name == '1DBurgers':
                #self.SAE = SparseAutoEncoder(layer_vec_SAE, activation_SAE).float()
                if self.dtype == 'float':
                    self.SAE = SparseAutoEncoder(layer_vec_SAE, activation_SAE,depth_hyper, width_hyper, act_hyper, num_sensor).float()
                elif self.dtype == 'double':
                    self.SAE = SparseAutoEncoder(layer_vec_SAE, activation_SAE,depth_hyper, width_hyper, act_hyper, num_sensor).double()
                    
                if self.device =='gpu':
                    self.SAE = self.SAE.to(torch.device('cuda'))

#             elif self.sys_name == 'rolling_tire':
#                 #self.SAE = StackedSparseAutoEncoder(layer_vec_SAE_q, layer_vec_SAE_v, layer_vec_SAE_sigma,
#                 #                                    activation_SAE).float()
#                 self.SAE = StackedSparseAutoEncoder(layer_vec_SAE_q, layer_vec_SAE_v, layer_vec_SAE_sigma,
#                                                     activation_SAE).double()
#                 if self.device =='gpu':
#                     self.SAE = self.SAE.to(torch.device('cuda'))

        print(sum(p.numel() for p in self.SAE .parameters() if p.requires_grad))
        print(sum(p.numel() for p in self.net.parameters() if p.requires_grad))

        # ALL parameters --------------------------------------------------------------------------------

        if self.sys_name == '1DBurgers':

            self.num_test = 64
            self.num_train = 4 # initial num_train
            self.err_type = 2  # residual of 1DBurgers

            amp_train = np.linspace(0.7, 0.9, 2)
            width_train = np.linspace(0.9, 1.1, 2)
            amp_test = np.linspace(0.7, 0.9, 8)
            width_test = np.linspace(0.9, 1.1, 8)



        elif self.sys_name == '2DBurgers':
            # self.num_test = 64
            # self.num_train = 4
            # self.err_type = 2
            #
            # amp_train = np.linspace(0.7, 0.9, 2)
            # width_train = np.linspace(0.9, 1.1, 2)
            # amp_test = np.linspace(0.7, 0.9, 8)
            # width_test = np.linspace(0.9, 1.1, 8)
            self.err_type = 3




        grid1, grid2 = np.meshgrid(amp_train, width_train)
        train_param = np.hstack((grid1.flatten().reshape(-1, 1), grid2.flatten().reshape(-1, 1)))
        grid1, grid2 = np.meshgrid(amp_test, width_test)
        test_param = np.hstack((grid1.flatten().reshape(-1, 1), grid2.flatten().reshape(-1, 1)))

        train_indices = []
        for i in range(self.num_test):
            for j in range(self.num_train):
                if np.abs(test_param[i, 0] - train_param[j, 0]) < 1e-8 and \
                        np.abs(test_param[i, 1] - train_param[j, 1]) < 1e-8:
                    train_indices.append(i)
        #print(train_indices)


        self.train_indices = train_indices
        self.test_indices = np.arange(self.num_test)

        self.dset_dir = dset_dir

        # Dataset Parameters
        self.dataset = load_dataset(self.sys_name, self.dset_dir,self.device,self.dtype)
        self.dt = self.dataset.dt
        self.dim_t = self.dataset.dim_t
        self.dim_z = self.dataset.dim_z
        self.mu1 = self.dataset.mu
        self.dim_mu = self.dataset.dim_mu

        self.mu_tr1 = self.mu1[self.train_indices,:]
        self.mu_tt1 = self.mu1[self.test_indices, :]

        # self.mu = np.repeat(self.mu1, self.dim_t, axis=0)
        # self.mu_tr = np.repeat(self.mu_tr1, self.dim_t-1, axis=0)
        # self.mu_tt = np.repeat(self.mu_tt1, self.dim_t-1, axis=0)

        #self.mu = self.mu1.repeat(self.dim_t,1)
        self.mu = torch.repeat_interleave(self.mu1, self.dim_t, dim=0)
        #self.mu_tr = self.mu_tr1.repeat(self.dim_t-1,1)
        self.mu_tr = torch.repeat_interleave(self.mu_tr1,self.dim_t-1,dim=0)
        #print(self.mu_tr)

        #self.mu_tt = self.mu_tt1.repeat(self.dim_t-1,1)
        self.mu_tt = torch.repeat_interleave(self.mu_tt1, self.dim_t - 1, dim=0)


#         self.z = torch.from_numpy(np.array([]))
#         self.z_tr = torch.from_numpy(np.array([]))
#         self.z1_tr = torch.from_numpy(np.array([]))
#         self.z_tt = torch.from_numpy(np.array([]))
#         self.z1_tt = torch.from_numpy(np.array([]))
#         self.z_tt_all = torch.from_numpy(np.array([]))
#         self.z_tr_all = torch.from_numpy(np.array([]))
#         self.dz_tt = torch.from_numpy(np.array([]))
#         self.dz_tr = torch.from_numpy(np.array([]))

#         for j in range(self.mu1.shape[0]):
#             self.z = torch.cat((self.z,torch.from_numpy(self.dataset.py_data['data'][j]['x'])),0)

#         for j in self.train_indices:
#             self.z_tr = torch.cat((self.z_tr,torch.from_numpy(self.dataset.py_data['data'][j]['x'][:-1,:])),0)
#             self.z1_tr = torch.cat((self.z1_tr, torch.from_numpy(self.dataset.py_data['data'][j]['x'][1:,:])), 0)
#             self.z_tr_all = torch.cat((self.z_tr_all, torch.from_numpy(self.dataset.py_data['data'][j]['x'])), 0)
#             self.dz_tr = torch.cat((self.dz_tr, torch.from_numpy(self.dataset.py_data['data'][j]['dx'][:-1, :])), 0)

#         for j in self.test_indices:
#             self.z_tt = torch.cat((self.z_tt,torch.from_numpy(self.dataset.py_data['data'][j]['x'][:-1:,:])),0)
#             self.z1_tt = torch.cat((self.z1_tt, torch.from_numpy(self.dataset.py_data['data'][j]['x'][1:,:])), 0)
#             self.z_tt_all = torch.cat((self.z_tt_all, torch.from_numpy(self.dataset.py_data['data'][j]['x'])), 0)
#             self.dz_tt = torch.cat((self.dz_tt, torch.from_numpy(self.dataset.py_data['data'][j]['dx'][:-1, :])), 0)
            #self.z_tt_all = self.z

        path = './data/'
#         torch.save({'z':self.z,'z_tr':self.z_tr,'z_tt':self.z_tt,'z1_tr':self.z1_tr ,'z1_tt':self.z1_tt,'z_tt_all':self.z_tt_all,'z_tr_all':self.z_tr_all},path + '/Z_data.p')
        
        z_data = torch.load(path + '/1DBG_Z_data.p')

        self.z = z_data['z']
        self.z_tr = z_data['z_tr']
        self.z_tt = z_data['z_tt']
        self.z1_tr = z_data['z1_tr']
        self.z1_tt = z_data['z1_tt']
        self.z_tt_all = z_data['z_tt_all']
        self.z_tr_all = z_data['z_tr_all']
        self.dz_tt = z_data['dz_tt']
        self.dz_tr = z_data['dz_tr']

        if self.dtype == 'float':
            self.z = self.z.to(torch.float32)
            self.z_tr = self.z_tr.to(torch.float32)
            self.z_tt = self.z_tt.to(torch.float32)
            self.z1_tr = self.z1_tr.to(torch.float32)
            self.z1_tt = self.z1_tt.to(torch.float32)
            self.z_tt_all = self.z_tt_all.to(torch.float32)
            self.z_tr_all = self.z_tr_all.to(torch.float32)
            self.dz_tt = self.dz_tt.to(torch.float32)
            self.dz_tr = self.dz_tr.to(torch.float32)



        if self.device == 'gpu':
            self.z = self.z.to(torch.device("cuda"))
            self.z_tr = self.z_tr.to(torch.device("cuda"))
            self.z_tt= self.z_tt.to(torch.device("cuda"))
            self.z1_tr = self.z1_tr.to(torch.device("cuda"))
            self.z1_tt = self.z1_tt.to(torch.device("cuda"))
            self.z_tr_all = self.z_tr_all.to(torch.device("cuda"))
            self.z_tt_all = self.z_tt_all.to(torch.device("cuda"))
            self.dz_tr = self.dz_tr.to(torch.device("cuda"))
            self.dz_tt = self.dz_tt.to(torch.device("cuda"))


        self.z_gt = self.z

        self.lambda_r = lambda_r_SAE
        self.lambda_jac = lambda_jac_SAE
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
        loss_history = []
        loss_GFINNs_history = []
        loss_AE_history = []
        loss_dx_history = []
        loss_dz_history = []
        loss_AE_jac_history = []
        testing_losses = []
        err_array = []
        err_max_para = []
        num_train = self.num_train


        #initial training, testing data (normalized)

        z_gt_tr = self.z_tr
        z_gt_tt = self.z_tt

        z1_gt_tr = self.z1_tr
        z1_gt_tt = self.z1_tt

        dz_gt_tr = self.dz_tr
        dz_gt_tt = self.dz_tt

        z_gt_tr_all = self.z_tr_all

        z_gt_norm = self.SAE.normalize(self.z)


        mu_tr1 = self.mu_tr1
        mu_tt1 = self.mu_tt1

        mu_tr = self.mu_tr
        mu_tt = self.mu_tt
        mu = self.mu




        w = 1
        prev_lr = self.__optimizer.param_groups[0]['lr']
        for i in range(self.iterations + 1):


            #
            z_gt_tr_norm = self.SAE.normalize(z_gt_tr)
            z_gt_tt_norm = self.SAE.normalize(z_gt_tt)
            #

            dz_gt_tr_norm = self.SAE.normalize(dz_gt_tr)
            dz_gt_tt_norm = self.SAE.normalize(dz_gt_tt)

            z1_gt_tr_norm = self.SAE.normalize(z1_gt_tr)
            z1_gt_tt_norm = self.SAE.normalize(z1_gt_tt)
            #
            z_gt_tr_all_norm = self.SAE.normalize(z_gt_tr_all)
            #

            # z_gt_tr_norm_mu = torch.cat((z_gt_tr_norm, mu_tr), axis=1)
            # z_gt_tt_norm_mu = torch.cat((z_gt_tt_norm, mu_tt), axis=1)
            #
            # z1_gt_tr_norm_mu = torch.cat((z1_gt_tr_norm, mu_tr), axis=1)
            # z1_gt_tt_norm_mu = torch.cat((z1_gt_tt_norm, mu_tt), axis=1)

            #
            z_sae_tr_norm, x = self.SAE(z_gt_tr_norm, mu_tr)
            z_sae_tt_norm, x_tt = self.SAE(z_gt_tt_norm, mu_tt)

            z1_sae_tr_norm, x1 = self.SAE(z1_gt_tr_norm,mu_tr)
            z1_sae_tt_norm, x1_tt = self.SAE(z1_gt_tt_norm,mu_tt)




            x_mu_tr, x1_mu_tr = torch.cat((x,mu_tr),axis=1),  torch.cat((x1,mu_tr),axis=1)
            x_mu_tt, x1_mu_tt = torch.cat((x_tt,mu_tt),axis=1),  torch.cat((x1_tt,mu_tt),axis=1)

            self.data = Data(x_mu_tr, x1_mu_tr, x_mu_tt, x1_mu_tt)

            self.data.device = self.device
            self.data.dtype = self.dtype

            # data = Data(x_mu_tr, x1_mu_tr, x_mu_tt, x1_mu_tt)
            # self.data = data
            #
            # data.device = self.device
            # data.dtype = self.dtype

            # X_mu_train, y_mu_train = data.get_batch(self.batch_size)

            X_train = x
            #mu_train = X_mu_train[:,-self.dim_mu:]

            y_train = x1

            # integrator loss
            
            
            loss_GFINNs = self.__criterion(self.net(X_train), y_train)

            # reconstruction loss
            loss_AE = torch.mean((z_sae_tr_norm - z_gt_tr_norm) ** 2)
            
            
            if  ((self.lambda_jac == 0 and self.lambda_dx == 0) and self.lambda_dz == 0): 
                loss_AE_jac = torch.tensor(0)
                loss_dx = torch.tensor(0)
                loss_dz = torch.tensor(0)
                
            else:


                if self.device == 'cpu':
                    loss_AE_jac, J_e, J_d, idx_trunc = self.SAE.jacobian_norm_trunc(z_gt_tr_norm, x, mu_tr, self.trunc_period)
                else:
                    loss_AE_jac, J_e, J_d, idx_trunc = self.SAE.jacobian_norm_trunc_gpu(z_gt_tr_norm, x, mu_tr, self.trunc_period)

                dx_train = self.net.f(X_train)

                dz_gt_tr_norm = dz_gt_tr_norm.unsqueeze(2)


                dx_data_train = J_e @ dz_gt_tr_norm[:, idx_trunc]
                dx_data_train = dx_data_train.squeeze()

                dz_gt_tr_norm = dz_gt_tr_norm.squeeze()

                dx_train = dx_train.unsqueeze(2)
                dz_train = J_d @ dx_train

                dx_train = dx_train.squeeze()
                dz_train = dz_train.squeeze()

                dz_gt_tr_norm = dz_gt_tr_norm.squeeze()

                # consistency loss
                loss_dx = torch.mean((dx_train - dx_data_train) ** 2)

                # model approximation loss
                loss_dz = torch.mean((dz_train - dz_gt_tr_norm[:, idx_trunc]) ** 2)

            loss = loss_GFINNs+self.lambda_r*loss_AE+ self.lambda_dx*loss_dx +self.lambda_dz*loss_dz+self.lambda_jac*loss_AE_jac



            #print(loss) #tensor(0.0008, grad_fn=<MseLossBackward0>)
            Loss_early = 1e-10

            self.N_subset = int(0.5 * self.num_test)

            param_flag = True

            if i % self.print_every == 0 or i == self.iterations:

                # select a random subset for evaluation
                rng = np.random.default_rng()
                a = np.setdiff1d(np.arange(self.num_test), self.train_indices)  # exclude existing training cases
                rng.shuffle(a)
                subset = a[:self.N_subset]


                err_array_tmp = np.zeros([self.num_test, 1])
                for i_test in np.arange(self.num_test):
                    if i_test in subset:
                        z_subset = torch.from_numpy(self.dataset.py_data['data'][i_test]['x'])
                        z0_subset = z_subset[0,:].unsqueeze(0)
                        
                        if self.dtype == 'float':
                            z_subset = z_subset.to(torch.float32)
                            z0_subset = z0_subset.to(torch.float32)


                        if self.device == 'gpu':
                            z_subset = z_subset.to(torch.device("cuda"))
                            z0_subset = z0_subset.to(torch.device("cuda"))

                        mu0 = self.mu1[i_test, :].unsqueeze(0)

                        z0_subset_norm = self.SAE.normalize(z0_subset)
                        #print(z0_subset_norm.shape)

                        _,x0_subset = self.SAE(z0_subset_norm,mu0)


                        
                        if self.dtype == 'double':
                            
                            x_net_subset = torch.zeros(self.dim_t, x0_subset.shape[1]).double()
                            
                        elif self.dtype == 'float':
                            x_net_subset = torch.zeros(self.dim_t, x0_subset.shape[1]).float()
                            

                        # print(x0_subset.shape)
                        # print(x_net_subset[0,:].shape)
                        x_net_subset[0,:] = x0_subset

                        if self.device == 'gpu':
                            x_net_subset = x_net_subset.to(torch.device('cuda'))

                        x0_subset = x0_subset
                        #mu0 = mu0.unsqueeze(0)
                        for snapshot in range(self.dim_t - 1):
                            #print(x0_subset.shape) #[1, 10]
                            #print(mu0.shape) # [1,2]
                            x1_net = self.net.integrator2(self.net(x0_subset))

                            x_net_subset[snapshot + 1, :] = x1_net

                            x0_subset = x1_net


                        z_sae_subset = self.SAE.decode(x_net_subset,mu0.squeeze(0).repeat(self.dim_t,1))
                        #print(z_sae_subset-self.SAE.denormalize(z_sae_subset))
                        z_sae_subset = self.SAE.denormalize(z_sae_subset)
                        #print(z_sae_subset.shape) # 101 101
                        #print(z_subset.shape) # 101 101
                        #print(z_subset.shape)
                        err_array_tmp[i_test] = self.err_indicator(z_sae_subset,z_subset,self.err_type)

                    else:
                        err_array_tmp[i_test] =-1

                #maximum residual errors
                #print(err_array_tmp)
                err_max = err_array_tmp.max() # maximum relative error measured in 'subset'
                err_idx = np.argmax(err_array_tmp)
                err_max_para_tmp = self.mu1[err_idx, :]

                #1142710507

                testing_losses.append(err_max)
                err_array.append(err_array_tmp)

                err_max_para.append(err_max_para_tmp)


                #update tolerance

                tol_old = self.tol

                err_rel_training = np.zeros(num_train)  # residual norm
                err_max_training = np.zeros(num_train)  # max relative error

                for i_train in range(num_train):

                    z0_train_tmp = z_gt_tr_all_norm[i_train*(self.dim_t),:]
                    mu_tmp = mu_tr1[i_train].unsqueeze(0)
                    z0_train_tmp = z0_train_tmp.unsqueeze(0)
                    _, x0_train_tmp = self.SAE(z0_train_tmp,mu_tmp)


                    
                    if self.dtype == 'double':

                        x_net_train = torch.zeros([self.dim_t, x0_train_tmp.shape[1]]).double()
                    elif self.dtype == 'float':
                        x_net_train = torch.zeros([self.dim_t, x0_train_tmp.shape[1]]).float()

                    if self.device == 'gpu':
                        x_net_train = x_net_train.to(torch.device('cuda'))



                    x_net_train[0, :] = x0_train_tmp
                    x0_train_tmp = x0_train_tmp
                    #mu_tmp = mu_tmp.unsqueeze(0)

                    for snapshot in range(self.dim_t - 1):
                        x1_train_tmp = self.net.integrator2(self.net(x0_train_tmp))

                        x_net_train[snapshot + 1, :] = x1_train_tmp

                        x0_train_tmp = x1_train_tmp

                    z_sae_train = self.SAE.decode(x_net_train,mu_tmp.squeeze(0).repeat(self.dim_t,1))
                    z_sae_train = self.SAE.denormalize(z_sae_train)

                    z_gt_tr_all_i = z_gt_tr_all[i_train*self.dim_t:(i_train+1)*self.dim_t,:]
                    # print(z_sae_train.shape)
                    # print(z_gt_tr_all_i.shape)
                    err_rel_training[i_train] = self.err_indicator(z_sae_train,z_gt_tr_all_i,self.err_type)
                    err_max_training[i_train] = self.err_indicator(z_sae_train,z_gt_tr_all_i,1)

                # update tolerance of error indicator
                if self.adaptive == 'mean':
                    tol_new = (err_rel_training / err_max_training).mean() * self.tol2
                elif self.adaptive == 'last':
                    tol_new = (err_rel_training[-1] / err_max_training[-1]).mean() * self.tol2
                else:
                    x = err_max_training.reshape(-1, 1)
                    y = err_rel_training.reshape(-1, 1)
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
                #print(f"  Max rel. err.: {err_max_taining:.1f}%, Update tolerance for error indicator from {tol_old:.5f} to {tol_new:.5f}")

                # Update training dataset and parameter set
                for i_trpara in mu_tr1:
                    if np.linalg.norm(i_trpara.detach().cpu().numpy() - err_max_para_tmp.detach().cpu().numpy()) < 1e-8:
                        print(f"  PARAMETERS EXIST, NOT adding it!")
                        param_flag = False
                        break
                if param_flag:
                    print(f'* Update Training set: add case {err_max_para_tmp}')
                    #training_data['data'].append(test_data['data'][idx])

                    num_train += 1
                    #params['param'] = training_data['param']
                    self.train_indices.append(err_idx)
                    err_max_para.append(err_max_para_tmp)

                    z_tr_add = torch.from_numpy(self.dataset.py_data['data'][err_idx]['x'][:-1, :])
                    z1_tr_add = torch.from_numpy(self.dataset.py_data['data'][err_idx]['x'][1:, :])
                    z_tr_all_add = torch.from_numpy(self.dataset.py_data['data'][err_idx]['x'])
                    dz_tr_add = torch.from_numpy(self.dataset.py_data['data'][err_idx]['dx'][:-1, :])
                    
                    if self.dtype == 'float':
                        z_tr_add = z_tr_add.to(torch.float32)
                        z1_tr_add = z1_tr_add.to(torch.float32)
                        z_tr_all_add = z_tr_all_add.to(torch.float32)
                        dz_tr_add = dz_tr_add.to(torch.float32)

                    if self.device == 'gpu':
                        z_tr_add = z_tr_add.to(torch.device("cuda"))
                        z1_tr_add = z1_tr_add.to(torch.device("cuda"))
                        z_tr_all_add = z_tr_all_add.to(torch.device("cuda"))
                        dz_tr_add = dz_tr_add.to(torch.device("cuda"))

                    z_gt_tr = torch.cat((z_gt_tr, z_tr_add),0)
                    z1_gt_tr = torch.cat((z1_gt_tr, z1_tr_add),0)
                    z_gt_tr_all = torch.cat((z_gt_tr_all, z_tr_all_add),0)
                    dz_gt_tr = torch.cat((dz_gt_tr, dz_tr_add),0)

                    #mu_tr1.append(err_max_para_tmp)
                    #print(err_max_para_tmp.shape)#[2]
                    mu_tr1 = torch.cat((mu_tr1,err_max_para_tmp.unsqueeze(0)),0)
                    #print(mu_tr1.shape)

                    # mu_tr = mu_tr1.repeat(self.dim_t - 1, 1)
                    mu_tr = torch.repeat_interleave(mu_tr1, self.dim_t - 1, dim=0)
                    #print(mu_tr.shape)


                # Update random subset size
                subset_ratio = self.N_subset / self.num_test * 100  # new subset size
                #print(err_rel_taining.shape)
                if err_rel_training.max() <= self.tol:
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
                #
                # X_mu_test, y_mu_test = data.get_batch_test(self.batch_size_test)
                #
                # X_test = X_mu_test[:, :-self.dim_mu]
                # mu_test = X_mu_test[:, -self.dim_mu:]
                # y_test = y_mu_test[:, :-self.dim_mu]
                # #print('test', X_test.shape) # [30,4]
                # #X_test1 = self.net.integrator2(self.net(X_test[:-1]))
                # #z_sae_gfinns_tt_norm = self.SAE.decode(X_test1)
                #
                # loss_GFINNs_test = self.__criterion(self.net(X_test),mu_test, y_test)
                # loss_AE_test = torch.mean((z_sae_tt_norm - z_gt_tt_norm) ** 2)
                # #loss_AE_GFINNs_test = torch.mean((z_sae_gfinns_tt_norm - z_gt_tt_norm1) ** 2)
                #
                # loss_test = loss_GFINNs_test+self.lambda_r*loss_AE_test#+loss_AE_GFINNs_test

                # print(i)
                # print(loss_GFINNs)
                # print(loss_AE)
                # print(err_max)
                print(' ADAM || It: %05d, Loss: %.4e, loss_GFINNs: %.4e, loss_AE_recon: %.4e, loss_dx: %.4e, loss_dz: %.4e, loss_jac: %.4e, validation test: %.4e' %
                    (i, loss.item(), loss_GFINNs.item(), loss_AE.item(), loss_dx.item(), loss_dz.item(),loss_AE_jac.item(), err_max))
                if torch.any(torch.isnan(loss)):
                    self.encounter_nan = True
                    print('Encountering nan, stop training', flush=True)
                    return None
                if self.save:
                    if not os.path.exists('model'): os.mkdir('model')
                    if self.path == None:
                        torch.save(self.net, 'model/model{}.pkl'.format(i))
                        torch.save(self.SAE, 'model/AE_model{}.pkl'.format(i))
                    else:
                        if not os.path.isdir('model/' + self.path): os.makedirs('model/' + self.path)
                        torch.save(self.net, 'model/{}/model{}.pkl'.format(self.path, i))
                        torch.save(self.SAE, 'model/{}/AE_model{}.pkl'.format(self.path, i))
                if self.callback is not None:
                    output = self.callback(self.data, self.net)
                    loss_history.append([i, loss.item(), err_max, *output])
                    loss_GFINNs_history.append([i, loss_GFINNs.item(), *output])#, loss_GFINNs_test.item()
                    loss_AE_history.append([i, loss_AE.item(), *output])#, loss_AE_test.item()
                    loss_dx_history.append([i, loss_dx.item(), *output])
                    loss_dz_history.append([i, loss_dz.item(), *output])
                    loss_AE_jac_history.append([i, loss_AE_jac.item(), *output])
             #       loss_AE_GFINNs_history.append([i, loss_AE_GFINNs.item(), loss_AE_GFINNs_test.item(), *output])
                else:
                    loss_history.append([i, loss.item(), err_max])
                    loss_history.append([i, loss.item(), err_max])
                    loss_GFINNs_history.append([i, loss_GFINNs.item()]) #, loss_GFINNs_test.item()])
                    loss_AE_history.append([i, loss_AE.item()]) #, loss_AE_test.item()])
                    loss_dx_history.append([i, loss_dx.item()])
                    loss_dz_history.append([i, loss_dz.item()])
                    loss_AE_jac_history.append([i, loss_AE_jac.item()])


                if loss <= Loss_early:
                    print('Stop training: Loss under %.2e' % Loss_early)
                    break
                    
                current_lr = self.__optimizer.param_groups[0]['lr']

                # Check if learning rate is updated
                if current_lr != prev_lr:
                    # Print the updated learning rate
                    print(f"Epoch {i + 1}: Learning rate updated to {current_lr}")

                    # Update the previous learning rate
                    prev_lr = current_lr
                    
            if i < self.iterations:
                self.__optimizer.zero_grad()
                #print(loss)
                loss.backward(retain_graph=True)
                #loss.backward()
                self.__optimizer.step()
                self.__scheduler.step()
                
        self.loss_history = np.array(loss_history)
        self.loss_GFINNs_history = np.array(loss_GFINNs_history)
        self.loss_AE_history = np.array(loss_AE_history)
        self.loss_dx_history = np.array(loss_dx_history)
        self.loss_dz_history = np.array(loss_dz_history)
        self.loss_AE_jac_history = np.array(loss_AE_jac_history)

        _, x_de = self.SAE(z_gt_norm,mu)

        plot_param_index = 0
        pid = plot_param_index
        if self.sys_name == 'viscoelastic':
            # Plot latent variables
            if (self.save_plots == True):
                plot_name = '[VC] AE Latent Variables_'+self.AE_name
                plot_latent_visco(x_de, self.dataset.dt, plot_name, self.output_dir)

        elif self.sys_name == '1DBurgers':

            # Plot latent variables
            if (self.save_plots == True):
                plot_name = '[1DBurgers] AE Latent Variables_'+self.AE_name
                plot_latent_visco(x_de[pid*self.dim_t:(pid+1)*self.dim_t], self.dataset.dt, plot_name, self.output_dir)

        elif self.sys_name == 'rolling_tire':
            x_q, x_v, x_sigma = self.SAE.split_latent(x_de)

            # Plot latent variables
            if (self.save_plots == True):
                plot_name = '[Rolling Tire] AE Latent Variables_'+self.AE_name
                plot_latent_tire(x_q, x_v, x_sigma, self.dataset.dt, plot_name, self.output_dir)



        # print('Done!', flush=True)
        return self.loss_history, self.loss_GFINNs_history, self.loss_AE_history, self.loss_dx_history, self.loss_dz_history, self.loss_AE_jac_history



    def restore(self):
        if self.loss_history is not None and self.save == True:
            best_loss_index = np.argmin(self.loss_history[:, 1])
            iteration = int(self.loss_history[best_loss_index, 0])
            loss_train = self.loss_history[best_loss_index, 1]
            loss_test = self.loss_history[best_loss_index, 2]
            # print('Best model at iteration {}:'.format(iteration), flush=True)
            # print('Train loss:', loss_train, 'Test loss:', loss_test, flush=True)
            print('BestADAM It: %05d, Loss: %.4e, Test: %.4e' %
                  (iteration, loss_train, loss_test))
            if self.path == None:
                self.best_model = torch.load('model/model{}.pkl'.format(iteration))
                self.best_model_AE = torch.load('model/AE_model{}.pkl'.format(iteration))
                #self.best_model = torch.load('model/model10000.pkl')
            else:
                self.best_model = torch.load('model/{}/model{}.pkl'.format(self.path, iteration))
                self.best_model_AE = torch.load('model/{}/AE_model{}.pkl'.format(self.path, iteration))
        else:
            raise RuntimeError('restore before running or without saved models')
        from torch.optim import LBFGS
        optim = LBFGS(self.best_model.parameters(), history_size=100,
                      max_iter=self.lbfgs_steps,
                      tolerance_grad=1e-09, tolerance_change=1e-09,
                      line_search_fn="strong_wolfe")
        self.it = 0
        if self.lbfgs_steps != 0:
            def closure():
                if torch.is_grad_enabled():
                    optim.zero_grad()
                X_mu_train, y_mu_train = self.data.get_batch(None)

                X_mu_test, y_mu_test = self.data.get_batch_test(None)

                X_train = X_mu_train[:, :-self.dim_mu]
                X_test = X_mu_test[:, :-self.dim_mu]
                # mu_train = X_mu_train[:, -self.dim_mu:]
                # mu_test = X_mu_test[:, -self.dim_mu:]


                y_train = y_mu_train[:, :-self.dim_mu]
                y_test = y_mu_test[:, :-self.dim_mu]


                loss = self.best_model.criterion(self.best_model(X_train), y_train)
                loss_test = self.best_model.criterion(self.best_model(X_test), y_test)
                # print('Train loss: {:<25}Test loss: {:<25}'.format(loss.item(), loss_test.item()), flush=True)
                it = self.it + 1
                if it % self.print_every == 0 or it == self.lbfgs_steps:
                    print('L-BFGS|| It: %05d, Loss: %.4e, Test: %.4e' %
                          (it, loss.item(), loss_test.item()))
                self.it = it
                if loss.requires_grad:
                    loss.backward(retain_graph=True)
                return loss

            optim.step(closure)
        print('Done!', flush=True)
        return self.best_model



    def output(self, best_model, loss_history, info, **kwargs):
        if self.path is None:
            path = './outputs/' + self.AE_name+'_'+ time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        else:
            path = './outputs/' + self.path
        if not os.path.isdir(path): os.makedirs(path)

        if best_model:
            torch.save(self.best_model, path + '/model_best.pkl')
            torch.save(self.best_model_AE, path + '/model_best_AE.pkl')
        if loss_history:
            np.savetxt(path + '/loss.txt', self.loss_history)
            p1,=plt.plot(self.loss_history[:,0], self.loss_history[:,1],'-')
            p2,= plt.plot(self.loss_history[:,0], self.loss_history[:,2],'--')
            plt.legend(['train loss', 'test loss'])  # , '$\hat{u}$'])
            plt.yscale('log')
            plt.savefig(path + '/loss_'+self.AE_name+'.png')
            p1.remove()
            p2.remove()

            # p3,=plt.plot(self.loss_GFINNs_history[:,0], self.loss_GFINNs_history[:,1],'-')
            # p4,=plt.plot(self.loss_GFINNs_history[:,0], self.loss_GFINNs_history[:,2],'--')
            np.savetxt(path + '/loss_GFINNs.txt', self.loss_GFINNs_history)
            p3,=plt.plot(self.loss_GFINNs_history[:,0], self.loss_GFINNs_history[:,1],'-')
            #p4,=plt.plot(self.loss_GFINNs_history[:,0], self.loss_GFINNs_history[:,2],'--')
            plt.legend(['train loss (GFINNs)', 'test loss (GFINNs)'])  # , '$\hat{u}$'])
            plt.yscale('log')
            plt.savefig(path + '/loss_GFINNs_'+self.AE_name+'.png')
            p3.remove()
            #p4.remove()

            np.savetxt(path + '/loss_AE.txt', self.loss_AE_history)
            p5,=plt.plot(self.loss_AE_history[:,0], self.loss_AE_history[:,1],'-')
            #p6,=plt.plot(self.loss_AE_history[:,0], self.loss_AE_history[:,2],'--')
            plt.legend(['train loss (AE)', 'test loss (AE)'])  # , '$\hat{u}$'])
            plt.yscale('log')
            plt.savefig(path + '/loss_AE_'+self.AE_name+'.png')
            p5.remove()
            #p6.remove()

            np.savetxt(path + '/loss_dx.txt', self.loss_dx_history)
            p7,=plt.plot(self.loss_dx_history[:,0], self.loss_dx_history[:,1],'-')
            #p10,=plt.plot(self.loss_AE_jac_history[:,0], self.loss_AE_jac_history[:,2],'--')
            plt.legend(['train loss (dx)', 'test loss (dx)'])  # , '$\hat{u}$'])
            plt.yscale('log')
            plt.savefig(path + '/loss_dx_'+self.AE_name+'.png')
            p7.remove()
            #p10.remove()

            np.savetxt(path + '/loss_dz.txt', self.loss_dz_history)
            p8,=plt.plot(self.loss_dz_history[:,0], self.loss_dz_history[:,1],'-')
            #p10,=plt.plot(self.loss_AE_jac_history[:,0], self.loss_AE_jac_history[:,2],'--')
            plt.legend(['train loss (dz)', 'test loss (dz)'])  # , '$\hat{u}$'])
            plt.yscale('log')
            plt.savefig(path + '/loss_dz_'+self.AE_name+'.png')
            p8.remove()

            np.savetxt(path + '/loss_jac.txt', self.loss_AE_jac_history)
            p9,=plt.plot(self.loss_AE_jac_history[:,0], self.loss_AE_jac_history[:,1],'-')
            #p10,=plt.plot(self.loss_AE_jac_history[:,0], self.loss_AE_jac_history[:,2],'--')
            plt.legend(['train loss (Jac)', 'test loss (Jac)'])  # , '$\hat{u}$'])
            plt.yscale('log')
            plt.savefig(path + '/loss_jac_'+self.AE_name+'.png')
            p9.remove()

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
        # self.data.device = self.device
        # self.data.dtype = self.dtype
        self.net.device = self.device
        self.net.dtype = self.dtype
        self.__init_optimizer()
        self.__init_criterion()

    def __init_optimizer(self):
        if self.optimizer == 'adam':
            self.__optimizer = torch.optim.Adam(list(self.net.parameters())+list(self.SAE.parameters()), lr=self.lr, weight_decay=self.weight_decay)                                   
            self.__scheduler = torch.optim.lr_scheduler.MultiStepLR(self.__optimizer, milestones=self.miles_lr,gamma=self.gamma_lr)
                   
                                                
        else:
            raise NotImplementedError

    def __init_criterion(self):
        #print(self.net.criterion)
        #print(isinstance(self.net, LossNN_hyper))
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

    ##from spnn
    def test(self):
        print("\n[GFNN Testing Started]\n")

        #self.dim_t = self.z_gt.shape[0]
        self.net = self.best_model
        self.SAE = self.best_model_AE

        z_gt_norm = self.SAE.normalize(self.z_gt)
        z_tt_norm = self.SAE.normalize(self.z_tt_all)

        z0 = z_tt_norm[::self.dim_t, :]

        mu0 = self.mu_tt[::self.dim_t, :]



        # Forward pass
        with torch.no_grad():
            z_sae_norm, x_all = self.SAE(z_tt_norm, self.mu)
            z_sae = self.SAE.denormalize(z_sae_norm)

        #z_norm = self.SAE.normalize(z)

        _, x0 = self.SAE(z0,mu0)






                                                
        if self.dtype == 'double':
            x_net = torch.zeros(x_all.shape).double()

            x_net_all = torch.zeros(x_all.shape).double()
        elif self.dtype == 'float':
            x_net = torch.zeros(x_all.shape).float()

            x_net_all = torch.zeros(x_all.shape).float()



        x_net[::self.dim_t,:] = x0

        # x_net_all[::self.dim_t,:] = x0
        #
        #
        # x_net_all[1:,:] = self.net.integrator2(self.net(x_all[:-1,:]))

        if self.device == 'gpu':
          x_net = x_net.to(torch.device('cuda'))
          #x_net_all = x_net_all.to(torch.device('cuda'))


        
        if self.dtype == 'double':
            dSdt_net = torch.zeros(x_all.shape[0]).double()
            dEdt_net = torch.zeros(x_all.shape[0]).double()
        elif self.dtype == 'float':
            dSdt_net = torch.zeros(x_all.shape[0]).float()
            dEdt_net = torch.zeros(x_all.shape[0]).float()

        dE, M = self.net.netE(x0)


          #     print(dE.shape)
          # print(M.shape)
        dS, L = self.net.netS(x0)


        dE = dE.unsqueeze(1)
        #print(dE.shape)
        dS = dS.unsqueeze(1)



        dEdt = torch.sum(dE.squeeze()* ((dE @ L) + (dS @ M)).squeeze(),1)
        dSdt = torch.sum(dS.squeeze()* ((dE @ L) + (dS @ M)).squeeze(),1)



        dEdt_net[::self.dim_t] = dEdt
        dSdt_net[::self.dim_t] = dSdt



        for snapshot in range(self.dim_t - 1):
            # Structure-Preserving Neural Network
            # print(snapshot)
            # print(x_net.shape)

            x1_net = self.net.integrator2(self.net(x0))
            #x1_net = self.net.criterion(self.net(x), self.dt)

            #dEdt, dSdt = self.SPNN.get_thermodynamics(x)

            # Save results and Time update
            x_net[snapshot + 1::self.dim_t, :] = x1_net
            # dEdt_net[snapshot] = dEdt
            # dSdt_net[snapshot] = dSdt
            x0 = x1_net

            dE, M = self.net.netE(x0)
            #     print(dE.shape)
            # print(M.shape)
            dS, L = self.net.netS(x0)

            dE = dE.unsqueeze(1)

            dS = dS.unsqueeze(1)


            dEdt = torch.sum(dE.squeeze() * ((dE @ L) + (dS @ M)).squeeze(), 1)
            dSdt = torch.sum(dS.squeeze() * ((dE @ L) + (dS @ M)).squeeze(), 1)
            #print(dSdt.shape)#256


            dEdt_net[snapshot+1::self.dim_t] = dEdt
            dSdt_net[snapshot+1::self.dim_t] = dSdt



        # Detruncate
        # x_gfinn = torch.zeros([self.dim_t, self.SAE.dim_latent])
        # x_gfinn[:, latent_idx] = x_net

        x_gfinn = x_net



        # Decode latent vector
        z_gfinn_norm = self.SAE.decode(x_gfinn,self.mu)
        z_gfinn = self.SAE.denormalize(z_gfinn_norm)



        # z_gfinn_all_norm = self.SAE.decode(x_net_all)
        # z_gfinn_all = self.SAE.denormalize(z_gfinn_all_norm)

        # Load Ground Truth and Compute MSE
        z_gt = self.z_gt
        z_tt_all = self.z_tt_all
        # print_mse(z_gfinn, z_gt, self.sys_name)
        print_mse(z_gfinn, z_tt_all, self.sys_name)
        # print_mse(z_gfinn_all, z_gt, self.sys_name)
        # print_mse(z_sae, z_gt, self.sys_name)

        # print(z_gfinn.shape)
        # print(z_gt.shape)


        # Plot results
        pid = 0 #index for test para
        if (self.save_plots):
            #plot_name = 'SPNN Full Integration (Latent)'
            #print((pid+1)*self.dim_t)
            plot_name = 'Energy_Entropy_Derivatives_' +self.AE_name
            plot_latent(dEdt_net[pid*self.dim_t:(pid+1)*self.dim_t], dSdt_net[pid*self.dim_t:(pid+1)*self.dim_t], self.dt, plot_name, self.output_dir, self.sys_name)
            plot_name = 'GFINNs Full Integration_'+self.AE_name
            #print(self.sys_name)
            plot_results(z_gfinn[pid*self.dim_t:(pid+1)*self.dim_t,:], z_tt_all[pid*self.dim_t:(pid+1)*self.dim_t,:], self.dt, plot_name, self.output_dir, self.sys_name)

            plot_name = 'AE Reduction Only_'+self.AE_name
            plot_results(z_sae[pid*self.dim_t:(pid+1)*self.dim_t,:], z_tt_all[pid*self.dim_t:(pid+1)*self.dim_t,:], self.dt, plot_name, self.output_dir, self.sys_name)

            if self.sys_name == 'viscoelastic':
                # Plot latent variables
                if (self.save_plots == True):
                    plot_name = '[VC] Latent Variables_' + self.AE_name
                    plot_latent_visco(x_gfinn, self.dataset.dt, plot_name, self.output_dir)

            elif self.sys_name == '1DBurgers':

                # Plot latent variables
                if (self.save_plots == True):
                    plot_name = '[1DBurgers] Latent Variables_' + self.AE_name
                    plot_latent_visco(x_gfinn[pid*self.dim_t:(pid+1)*self.dim_t], self.dataset.dt, plot_name, self.output_dir)

            elif self.sys_name == 'rolling_tire':
                x_q, x_v, x_sigma = self.SAE.split_latent(x_gfinn)

                # Plot latent variables
                if (self.save_plots == True):
                    plot_name = '[RT] Latent Variables_' + self.AE_name
                    plot_latent_tire(x_q, x_v, x_sigma, self.dataset.dt, plot_name, self.output_dir)

        print("\n[GFINNs Testing Finished]\n")

    def err_indicator(self,z,data, err_type):
        """
        This function computes errors using a speciffied error indicator.
        inputs:
            data: dict, data of the evalution case
            err_type: int, types of error indicator
                    1: max relative error (if test data is available)
                    2: residual norm (mean), 1D Burger's eqn
                    3: residual norm (mean), 2D Burger's eqn
                    4: MFEM example 16: Time dependent heat conduction
                    5: MFEM example 9: DG advection
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
            #err = torch.stack(res).mean()
            err = np.stack(res).mean()
        elif err_type == 3:
            res = []
            for k in range(z.shape[0] - 1):
                res.append(self.residual_2Dburger(z[k, :], z[k + 1, :]))
            #err = torch.stack(res).mean()
            err = np.stack(res).mean()

        return err

    def residual_1Dburger(self, u0, u1):
        """
        r = -u^{n} + u^{n+1} -dt*f(u^{n+1})
        """
        # nx = params['pde']['nx']
        # nt = params['pde']['nt']
        nx = self.dim_z
        #tstop = params['pde']['tstop']
        dx = 6 / (nx - 1)
        #dt = tstop / nt
        dt = self.dt
        #print(dt)
        c = dt / dx

        idxn1 = np.zeros(nx, dtype='int')
        idxn1[1:] = np.arange(nx - 1)
        idxn1[0] = nx - 1

        f = c * (u1 ** 2 - u1 * u1[idxn1])
        r = -u0 + u1 + f
        #print(r.shape)
        return np.linalg.norm(r)
        #return torch.linalg.norm(r)

    def residual_2Dburger(x_prev, x, params):
        Re = params['pde']['Re']
        nx = params['pde']['nx']
        ny = nx
        nt = params['pde']['nt']
        tstop = params['pde']['tstop']
        ic = params['pde']['ic']  # initial condition, 1: Sine, 2: Gaussian
        u_prev = x_prev[:nx * ny]
        u = x[:nx * ny]
        v_prev = x_prev[nx * ny:]
        v = x[nx * ny:]

        dt = tstop / nt
        t = np.linspace(0, tstop, nt + 1)
        nxy = (nx - 2) * (ny - 2)
        dx = 1 / (nx - 1)
        dy = 1 / (ny - 1)

        if ic == 1:  # sine
            xmin = 0
            xmax = 1
            ymin = 0
            ymax = 1
        elif ic == 2:  # Gaussian
            xmin = -3
            xmax = 3
            ymin = -3
            ymax = 3
            x0 = 0  # Gaussian center
            y0 = 0  # Gaussian center
        else:
            print('wrong values for IC!')
        I = sp.eye(nxy, format='csr')

        # full indices, free indices, fixed indices
        [xv, yv] = np.meshgrid(np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny), indexing='xy')
        x = xv.flatten()
        y = yv.flatten()

        multi_index_i, multi_index_j = np.meshgrid(np.arange(nx), np.arange(ny), indexing='xy')
        full_multi_index = (multi_index_j.flatten(), multi_index_i.flatten())
        free_multi_index = (multi_index_j[1:-1, 1:-1].flatten(), multi_index_i[1:-1, 1:-1].flatten())
        x0_multi_index = (multi_index_j[1:-1, 0].flatten(), multi_index_i[1:-1, 0].flatten())
        x1_multi_index = (multi_index_j[1:-1, -1].flatten(), multi_index_i[1:-1, -1].flatten())
        y0_multi_index = (multi_index_j[0, 1:-1].flatten(), multi_index_i[0, 1:-1].flatten())
        y1_multi_index = (multi_index_j[-1, 1:-1].flatten(), multi_index_i[-1, 1:-1].flatten())

        dims = (ny, nx)
        full_raveled_indices = np.ravel_multi_index(full_multi_index, dims)
        free_raveled_indices = np.ravel_multi_index(free_multi_index, dims)
        x0_raveled_indices = np.ravel_multi_index(x0_multi_index, dims)
        x1_raveled_indices = np.ravel_multi_index(x1_multi_index, dims)
        x01_raveled_indices = np.concatenate((x0_raveled_indices, x1_raveled_indices))
        y0_raveled_indices = np.ravel_multi_index(y0_multi_index, dims)
        y1_raveled_indices = np.ravel_multi_index(y1_multi_index, dims)
        y01_raveled_indices = np.concatenate((y0_raveled_indices, y1_raveled_indices))
        fixed_raveled_indices = np.setdiff1d(full_raveled_indices, free_raveled_indices)

        # boundary one-hot vector
        x0_one_hot = np.eye(nx - 2)[0]
        y0_one_hot = np.eye(ny - 2)[0]
        x1_one_hot = np.eye(nx - 2)[-1]
        y1_one_hot = np.eye(ny - 2)[-1]

        # inner grid
        inner_multi_index_i, inner_multi_index_j = np.meshgrid(np.arange(nx - 2), np.arange(ny - 2), indexing='xy')
        inner_x_multi_index = (
        np.concatenate((inner_multi_index_j[:, 0].flatten(), inner_multi_index_j[:, -1].flatten())),
        np.concatenate((inner_multi_index_i[:, 0].flatten(), inner_multi_index_i[:, -1].flatten())))
        inner_y_multi_index = (
        np.concatenate((inner_multi_index_j[0, :].flatten(), inner_multi_index_j[-1, :].flatten())),
        np.concatenate((inner_multi_index_i[0, :].flatten(), inner_multi_index_i[-1, :].flatten())))

        inner_dims = (ny - 2, nx - 2)
        inner_x_raveled_indices = np.ravel_multi_index(inner_x_multi_index, inner_dims)
        inner_y_raveled_indices = np.ravel_multi_index(inner_y_multi_index, inner_dims)

        # first order derivative
        # central
        Mcb = sp.diags([np.zeros(nx - 2), -np.ones(nx - 2), np.ones(nx - 2)], [0, -1, 1], (nx - 2, nx - 2))
        Mc = sp.kron(sp.eye(ny - 2), Mcb, format="csr")

        Ib = sp.eye(nx - 2)
        Nc = sp.kron(sp.diags([np.zeros(ny - 2), -np.ones(ny - 2), np.ones(ny - 2)], [0, -1, 1], (ny - 2, ny - 2)), Ib,
                     format="csr")

        # forward
        Mfb = sp.diags([-np.ones(nx - 2), np.ones(nx - 2)], [0, 1], (nx - 2, nx - 2))
        Mf = sp.kron(sp.eye(ny - 2), Mfb, format="csr")

        Ib = sp.eye(nx - 2)
        Nf = sp.kron(sp.diags([-np.ones(ny - 2), np.ones(ny - 2)], [0, 1], (ny - 2, ny - 2)), Ib, format="csr")

        # backward
        Mbb = sp.diags([np.ones(nx - 2), -np.ones(nx - 2)], [0, -1], (nx - 2, nx - 2))
        Mb = sp.kron(sp.eye(ny - 2), Mbb, format="csr")

        Ib = sp.eye(nx - 2)
        Nb = sp.kron(sp.diags([np.ones(ny - 2), -np.ones(ny - 2)], [0, -1], (ny - 2, ny - 2)), Ib, format="csr")

        # laplacian operator
        Dxb = sp.diags([-2 * np.ones(nx - 2), np.ones(nx - 2), np.ones(nx - 2)], [0, -1, 1], (nx - 2, nx - 2))
        Dx = sp.kron(sp.eye(ny - 2), Dxb, format="csr")

        Ib = sp.eye(nx - 2)
        Dy = sp.kron(sp.diags([-2 * np.ones(ny - 2), np.ones(ny - 2), np.ones(ny - 2)], [0, -1, 1], (ny - 2, ny - 2)),
                     Ib,
                     format="csr")

        # Initial condition
        amp = params['pde']['param'][0]
        width = params['pde']['param'][1]
        if ic == 1:  # IC: sine
            zv = amp * np.sin(2 * np.pi * xv) * np.sin(2 * np.pi * yv)
            zv[np.nonzero(xv > 0.5)] = 0.0
            zv[np.nonzero(yv > 0.5)] = 0.0
        elif ic == 2:  # IC: Gaussian
            zv = amp * np.exp(-((xv - x0) ** 2 + (yv - y0) ** 2) / width)
            z = zv.flatten()
        u0 = z.copy()
        v0 = z.copy()

        # boundary for first order derivative term
        Bdudx0_cur = np.kron(u0[x0_raveled_indices], x0_one_hot)
        Bdudy0_cur = np.kron(y0_one_hot, u0[y0_raveled_indices])
        Bdvdx0_cur = np.kron(v0[x0_raveled_indices], x0_one_hot)
        Bdvdy0_cur = np.kron(y0_one_hot, v0[y0_raveled_indices])
        Bdudx1_cur = np.kron(u0[x1_raveled_indices], x1_one_hot)
        Bdudy1_cur = np.kron(y1_one_hot, u0[y1_raveled_indices])
        Bdvdx1_cur = np.kron(v0[x1_raveled_indices], x1_one_hot)
        Bdvdy1_cur = np.kron(y1_one_hot, v0[y1_raveled_indices])

        # boundary for second order derivative term
        bxu_cur = np.zeros(nxy)
        byu_cur = np.zeros(nxy)
        bxv_cur = np.zeros(nxy)
        byv_cur = np.zeros(nxy)

        bxu_cur[inner_x_raveled_indices] = u0[x01_raveled_indices]
        byu_cur[inner_y_raveled_indices] = u0[y01_raveled_indices]
        bxv_cur[inner_x_raveled_indices] = v0[x01_raveled_indices]
        byv_cur[inner_y_raveled_indices] = v0[y01_raveled_indices]

        u_free_prev = np.copy(u_prev[free_raveled_indices])
        v_free_prev = np.copy(v_prev[free_raveled_indices])

        u_free = np.copy(u[free_raveled_indices])
        v_free = np.copy(v[free_raveled_indices])

        Mu_free = Mb.dot(u_free)
        Mv_free = Mb.dot(v_free)
        Nu_free = Nb.dot(u_free)
        Nv_free = Nb.dot(v_free)

        f_u = (-1 / dx * (u_free * (Mu_free - Bdudx0_cur))
               - 1 / dy * (v_free * (Nu_free - Bdudy0_cur))
               + 1 / (Re * dx ** 2) * (Dx.dot(u_free) + bxu_cur)
               + 1 / (Re * dy ** 2) * (Dy.dot(u_free) + byu_cur))

        f_v = (-1 / dx * (u_free * (Mv_free - Bdvdx0_cur))
               - 1 / dy * (v_free * (Nv_free - Bdvdy0_cur))
               + 1 / (Re * dx ** 2) * (Dx.dot(v_free) + bxv_cur)
               + 1 / (Re * dy ** 2) * (Dy.dot(v_free) + byv_cur))

        r_u = u_free - u_free_prev - dt * f_u
        r_v = v_free - v_free_prev - dt * f_v

        return np.linalg.norm(r_u) + np.linalg.norm(r_v)
