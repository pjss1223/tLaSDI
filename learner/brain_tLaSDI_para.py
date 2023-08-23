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

from model import SparseAutoEncoder #, StackedSparseAutoEncoder
from dataset_sim import load_dataset, split_dataset
from utilities.plot import plot_results, plot_latent_visco, plot_latent_tire, plot_latent
from utilities.utils import print_mse, all_latent
import matplotlib.pyplot as plt

### ------------------------- no parameter involved in AE, GFINNs
class Brain_tLaSDI_para:
    '''Runner based on torch.
    '''
    brain = None

    @classmethod
    def Init(cls,  net, dt, sys_name, output_dir, save_plots, criterion, optimizer, lr,
             epochs, lbfgs_steps, AE_name,dset_dir,output_dir_AE,save_plots_AE,layer_vec_SAE,layer_vec_SAE_q,layer_vec_SAE_v,layer_vec_SAE_sigma,
             activation_SAE,depth_hyper, width_hyper, act_hyper, num_sensor,lr_SAE,lambda_r_SAE,lambda_jac_SAE,lambda_dx,lambda_dz,miles_lr = [10000],gamma_lr = 1e-1, path=None, load_path = None, batch_size=None,
             batch_size_test=None, weight_decay=0,update_epochs=1000, print_every=1000, save=False, load=False, callback=None, dtype='float',
             device='cpu',tol = 1e-3, tol2 = 2, adaptive = 'reg_max',n_train_max = 30,subset_size_max=80,trunc_period =1):
        cls.brain = cls( net, dt, sys_name, output_dir, save_plots, criterion,
                         optimizer, lr, weight_decay, epochs, lbfgs_steps,AE_name,dset_dir,output_dir_AE,save_plots_AE,layer_vec_SAE,
                         layer_vec_SAE_q,layer_vec_SAE_v,layer_vec_SAE_sigma,activation_SAE,depth_hyper, width_hyper, act_hyper, num_sensor,lr_SAE,lambda_r_SAE,lambda_jac_SAE,lambda_dx,lambda_dz,miles_lr,gamma_lr, path,load_path, batch_size,
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

    def __init__(self,  net, dt,sys_name, output_dir,save_plots, criterion, optimizer, lr, weight_decay, epochs, lbfgs_steps,AE_name,dset_dir,output_dir_AE,save_plots_AE,layer_vec_SAE,layer_vec_SAE_q,layer_vec_SAE_v,layer_vec_SAE_sigma,
             activation_SAE,depth_hyper, width_hyper, act_hyper, num_sensor,lr_SAE,lambda_r_SAE,lambda_jac_SAE,lambda_dx,lambda_dz,miles_lr,gamma_lr, path, load_path, batch_size,
                 batch_size_test, update_epochs, print_every, save, load, callback, dtype, device, tol, tol2, adaptive,n_train_max,subset_size_max,trunc_period):
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
        self.epochs = epochs
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
        self.update_epochs = update_epochs
        
        self.miles_lr = miles_lr
        self.gamma_lr = gamma_lr

        
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
            if self.device == 'gpu':
                self.SAE = self.SAE.to(torch.device('cuda'))
                self.net = self.net.to(torch.device('cuda'))
            else:
                self.SAE = self.SAE.to(torch.device('cpu'))
                self.net = self.net.to(torch.device('cpu'))
        else:

            if self.sys_name == '1DBurgers':
                #self.SAE = SparseAutoEncoder(layer_vec_SAE, activation_SAE).float()
                if self.dtype == 'float':
                    self.SAE = SparseAutoEncoder(layer_vec_SAE, activation_SAE).float()
                elif self.dtype == 'double':
                    self.SAE = SparseAutoEncoder(layer_vec_SAE, activation_SAE).double()
                    
                if self.device =='gpu':
                    self.SAE = self.SAE.to(torch.device('cuda'))


        print(sum(p.numel() for p in self.SAE .parameters() if p.requires_grad))
        print(sum(p.numel() for p in self.net.parameters() if p.requires_grad))

        # ALL parameters --------------------------------------------------------------------------------

        if self.sys_name == '1DBurgers':
            self.num_para = 64
            self.num_test = 32
            self.num_train = 32 # num_train

#             amp_train = np.linspace(0.7, 0.9, 2)
#             width_train = np.linspace(0.9, 1.1, 2)
#             amp_test = np.linspace(0.7, 0.9, 8)
#             width_test = np.linspace(0.9, 1.1, 8)



       # elif self.sys_name == '2DBurgers':
            # self.num_test = 64
            # self.num_train = 4
            # self.err_type = 2
            #
            # amp_train = np.linspace(0.7, 0.9, 2)
            # width_train = np.linspace(0.9, 1.1, 2)
            # amp_test = np.linspace(0.7, 0.9, 8)
            # width_test = np.linspace(0.9, 1.1, 8)
            #self.err_type = 3




        
        
        self.dset_dir = dset_dir

        # Dataset Parameters
        self.dataset = load_dataset(self.sys_name, self.dset_dir,self.device,self.dtype)
        self.dt = self.dataset.dt
        self.dim_t = self.dataset.dim_t
        self.dim_z = self.dataset.dim_z
        self.mu1 = self.dataset.mu
#         self.dim_mu = self.dataset.dim_mu

#         self.mu_tr1 = self.mu1[self.train_indices,:]
#         self.mu_tt1 = self.mu1[self.test_indices, :]

#         self.mu = torch.repeat_interleave(self.mu1, self.dim_t, dim=0)
#         self.mu_tr = torch.repeat_interleave(self.mu_tr1,self.dim_t-1,dim=0)

#         self.mu_tt = torch.repeat_interleave(self.mu_tt1, self.dim_t - 1, dim=0)
        
        self.train_indices, self.test_indices = split_dataset(self.sys_name, self.num_para)


        self.z = torch.from_numpy(np.array([]))
        self.dz = torch.from_numpy(np.array([]))
        self.z_tr = torch.from_numpy(np.array([]))
        self.z1_tr = torch.from_numpy(np.array([]))
        self.z_tt = torch.from_numpy(np.array([]))
        self.z1_tt = torch.from_numpy(np.array([]))
        self.z_tt_all = torch.from_numpy(np.array([]))
        self.z_tr_all = torch.from_numpy(np.array([]))
        self.dz_tt = torch.from_numpy(np.array([]))
        self.dz_tr = torch.from_numpy(np.array([]))

        for j in range(self.mu1.shape[0]):
            self.z = torch.cat((self.z,torch.from_numpy(self.dataset.py_data['data'][j]['x'])),0)
            self.dz = torch.cat((self.dz,torch.from_numpy(self.dataset.py_data['data'][j]['dx'])),0)

#         print(self.num_para)
#         print(len(self.train_indices))
#         print(len(self.test_indices))
        
        for j in self.train_indices:
            self.z_tr = torch.cat((self.z_tr,torch.from_numpy(self.dataset.py_data['data'][j]['x'][:-1,:])),0)
            self.z1_tr = torch.cat((self.z1_tr, torch.from_numpy(self.dataset.py_data['data'][j]['x'][1:,:])), 0)
            self.z_tr_all = torch.cat((self.z_tr_all, torch.from_numpy(self.dataset.py_data['data'][j]['x'])), 0)
            self.dz_tr = torch.cat((self.dz_tr, torch.from_numpy(self.dataset.py_data['data'][j]['dx'][:-1, :])), 0)

        for j in self.test_indices:
            self.z_tt = torch.cat((self.z_tt,torch.from_numpy(self.dataset.py_data['data'][j]['x'][:-1:,:])),0)
            self.z1_tt = torch.cat((self.z1_tt, torch.from_numpy(self.dataset.py_data['data'][j]['x'][1:,:])), 0)
            self.z_tt_all = torch.cat((self.z_tt_all, torch.from_numpy(self.dataset.py_data['data'][j]['x'])), 0)
            self.dz_tt = torch.cat((self.dz_tt, torch.from_numpy(self.dataset.py_data['data'][j]['dx'][:-1, :])), 0)
            self.z_tt_all = self.z

        path = './data/'
        torch.save({'z':self.z,'dz':self.dz,'z_tr':self.z_tr,'z_tt':self.z_tt,'z1_tr':self.z1_tr,'dz_tr':self.dz_tr,'dz_tt':self.dz_tt   ,'z1_tt':self.z1_tt,'z_tt_all':self.z_tt_all,'z_tr_all':self.z_tr_all},path + '/1DBG_Z_data_para_400_300.p')
        
        z_data = torch.load(path + '/1DBG_Z_data_para_400_300.p')

        
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
#         err_array = []
#         err_max_para = []
        num_train = self.num_train
        


        #initial training, testing data (normalized)

        z_gt_tr = self.z_tr
        self.z_tr = None
        
        z_gt_tt = self.z_tt
        self.z_tt = None
        

        z1_gt_tr = self.z1_tr
        self.z1_tr = None
        
        z1_gt_tt = self.z1_tt
        self.z1_tt = None

        
        dz_gt_tr = self.dz_tr
        self.dz_tr = None
        
        dz_gt_tt = self.dz_tt
        self.dz_tt = None

        z_gt_tr_all = self.z_tr_all
        self.z_tr_all = None
        

        z_gt = self.z_gt
        

#         mu_tr1 = self.mu_tr1
#         mu_tt1 = self.mu_tt1

#         mu_tr = self.mu_tr
#         mu_tt = self.mu_tt
#         mu = self.mu


        self.batch_num = (self.dim_t-1) // self.batch_size
        Loss_early = 1e-10

        w = 1
        prev_lr = self.__optimizer.param_groups[0]['lr']
        for i in range(self.epochs + 1):
            
            for batch in range(self.batch_num):
                start_idx = batch * self.batch_size
                end_idx = (batch + 1) * self.batch_size
                if batch == self.batch_num-1:
                    end_idx = self.dim_t-1
                
                row_indices_batch = torch.cat([torch.arange(idx_r+start_idx, idx_r + end_idx) for idx_r in range(0, z_gt_tr.size(0), self.dim_t-1)])


            #
                z_gt_tr_batch = z_gt_tr[row_indices_batch,:]

                z1_gt_tr_batch = z1_gt_tr[row_indices_batch,:]


                dz_gt_tr_batch = dz_gt_tr[row_indices_batch,:]
            
            #
                z_sae_tr, X_train = self.SAE(z_gt_tr_batch)

                z1_sae_tr, y_train = self.SAE(z1_gt_tr_batch)

                
#                 X_mu_train, y_mu_train = torch.cat((X_train,mu_tr_batch),axis=1),  torch.cat((y_train,mu_tr_batch),axis=1)
#                 x_mu_tt, x1_mu_tt = torch.cat((x_tt,mu_tt),axis=1),  torch.cat((x1_tt,mu_tt),axis=1)


#                 self.data = Data(X_train, y_train, x_tt, x1_tt)

#                 self.data.device = self.device
#                 self.data.dtype = self.dtype
                
#                 mu_train = mu_tr_batch

            
                loss_GFINNs = self.__criterion(self.net(X_train), y_train)

                # reconstruction loss
                loss_AE = torch.mean((z_sae_tr - z_gt_tr_batch) ** 2)
                
            
            
                if  ((self.lambda_jac == 0 and self.lambda_dx == 0) and self.lambda_dz == 0): 
                    loss_AE_jac = torch.tensor(0, dtype=torch.float64)
                    loss_dx = torch.tensor(0, dtype=torch.float64)
                    loss_dz = torch.tensor(0, dtype=torch.float64)

                else:

                    J_ed, J_e, J_d, idx_trunc = self.SAE.jacobian_norm_trunc_wo_jac_loss(z_gt_tr_batch, X_train, self.trunc_period)
                    

                    dx_train = self.net.f(X_train)


                    dz_gt_tr_batch = dz_gt_tr_batch.unsqueeze(2)

                    dx_data_train = J_e @ dz_gt_tr_batch[:, idx_trunc]
                    dx_data_train = dx_data_train.squeeze()

                    dx_train = dx_train.unsqueeze(2)

                    dx_train = dx_train.squeeze()
                    
#                   # consistency loss
                    loss_dx = torch.mean((dx_train - dx_data_train) ** 2)
                    
                    dz_train = J_ed @ dz_gt_tr_batch[:, idx_trunc]
        
                    dx_train = dx_train.unsqueeze(2)
                    dz_train_dec = J_d @ dx_train
                    dz_gt_tr_batch = dz_gt_tr_batch.squeeze()
            
                    dz_train = dz_train.unsqueeze(2)
                    dz_train_dec = dz_train_dec.unsqueeze(2)
                    

                    dz_train = dz_train.squeeze()
                    dz_train_dec = dz_train_dec.squeeze()

                    loss_AE_jac = torch.mean((dz_gt_tr_batch[:, idx_trunc] - dz_train) ** 2)
                    loss_dz = torch.mean((dz_gt_tr_batch[:, idx_trunc] - dz_train_dec) ** 2)

                loss = loss_GFINNs+self.lambda_r*loss_AE+ self.lambda_dx*loss_dx +self.lambda_dz*loss_dz+ self.lambda_jac*loss_AE_jac

                if i < self.epochs:
                    self.__optimizer.zero_grad()
                    #print(loss)
                    loss.backward(retain_graph=True)
                    #loss.backward()
                    self.__optimizer.step()
                    self.__scheduler.step()

                
            if  i % self.print_every == 0 or i == self.epochs:
                
                indices  = np.arange(self.dim_t-1)
                np.random.shuffle(indices)
                batch_indices_test = indices[:self.batch_size]
                row_indices_batch = torch.cat([torch.tensor(idx_r+batch_indices_test) for idx_r in range(0, z_gt_tt.size(0), self.dim_t-1)])
                
                z_gt_tt_batch = z_gt_tt[row_indices_batch,:]
                z1_gt_tt_batch = z1_gt_tt[row_indices_batch,:]
                dz_gt_tt_batch = dz_gt_tt[row_indices_batch,:]
                
                z_sae_tt, X_test = self.SAE(z_gt_tt_batch)

                z1_sae_tt, y_test = self.SAE(z1_gt_tt_batch)
                
                

#                 with torch.no_grad():
                loss_GFINNs_test = self.__criterion(self.net(X_test), y_test)

                # reconstruction loss
                loss_AE_test = torch.mean((z_sae_tt - z_gt_tt_batch) ** 2)


            
                if  ((self.lambda_jac == 0 and self.lambda_dx == 0) and self.lambda_dz == 0): 
                    loss_AE_jac_test = torch.tensor(0, dtype=torch.float64)
                    loss_dx_test = torch.tensor(0, dtype=torch.float64)
                    loss_dz_test = torch.tensor(0, dtype=torch.float64)

                else:

                    J_ed, J_e, J_d, idx_trunc = self.SAE.jacobian_norm_trunc_wo_jac_loss(z_gt_tt_batch, X_test, self.trunc_period)
                   
                    
#                     with torch.no_grad()
                    dx_test = self.net.f(X_test)


                    dz_gt_tt_batch = dz_gt_tt_batch.unsqueeze(2)

                    dx_data_test = J_e @ dz_gt_tt_batch[:, idx_trunc]
                    dx_data_test = dx_data_test.squeeze()

                    dx_test = dx_test.unsqueeze(2)

                    dx_test = dx_test.squeeze()

#                   # consistency loss
                    loss_dx_test = torch.mean((dx_test - dx_data_test) ** 2)
                    
                    dz_test = J_ed @ dz_gt_tt_batch[:, idx_trunc]
        
                    dx_test = dx_test.unsqueeze(2)
                    dz_test_dec = J_d @ dx_test
                    dz_gt_tt_batch = dz_gt_tt_batch.squeeze()
            
                    dz_test = dz_test.unsqueeze(2)
                    dz_test_dec = dz_test_dec.unsqueeze(2)
                    

                    dz_test = dz_test.squeeze()
                    dz_test_dec = dz_test_dec.squeeze()

                    loss_AE_jac_test = torch.mean((dz_gt_tt_batch[:, idx_trunc] - dz_test) ** 2)
                    loss_dz_test = torch.mean((dz_gt_tt_batch[:, idx_trunc] - dz_test_dec) ** 2)

                loss_test = loss_GFINNs_test+self.lambda_r*loss_AE_test+ self.lambda_dx*loss_dx_test +self.lambda_dz*loss_dz_test+ self.lambda_jac*loss_AE_jac_test
  
  

                print(' ADAM || It: %05d, Loss: %.4e, loss_GFINNs: %.4e, loss_AE_recon: %.4e, loss_jac: %.4e, loss_dx: %.4e, loss_dz: %.4e, Test: %.4e' %
                      (i, loss.item(),loss_GFINNs.item(),loss_AE.item(),loss_AE_jac.item(),loss_dx.item(),loss_dz.item(), loss_test.item()))
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
                    #loss_history.append([i, loss.item(), err_max, *output])
                    loss_history.append([i, loss.item(), loss_test.item(), *output])
                    loss_GFINNs_history.append([i, loss_GFINNs.item(), *output])#, loss_GFINNs_test.item()
                    loss_AE_history.append([i, loss_AE.item(), *output])#, loss_AE_test.item()
                    loss_dx_history.append([i, loss_dx.item(), *output])
                    loss_dz_history.append([i, loss_dz.item(), *output])
                    loss_AE_jac_history.append([i, loss_AE_jac.item(), *output])
             #       loss_AE_GFINNs_history.append([i, loss_AE_GFINNs.item(), loss_AE_GFINNs_test.item(), *output])
                else:
#                     loss_history.append([i, loss.item(), err_max])
#                     loss_history.append([i, loss.item(), err_max])
                    loss_history.append([i, loss.item(), loss_test.item()])
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
                    
            
                
        self.loss_history = np.array(loss_history)
        self.loss_GFINNs_history = np.array(loss_GFINNs_history)
        self.loss_AE_history = np.array(loss_AE_history)
        self.loss_AE_jac_history = np.array(loss_AE_jac_history)
        self.loss_dx_history = np.array(loss_dx_history)
        self.loss_dz_history = np.array(loss_dz_history)
                
        self.loss_AE_history[:,1:]*= self.lambda_r
        self.loss_AE_jac_history[:,1:]*= self.lambda_jac
        self.loss_dx_history[:,1:]*= self.lambda_dx
        self.loss_dz_history[:,1:]*= self.lambda_dz
        
        _, x_de = self.SAE(z_gt)

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
                
                
        ##clear some memory
        z_gt_tr = None
        
        z_gt_tt = None

        z1_gt_tr = None
        
        z1_gt_tt = None   
        dz_gt_tr = None
        
        dz_gt_tt = None

        z_gt_tr_all = None



        # print('Done!', flush=True)
        return self.loss_history, self.loss_GFINNs_history, self.loss_AE_history, self.loss_AE_jac_history, self.loss_dx_history, self.loss_dz_history



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
#             torch.save({'train_indices':self.train_indices,'err_array':self.err_array,'err_max_para':self.err_max_para}, path+'/train_indices.p')
            
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

        
        z_gt = self.z_gt
        z_tt = self.z_tt_all

        z0 = z_tt[::self.dim_t, :]

#         mu0 = self.mu_tt[::self.dim_t, :]



        # Forward pass
        with torch.no_grad():
            z_sae, x_all = self.SAE(z_tt)

        #z_norm = self.SAE.normalize(z)

        _, x0 = self.SAE(z0)
        
                                     
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


        dS, L = self.net.netS(x0)


        dE = dE.unsqueeze(1)

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
        z_gfinn = self.SAE.decode(x_gfinn)



        # z_gfinn_all_norm = self.SAE.decode(x_net_all)
        # z_gfinn_all = self.SAE.denormalize(z_gfinn_all_norm)

        # Load Ground Truth and Compute MSE
        z_gt = self.z_gt
        z_tt_all = self.z_tt_all
        print_mse(z_gfinn, z_tt_all, self.sys_name)
        print_mse(z_sae, z_gt, self.sys_name)

#         print(z_gfinn.shape)
#         print(z_gt.shape)


        # Plot results
        pid = 0 #index for test para
        if (self.save_plots):
            #plot_name = 'SPNN Full Integration (Latent)'
            #print((pid+1)*self.dim_t)
            #print(z_tt_all.shape)
            
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
                    
                    
                fig, ax1 = plt.subplots(1,1, figsize=(10, 10))
     
                plot_name = '[1DBurgers] solution_' + self.AE_name
                fig.suptitle(plot_name)
            
                pid = 15
                
                z_gfinn_plot = z_gfinn[pid*self.dim_t:(pid+1)*self.dim_t,:]
                z_gt_plot = z_tt_all[pid*self.dim_t:(pid+1)*self.dim_t,:]
                
                N = z_gfinn_plot.shape[1]
                dx = 0.02
                x_vec = np.linspace(dx,N*dx,N)
                ax1.plot(x_vec, z_gfinn_plot[-1,:].detach().cpu(),'b')
                ax1.plot(x_vec, z_gt_plot[-1,:].detach().cpu(),'k--')
                l1, = ax1.plot([],[],'k--')
                l2, = ax1.plot([],[],'b')
                ax1.legend((l1, l2), ('GT','Net'))
                ax1.set_ylabel('$u$ [-]')
                ax1.set_xlabel('$x$ [s]')
                ax1.grid()

                save_dir = os.path.join(self.output_dir, plot_name)
                plt.savefig(save_dir)
                plt.clf()
            

            elif self.sys_name == 'rolling_tire':
                x_q, x_v, x_sigma = self.SAE.split_latent(x_gfinn)

                # Plot latent variables
                if (self.save_plots == True):
                    plot_name = '[RT] Latent Variables_' + self.AE_name
                    plot_latent_tire(x_q, x_v, x_sigma, self.dataset.dt, plot_name, self.output_dir)
                    
            

        print("\n[GFINNs Testing Finished]\n")

