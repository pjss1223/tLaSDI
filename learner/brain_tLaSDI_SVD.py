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

#from utilities.plot_gfinns import plot_results, plot_latent

import torch
import torch.optim as optim
import numpy as np

from model import SparseAutoEncoder, StackedSparseAutoEncoder
from dataset_sim import load_dataset, split_dataset
from utilities.plot import plot_results, plot_latent_visco, plot_latent_tire, plot_latent
from utilities.utils import print_mse, all_latent
import matplotlib.pyplot as plt

from learner.utils import mse, wasserstein, div, grad



class Brain_tLaSDI_SVD:
    '''Runner based on torch.
    '''
    brain = None

    @classmethod
    def Init(cls,  net, dt, z_gt, sys_name, output_dir, save_plots, criterion, optimizer, lr,
             iterations, lbfgs_steps, AE_name,dset_dir,output_dir_AE,save_plots_AE,latent_dim,layer_vec_SAE,layer_vec_SAE_q,layer_vec_SAE_v,layer_vec_SAE_sigma,
             activation_SAE,lr_SAE,lambda_r_SAE,lambda_jac_SAE,lambda_dx,lambda_dz,miles_lr=[30000],gamma_lr=0.1, path=None, load_path=None, batch_size=None,
             batch_size_test=None, weight_decay_AE = 0, weight_decay_GFINNs = 0, print_every=1000, save=False, load = False,  callback=None, dtype='double',
             device='cpu',trunc_period=1):
        cls.brain = cls( net, dt, z_gt, sys_name, output_dir, save_plots, criterion,
                         optimizer, lr, iterations, lbfgs_steps,AE_name,dset_dir,output_dir_AE,save_plots_AE,latent_dim,layer_vec_SAE,
                         layer_vec_SAE_q,layer_vec_SAE_v,layer_vec_SAE_sigma,activation_SAE,lr_SAE,lambda_r_SAE,lambda_jac_SAE,lambda_dx,lambda_dz,miles_lr,gamma_lr, path,load_path, batch_size,
                         batch_size_test, weight_decay_AE, weight_decay_GFINNs, print_every, save, load, callback, dtype, device,trunc_period)

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

    def __init__(self,  net, dt,z_gt,sys_name, output_dir,save_plots, criterion, optimizer, lr, iterations, lbfgs_steps,AE_name,dset_dir,output_dir_AE,save_plots_AE,latent_dim,layer_vec_SAE,layer_vec_SAE_q,layer_vec_SAE_v,layer_vec_SAE_sigma, activation_SAE,lr_SAE,lambda_r_SAE,lambda_jac_SAE,lambda_dx,lambda_dz,miles_lr,gamma_lr, path,load_path, batch_size, batch_size_test, weight_decay_AE, weight_decay_GFINNs, print_every, save, load, callback, dtype, device,trunc_period):
        #self.data = data
        self.net = net
        self.sys_name = sys_name
        self.output_dir = output_dir
        self.save_plots = save_plots
        #self.x_trunc = x_trunc
#        self.latent_idx = latent_idx
        self.dt = dt
        self.z_gt = z_gt
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay_GFINNs = weight_decay_GFINNs
        self.weight_decay_AE = weight_decay_AE
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
        #self.dset_dir = dset_dir
        self.output_dir_AE = output_dir_AE
        self.trunc_period = trunc_period
        self.miles_lr = miles_lr
        self.gamma_lr = gamma_lr
        self.latent_dim = latent_dim
        

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self.save_plots_AE = save_plots_AE




        # Dataset Parameters
        self.dset_dir = dset_dir
        self.dataset = load_dataset(self.sys_name, self.dset_dir, self.device, self.dtype)
        self.dt = self.dataset.dt
        #print(self.dt)
        self.dim_t = self.dataset.dim_t
        
        
        ###--------------------- Half trajectories        
#         if self.sys_name == 'GC_SVD':
#             self.dataset.z = self.dataset.z[:40]


        self.train_snaps, self.test_snaps = split_dataset(self.sys_name, self.dim_t-1)
        self.train_traj, self.test_traj = split_dataset(self.sys_name, self.dataset.z.shape[0]) # valid only for GC_SVD, VC_SPNN_SVD

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
        loss_AE_recon_history = []
        loss_AE_jac_history = []
        loss_dx_history = []
        loss_dz_history = []



        if (self.sys_name == "GC_SVD") or (self.sys_name == "VC_SPNN_SVD"):
            
            self.dim_full = 100
            
            random_matrix = torch.nn.Parameter((torch.randn([self.dim_full, self.dataset.dim_z])).requires_grad_(False))
            #random_matrix = np.random.rand(self.dim_full, self.dim_z)

            # Calculate the QR decomposition of the matrix
            q, _ = torch.linalg.qr(random_matrix)
            
            if self.dtype == "double":
                q = q.double()
            if self.device == 'gpu':
                q = q.to(torch.device('cuda'))
                
            q = q.detach()
            
            path = './data/'

            torch.save(q,path + '/q_data_GC.p')
            
            if self.sys_name == "GC_SVD":
                q  = torch.load(path + '/q_data_GC.p')
            elif self.sys_name == "VC_SPNN_SVD":
                q  = torch.load(path + '/q_data.p')

          
            
            x_tmp = self.dataset.z.reshape([-1,self.dataset.dim_z])
            dx_tmp = self.dataset.dz.reshape([-1,self.dataset.dim_z])
            
#             print(self.dataset.dim_z)


            z_gt =  x_tmp @ q.t()
    
#             print(x_tmp.requires_grad)
#             print(z_gt.requires_grad)

            dz_gt = dx_tmp @ q.t()
            
            z_gt_traj = z_gt.reshape([-1,self.dataset.dim_t,self.dim_full])
            dz_gt_traj = dz_gt.reshape([-1,self.dataset.dim_t,self.dim_full])

            
            z_gt_tr = z_gt_traj[self.train_traj,:-1,:]
            z_gt_tt = z_gt_traj[self.test_traj,:-1,:]
            z_gt_tt_all = z_gt_traj[self.test_traj,:,:]
            
            
            
            z1_gt_tr = z_gt_traj[self.train_traj,1:,:]
            z1_gt_tt = z_gt_traj[self.test_traj,1:,:]
            
            dz_gt_tr = dz_gt_traj[self.train_traj,:-1,:]
            dz_gt_tt = dz_gt_traj[self.test_traj,:-1:,:] # fixed
            
            
            z_gt_tr = z_gt_tr.reshape([-1,self.dim_full])
            z_gt_tt = z_gt_tt.reshape([-1,self.dim_full])
            
            
            

            z1_gt_tr = z1_gt_tr.reshape([-1,self.dim_full])
            z1_gt_tt = z1_gt_tt.reshape([-1,self.dim_full])
            
            
            dz_gt_tr = dz_gt_tr.reshape([-1,self.dim_full])
            dz_gt_tt = dz_gt_tt.reshape([-1,self.dim_full])
            
            self.z_gt = z_gt
            self.z_gt_tt_all = z_gt_tt_all
            self.z_gt_tt = z_gt_tt
            self.z1_gt_tt = z1_gt_tt
            self.z_gt_x_traj = z_gt_traj[:,:-1,:]
            self.z_gt_y_traj = z_gt_traj[:,1:,:]
            
            self.z_gt_x = self.z_gt_x_traj.reshape([-1,self.dim_full])
            self.z_gt_y = self.z_gt_y_traj.reshape([-1,self.dim_full])
            
            

            
        else:
            z_gt_tr = self.dataset.z[self.train_snaps, :]
            z_gt_tt = self.dataset.z[self.test_snaps, :]
            dz_gt_tr = self.dataset.dz[self.train_snaps, :]



            dz_gt_tt = self.dataset.dz[self.test_snaps, :]



            z1_gt_tr = self.dataset.z[self.train_snaps+1, :]
            z1_gt_tt = self.dataset.z[self.test_snaps+1, :]

            z_gt = self.dataset.z
            
            
        
        self.z_data = Data(z_gt_tr,z1_gt_tr,z_gt_tt,z1_gt_tt,self.device)
        
        
        
        
        
        self.dataset.dz = None
        
        dz_gt_tr_tmp = dz_gt_tr
        dz_gt_tt_tmp = dz_gt_tt
        
        
        U, S, VT = torch.svd(z_gt_tr.t())
        
        

        Ud = U[:,:self.latent_dim]

        
        self.Ud = Ud

        
        prev_lr = self.__optimizer.param_groups[0]['lr']
        
        
        x_gt = z_gt @ Ud
        
        #Ud.requires_grad_(True)
        
#         print(Ud.requires_grad)

#         if self.sys_name == "GC_SVD":
            
#             self.min_value_col1 = torch.min(x_gt[:, 0])
#             self.max_value_col1 = torch.max(x_gt[:, 0])

#             self.min_value_col2 = torch.min(x_gt[:, 1])
#             self.max_value_col2 = torch.max(x_gt[:, 1])

#             self.min_value_col3 = torch.min(x_gt[:, 2])
#             self.max_value_col3 = torch.max(x_gt[:, 2])

#             self.min_value_col4 = torch.min(x_gt[:, 3])
#             self.max_value_col4 = torch.max(x_gt[:, 3])

#         print(self.min_value_col1)
#         print(self.max_value_col1)

#         print(self.min_value_col2)
#         print(self.max_value_col2)

#         print(self.min_value_col3)
#         print(self.max_value_col3)

#         print(self.min_value_col4)
#         print(self.max_value_col4)
        
        for i in range(self.iterations + 1):

            
            z_gt_tr,z1_gt_tr, mask_tr = self.z_data.get_batch(self.batch_size)
 
            
            dz_gt_tr= dz_gt_tr_tmp[mask_tr]

            X_train = z_gt_tr @ Ud          
            
            z_sae_tr = X_train @ Ud.t()
            
            y_train = z1_gt_tr @ Ud
            
            z1_sae_tr = y_train @ Ud.t()


#             X_train, y_train= x, x1

#             if self.sys_name == 'GC_SVD':


#                 # Define the desired range for rescaling
#                 self.new_min = 0.15
#                 self.new_max = 1.85



#                 #only for GC_SVD ESP
#                 scale_factor = (1.85 - 0.15) / (-1.5686 +4.7011)
#                 shift_factor = 0.15 +4.7011 * scale_factor
#                 X_train[:,0] = X_train[:,0] * scale_factor + shift_factor

#                 scale_factor = (4.4066+4.1807) / (4.0967 +4.1021)
#                 shift_factor = -4.1807+4.1021 * scale_factor
#                 X_train[:,1] = X_train[:,1] * scale_factor + shift_factor

#                 scale_factor = (3.2 - 0.7) / (1.0980 + 1.0939)
#                 shift_factor = 0.7+1.0939* scale_factor
#                 X_train[:,2] = X_train[:,2] * scale_factor + shift_factor

#                 scale_factor = (3.2 - 0.7) / (1.1758 +1.1194)
#                 shift_factor = 0.7+1.1194* scale_factor
#                 X_train[:,3] = X_train[:,3] * scale_factor + shift_factor
                
#                 scale_factor = (1.85 - 0.15) / (-1.5686 +4.7011)
#                 shift_factor = 0.15+4.7011 * scale_factor
#                 y_train[:,0] = y_train[:,0] * scale_factor + shift_factor

#                 scale_factor = (4.4066+4.1807) / (4.0967 +4.1021)
#                 shift_factor = -4.1807 + 4.1021 * scale_factor
#                 y_train[:,1] = y_train[:,1] * scale_factor + shift_factor

#                 scale_factor = (3.2 - 0.7) / (1.0980 + 1.0939)
#                 shift_factor = 0.7+ 1.0939* scale_factor
#                 y_train[:,2] = y_train[:,2] * scale_factor + shift_factor

#                 scale_factor = (3.2 - 0.7) / (1.1758 +1.1194)
#                 shift_factor = 0.7+1.1194* scale_factor
#                 y_train[:,3] = y_train[:,3] * scale_factor + shift_factor
                
                
                
                
#                 #only for GC_SVD ESP
#                 scale_factor = (1.85 - 0.15) / (self.max_value_col1 -self.min_value_col1)
#                 shift_factor = 0.15 -self.min_value_col1 * scale_factor
#                 X_train[:,0] = X_train[:,0] * scale_factor + shift_factor
#                 y_train[:,0] = y_train[:,0] * scale_factor + shift_factor

#                 scale_factor = (4.4066+4.1807) / (self.max_value_col2 -self.min_value_col2)
#                 shift_factor = -4.1807-self.min_value_col2* scale_factor
#                 X_train[:,1] = X_train[:,1] * scale_factor + shift_factor
#                 y_train[:,1] = y_train[:,1] * scale_factor + shift_factor

#                 scale_factor = (3.2 - 0.7) / (self.max_value_col3 -self.min_value_col3)
#                 shift_factor = 0.7-self.min_value_col3* scale_factor
#                 X_train[:,2] = X_train[:,2] * scale_factor + shift_factor
#                 y_train[:,2] = y_train[:,2] * scale_factor + shift_factor

#                 scale_factor = (3.2 - 0.7) / (self.max_value_col4 -self.min_value_col4)
#                 shift_factor = 0.7-self.min_value_col4* scale_factor
#                 X_train[:,3] = X_train[:,3] * scale_factor + shift_factor
#                 y_train[:,3] = y_train[:,3] * scale_factor + shift_factor
    
                
#                 min_value_col1 = torch.min(X_train[:, 0])
#                 max_value_col1 = torch.max(X_train[:, 0])
                
#                 min_value_col2 = torch.min(X_train[:, 1])
#                 max_value_col2 = torch.max(X_train[:, 1])
                
#                 min_value_col3 = torch.min(X_train[:, 2])
#                 max_value_col3 = torch.max(X_train[:, 2])
                
#                 min_value_col4 = torch.min(X_train[:, 3])
#                 max_value_col4 = torch.max(X_train[:, 3])
                
#                 print(min_value_col1)
#                 print(max_value_col1)
                
#                 print(min_value_col2)
#                 print(max_value_col2)
                
#                 print(min_value_col3)
#                 print(max_value_col3)
                
#                 print(min_value_col4)
#                 print(max_value_col4)

            loss_GFINNs = self.__criterion(X_train, y_train)
    
            


            loss_AE_recon = torch.mean((z_sae_tr - z_gt_tr) ** 2)

            
            if  ((self.lambda_jac == 0 and self.lambda_dx == 0) and self.lambda_dz == 0): 
                loss_AE_jac = torch.tensor(0, dtype=torch.float64)
                loss_dx = torch.tensor(0, dtype=torch.float64)
                loss_dz = torch.tensor(0, dtype=torch.float64)                
            else:
                

                J_e = Ud.t()
                J_d = Ud
            
                J_ed = Ud @ Ud.t()
            
                

                dx_train = self.net.f(X_train)


                dz_gt_tr = dz_gt_tr.unsqueeze(2)
                
                idx_trunc = range(dz_gt_tr.shape[1])
                
#                 print(J_e.shape)
#                 print(dz_gt_tr.shape)



                dx_data_train = J_e @ dz_gt_tr[:,idx_trunc]
                dx_data_train = dx_data_train.squeeze()

                # print(dx_train.shape)
                # print(grad(self.SAE.decode(X_train),X_train).shape)
                dx_train = dx_train.unsqueeze(2)
                dx_train = dx_train.squeeze()
                #dz_train = grad(self.SAE.decode(X_train),X_train)@dx_train
                loss_dx = torch.mean((dx_train - dx_data_train) ** 2)

                
                dz_train = J_ed @ dz_gt_tr[:, idx_trunc]

        
                dx_train = dx_train.unsqueeze(2)
                dz_train_dec = J_d @ dx_train
                dz_gt_tr = dz_gt_tr.squeeze()
            
                dz_train = dz_train.unsqueeze(2)
                dz_train_dec = dz_train_dec.unsqueeze(2)
                    

                dz_train = dz_train.squeeze()
                dz_train_dec = dz_train_dec.squeeze()

                loss_AE_jac =  torch.mean((dz_train - dz_gt_tr[:,idx_trunc]) ** 2)
                loss_dz = torch.mean((dz_gt_tr[:, idx_trunc] - dz_train_dec) ** 2)

            loss = loss_GFINNs+self.lambda_r*loss_AE_recon+self.lambda_dx*loss_dx+self.lambda_dz*loss_dz+self.lambda_jac*loss_AE_jac
            
            



            #print(loss) #tensor(0.0008, grad_fn=<MseLossBackward0>)
            Loss_early = 1e-10



            if i % self.print_every == 0 or i == self.iterations:
                z_gt_tt,z1_gt_tt, mask_tt = self.z_data.get_batch_test(self.batch_size_test)
                dz_gt_tt = dz_gt_tt_tmp[mask_tt]
                x_tt = z_gt_tt @ Ud
                z_sae_tt = x_tt @ Ud.t()
                x1_tt = z1_gt_tt @ Ud
                z1_sae_tt = x1_tt @ Ud.t()
                X_test, y_test = x_tt, x1_tt
                
#                 scale_factor = (1.85 - 0.15) / (-1.5686 +4.7011)
#                 shift_factor = 0.15 +4.7011 * scale_factor
#                 X_test[:,0] = X_test[:,0] * scale_factor + shift_factor

#                 scale_factor = (4.4066+4.1807) / (4.0967 +4.1021)
#                 shift_factor = -4.1807 + 4.1021 * scale_factor
#                 X_test[:,1] = X_test[:,1] * scale_factor + shift_factor

#                 scale_factor = (3.2 - 0.7) / (1.0980 + 1.0939)
#                 shift_factor = 0.7+ 1.0939* scale_factor
#                 X_test[:,2] = X_test[:,2] * scale_factor + shift_factor

#                 scale_factor = (3.2 - 0.7) / (1.1758 +1.1194)
#                 shift_factor = 0.7+1.1194* scale_factor
#                 X_test[:,3] = X_test[:,3] * scale_factor + shift_factor

#                 if self.sys_name == 'GC_SVD':

#                     scale_factor = (1.85 - 0.15) / (self.max_value_col1 -self.min_value_col1)
#                     shift_factor = 0.15 -self.min_value_col1 * scale_factor
#                     X_test[:,0] = X_test[:,0] * scale_factor + shift_factor
#                     y_test[:,0] = y_test[:,0] * scale_factor + shift_factor

#                     scale_factor = (4.4066+4.1807) / (self.max_value_col2 -self.min_value_col2)
#                     shift_factor = -4.1807-self.min_value_col2* scale_factor
#                     X_test[:,1] = X_test[:,1] * scale_factor + shift_factor
#                     y_test[:,1] = y_test[:,1] * scale_factor + shift_factor

#                     scale_factor = (3.2 - 0.7) / (self.max_value_col3 -self.min_value_col3)
#                     shift_factor = 0.7-self.min_value_col3* scale_factor
#                     X_test[:,2] = X_test[:,2] * scale_factor + shift_factor
#                     y_test[:,2] = y_test[:,2] * scale_factor + shift_factor

#                     scale_factor = (3.2 - 0.7) / (self.max_value_col4 -self.min_value_col4)
#                     shift_factor = 0.7-self.min_value_col4* scale_factor
#                     X_test[:,3] = X_test[:,3] * scale_factor + shift_factor
#                     y_test[:,3] = y_test[:,3] * scale_factor + shift_factor

                
                dx_test = self.net.f(X_test)
                dz_gt_tt = dz_gt_tt.unsqueeze(2)
                
                with torch.no_grad():
                    loss_AE_recon_test = torch.mean((z_sae_tt - z_gt_tt) ** 2)


                #loss_AE_jac_test, J_e, J_d, idx_trunc = self.SAE.jacobian_norm_trunc(z_gt_tt_norm, x_tt)
#                 print('Current GPU memory allocated before loss_GFINNs_test: ', torch.cuda.memory_allocated() / 1024 ** 3, 'GB')
                

                loss_GFINNs_test = self.__criterion(X_test, y_test)
                
                
                if  ((self.lambda_jac == 0 and self.lambda_dx == 0) and self.lambda_dz == 0): 
                    loss_AE_jac_test = torch.tensor(0, dtype=torch.float64)
                    loss_dx_test = torch.tensor(0, dtype=torch.float64)
                    loss_dz_test = torch.tensor(0, dtype=torch.float64)
                else:

#                     if self.device == 'cpu':
#                         loss_AE_jac_test, J_e, J_d, idx_trunc  = self.SAE.jacobian_norm_trunc(z_gt_tt_norm,x_tt ,self.trunc_period)
#                     else:
#                         loss_AE_jac_test, J_e, J_d, idx_trunc  = self.SAE.jacobian_norm_trunc_gpu(z_gt_tt_norm, x_tt, self.trunc_period)


                    #dx_data_test = grad(self.SAE.encode(z_gt_tt_norm), z_gt_tt_norm) @ dz_gr_tt_norm
                    #J_ed, J_e, J_d, idx_trunc = self.SAE.jacobian_norm_trunc_wo_jac_loss(z_gt_tt_norm, x_tt, self.trunc_period)

                    dx_data_test = J_e @ dz_gt_tt[:,idx_trunc]
                    dx_data_test = dx_data_test.squeeze()
                    dz_gt_tt = dz_gt_tt.squeeze()

                    dx_test = dx_test.unsqueeze(2)
                    dx_test = dx_test.squeeze()
                    loss_dx_test = torch.mean((dx_test - dx_data_test) ** 2)
                    #dz_test = grad(self.SAE.decode(X_test), X_test) @ dx_test
                    
                    
                    dz_gt_tt = dz_gt_tt.unsqueeze(2)
                    
                    dz_test = J_ed @ dz_gt_tt[:,idx_trunc]
                    
                    
                    dx_test = dx_test.unsqueeze(2)
                    dz_test_dec = J_d @ dx_test
                    dz_gt_tt = dz_gt_tt.squeeze()
            
                    dz_test = dz_test.unsqueeze(2)
                    dz_test_dec = dz_test_dec.unsqueeze(2)
                    

                    dz_test = dz_test.squeeze()
                    dz_test_dec = dz_test_dec.squeeze()
#                     dz_train = dz_train.squeeze()
#                     dz_gt_tr_batch = dz_gt_tr_batch.squeeze()

#                     # model approximation loss
#                     loss_dz = torch.mean((dz_train - dz_gt_tr_batch[:, idx_trunc]) ** 2)
#                     print(dz_test.shape)
#                     print(dz_gt_tt_norm[:,idx_trunc].shape)
                    loss_AE_jac_test =  torch.mean((dz_test - dz_gt_tt[:,idx_trunc]) ** 2)
                    loss_dz_test = torch.mean((dz_gt_tt[:, idx_trunc] - dz_test_dec) ** 2)



                loss_test = loss_GFINNs_test+self.lambda_r*loss_AE_recon_test+self.lambda_dx*loss_dx_test+self.lambda_dz*loss_dz_test+self.lambda_jac*loss_AE_jac_test

                # print('{:<9}a loss: %.4e{:<25}Test loss: %.4e{:<25}'.format(i, loss.item(), loss_test.item()), flush=True)
                print(' ADAM || It: %05d, Loss: %.4e, loss_GFINNs: %.4e, loss_AE_recon: %.4e, loss_dx: %.4e,loss_jac: %.4e, loss_dz: %.4e, Test: %.4e' %
                      (i, loss.item(),loss_GFINNs.item(),loss_AE_recon.item(),loss_dx.item(),loss_AE_jac.item(),loss_dz.item(), loss_test.item()))
                if torch.any(torch.isnan(loss)):
                    self.encounter_nan = True
                    print('Encountering nan, stop training', flush=True)
                    return None
                if self.save:
                    if not os.path.exists('model'): os.mkdir('model')
                    if self.path == None:
                        torch.save(self.net, 'model/model{}.pkl'.format(i))
                        #torch.save(self.SAE, 'model/AE_model{}.pkl'.format(i))
                    else:
                        if not os.path.isdir('model/' + self.path): os.makedirs('model/' + self.path)
                        torch.save(self.net, 'model/{}/model{}.pkl'.format(self.path, i))
                        #torch.save(self.SAE, 'model/{}/AE_model{}.pkl'.format(self.path, i))
                if self.callback is not None:
                    output = self.callback(self.data, self.net)
                    loss_history.append([i, loss.item(), loss_test.item(), *output])
                    loss_GFINNs_history.append([i, loss_GFINNs.item(), loss_GFINNs_test.item(), *output])
                    loss_AE_recon_history.append([i, loss_AE_recon.item(), loss_AE_recon_test.item(), *output])
                    loss_AE_jac_history.append([i, loss_AE_jac.item(), loss_AE_jac_test.item(), *output])
                    loss_dx_history.append([i, loss_dx.item(), loss_dx_test.item(), *output])
                    loss_dz_history.append([i, loss_dz.item(), loss_dz_test.item(), *output])
             #       loss_AE_GFINNs_history.append([i, loss_AE_GFINNs.item(), loss_AE_GFINNs_test.item(), *output])
                else:
                    loss_history.append([i, loss.item(), loss_test.item()])
                    loss_history.append([i, loss.item(), loss_test.item()])
                    loss_GFINNs_history.append([i, loss_GFINNs.item(), loss_GFINNs_test.item()])
                    loss_AE_recon_history.append([i, loss_AE_recon.item(), loss_AE_recon_test.item()])
                    loss_AE_jac_history.append([i, loss_AE_jac.item(), loss_AE_jac_test.item()])
                    loss_dx_history.append([i, loss_dx.item(), loss_dx_test.item()])
                    loss_dz_history.append([i, loss_dz.item(), loss_dz_test.item()])
               #     loss_AE_GFINNs_history.append([i, loss_AE_GFINNs.item(), loss_AE_GFINNs_test.item(), *output])
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
                #print('Current GPU memory allocated before zerograd: ', torch.cuda.memory_allocated() / 1024 ** 3, 'GB')
                self.__optimizer.zero_grad()
                #print(loss)
                loss.backward(retain_graph=False)
                #loss.backward()
                #print('Current GPU memory allocated before step: '+ str(i), torch.cuda.memory_allocated() / 1024 ** 3, 'GB')
                self.__optimizer.step()
                #print('Current GPU memory allocated after step: '+ str(i), torch.cuda.memory_allocated() / 1024 ** 3, 'GB')

                self.__scheduler.step()
                
        self.loss_history = np.array(loss_history)
        self.loss_GFINNs_history = np.array(loss_GFINNs_history)
        self.loss_AE_recon_history = np.array(loss_AE_recon_history)
        self.loss_AE_jac_history = np.array(loss_AE_jac_history)
        self.loss_dx_history = np.array(loss_dx_history)
        self.loss_dz_history = np.array(loss_dz_history)
                
        self.loss_AE_recon_history[:,1:]*= self.lambda_r
        self.loss_AE_jac_history[:,1:]*= self.lambda_jac
        self.loss_dx_history[:,1:]*= self.lambda_dx
        self.loss_dz_history[:,1:]*= self.lambda_dz
        

#         _, x_de = self.SAE(z_gt_norm)
        if self.sys_name == 'GC_SVD':
            x_de = z_gt.reshape([-1,self.dim_full]) @ Ud
        else:
            x_de = z_gt @ Ud

        if self.sys_name == 'viscoelastic':
            # Plot latent variables
            if (self.save_plots == True):
                plot_name = '[VC] AE Latent Variables_' + self.AE_name
                plot_latent_visco(x_de, self.dataset.dt, plot_name, self.output_dir)
        elif self.sys_name == 'GC_SVD_concat':
            # Plot latent variables
            if (self.save_plots == True):
                plot_name = '[GC_SVD_concat] AE Latent Variables_' + self.AE_name
                plot_latent_visco(x_de, self.dataset.dt, plot_name, self.output_dir)
        elif self.sys_name == 'GC':
            # Plot latent variables
            if (self.save_plots == True):
                plot_name = '[GC] AE Latent Variables_' + self.AE_name
                plot_latent_visco(x_de, self.dataset.dt, plot_name, self.output_dir)
        elif self.sys_name == 'GC_SVD':
            # Plot latent variables
            pid = 0
            
            if (self.save_plots == True):
                plot_name = '[GC_SVD] AE Latent Variables_' + self.AE_name
                plot_latent_visco(x_de[pid*self.dim_t:(pid+1)*self.dim_t], self.dataset.dt, plot_name, self.output_dir)

        elif self.sys_name == '1DBurgers':

            # Plot latent variables
            if (self.save_plots == True):
                plot_name = '[1DBurgers] AE Latent Variables_' + self.AE_name
                plot_latent_visco(x_de, self.dataset.dt, plot_name, self.output_dir)

#         elif self.sys_name == 'rolling_tire':
#             x_q, x_v, x_sigma = self.SAE.split_latent(x_de)

#             # Plot latent variables
#             if (self.save_plots == True):
#                 plot_name = '[Rolling Tire] AE Latent Variables_' + self.AE_name
#                 plot_latent_tire(x_q, x_v, x_sigma, self.dataset.dt, plot_name, self.output_dir)

        # print('Done!', flush=True)
        self.dataset.z = None
        
        return self.loss_history, self.loss_GFINNs_history, self.loss_AE_recon_history, self.loss_dx_history, self.loss_AE_jac_history



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
                #self.best_model_AE = torch.load('model/AE_model{}.pkl'.format(iteration))
            else:
                self.best_model = torch.load('model/{}/model{}.pkl'.format(self.path, iteration))
                #self.best_model_AE = torch.load('model/{}/AE_model{}.pkl'.format(self.path, iteration))
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
                X_train, y_train,_ = self.data.get_batch(None)

                X_test, y_test,_ = self.data.get_batch_test(None)

                # loss, _ = self.best_model.criterion(self.best_model(X_train), y_train)
                # loss_test, _ = self.best_model.criterion(self.best_model(X_test), y_test)

                loss = self.best_model.criterion(self.best_model(X_train), y_train)
                loss_test = self.best_model.criterion(self.best_model(X_test), y_test)
                # print('Train loss: {:<25}Test loss: {:<25}'.format(loss.item(), loss_test.item()), flush=True)
                it = self.it + 1
                if it % self.print_every == 0 or it == self.lbfgs_steps:
                    print('L-BFGS|| It: %05d, Loss: %.4e, Test: %.4e' %
                          (it, loss.item(), loss_test.item()))
                self.it = it
                if loss.requires_grad:
                    loss.backward(retain_graph=False)
                return loss

            optim.step(closure)
        print('Done!', flush=True)
        return self.best_model

    # def output(self, data, best_model, loss_history, info, **kwargs):
    #     if self.path is None:
    #         path = './outputs/' + time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    #     else:
    #         path = './outputs/' + self.path
    #     if not os.path.isdir(path): os.makedirs(path)
    #     if data:
    #         def save_data(fname, data):
    #             if isinstance(data, dict):
    #                 np.savez_compressed(path + '/' + fname, **data)
    #             else:
    #                 np.save(path + '/' + fname, data)
    #
    #         save_data('X_train', self.data.X_train_np)
    #         save_data('y_train', self.data.y_train_np)
    #         save_data('X_test', self.data.X_test_np)
    #         save_data('y_test', self.data.y_test_np)
    #     if best_model:
    #         torch.save(self.best_model, path + '/model_best.pkl')
    #     if loss_history:
    #         np.savetxt(path + '/loss.txt', self.loss_history)
    #         plt.plot(self.loss_history[:,0], self.loss_history[:,1],'-')
    #         plt.plot(self.loss_history[:,0], self.loss_history[:,2],'--')
    #         plt.legend(['train loss', 'test loss'])  # , '$\hat{u}$'])
    #         plt.yscale('log')
    #         plt.savefig(path + '/loss_AE_AE10.png')
    #         plt.show()
    #     if info is not None:
    #         with open(path + '/info.txt', 'w') as f:
    #             for key, arg in info.items():
    #                 f.write('{}: {}\n'.format(key, str(arg)))
    #     for key, arg in kwargs.items():
    #         np.savetxt(path + '/' + key + '.txt', arg)

    def output(self, best_model, loss_history, info, **kwargs):
        if self.path is None:
            path = './outputs/' + self.AE_name+'_'+ time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        else:
            path = './outputs/' + self.path
        if not os.path.isdir(path): os.makedirs(path)

        if best_model:
            torch.save(self.best_model, path + '/model_best.pkl')
            #torch.save(self.best_model_AE, path + '/model_best_AE.pkl')
        if loss_history:
            np.savetxt(path + '/loss.txt', self.loss_history)
            p1,=plt.plot(self.loss_history[:,0], self.loss_history[:,1],'-')
            p2,= plt.plot(self.loss_history[:,0], self.loss_history[:,2],'--')
            plt.legend(['train loss', 'test loss'])  # , '$\hat{u}$'])
            plt.yscale('log')
            plt.savefig(path + '/loss_'+self.AE_name+self.sys_name+'.png')
            p1.remove()
            p2.remove()

            p3,=plt.plot(self.loss_GFINNs_history[:,0], self.loss_GFINNs_history[:,1],'-')
            p4,=plt.plot(self.loss_GFINNs_history[:,0], self.loss_GFINNs_history[:,2],'--')
            plt.legend(['train loss (GFINNs)', 'test loss (GFINNs)'])  # , '$\hat{u}$'])
            plt.yscale('log')
            plt.savefig(path + '/loss_GFINNs_'+self.AE_name+self.sys_name+'.png')
            p3.remove()
            p4.remove()

            p5,=plt.plot(self.loss_AE_recon_history[:,0], self.loss_AE_recon_history[:,1],'-')
            p6,=plt.plot(self.loss_AE_recon_history[:,0], self.loss_AE_recon_history[:,2],'--')
            plt.legend(['train loss (AE recon)', 'test loss (AE recon)'])  # , '$\hat{u}$'])
            plt.yscale('log')
            plt.savefig(path + '/loss_AE_recon_'+self.AE_name+self.sys_name+'.png')
            p5.remove()
            p6.remove()

            p7,=plt.plot(self.loss_AE_jac_history[:,0], self.loss_AE_jac_history[:,1],'-')
            p8,=plt.plot(self.loss_AE_jac_history[:,0], self.loss_AE_jac_history[:,2],'--')
            plt.legend(['train loss (AE jac)', 'test loss (AE jac)'])  # , '$\hat{u}$'])
            plt.yscale('log')
            plt.savefig(path + '/loss_AE_jac_'+self.AE_name+self.sys_name+'.png')
            p7.remove()
            p8.remove()

            p9,=plt.plot(self.loss_dx_history[:,0], self.loss_dx_history[:,1],'-')
            p10,=plt.plot(self.loss_dx_history[:,0], self.loss_dx_history[:,2],'--')
            plt.legend(['train loss (dx)', 'test loss (dx)'])  # , '$\hat{u}$'])
            plt.yscale('log')
            plt.savefig(path + '/loss_dx_'+self.AE_name+self.sys_name+'.png')
            p9.remove()
            p10.remove()

            p11,=plt.plot(self.loss_dz_history[:,0], self.loss_dz_history[:,1],'-')
            p12,=plt.plot(self.loss_dz_history[:,0], self.loss_dz_history[:,2],'--')
            plt.legend(['train loss (dz)', 'test loss (dz)'])  # , '$\hat{u}$'])
            plt.yscale('log')
            plt.savefig(path + '/loss_dz_'+self.AE_name+self.sys_name+'.png')
            p11.remove()
            p12.remove()

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
            params = [
                {'params': self.net.parameters(), 'lr': self.lr, 'weight_decay': self.weight_decay_GFINNs}
#                 {'params': self.Ud, 'lr': self.lr, 'weight_decay': self.weight_decay_AE}
            ]
            #self.__optimizer = torch.optim.Adam(list(self.net.parameters())+list(self.SAE.parameters()), lr=self.lr, weight_decay=self.weight_decay)
            self.__optimizer = torch.optim.Adam(params)
            self.__scheduler = torch.optim.lr_scheduler.MultiStepLR(self.__optimizer, milestones=self.miles_lr,gamma=self.gamma_lr)
        else:
            raise NotImplementedError

    def __init_criterion(self):
        #print(self.net.criterion)
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

    ##frim spnn
    def test(self):
        print("\n[GFNN Testing Started]\n")

        print('Current GPU memory allocated before testing: ', torch.cuda.memory_allocated() / 1024 ** 3, 'GB')
        self.net = self.best_model

        

        z_gt = self.z_gt_tt_all
        if (self.sys_name == 'GC_SVD') or (self.sys_name == 'VC_SPNN_SVD'):
            self.dim_t = self.dataset.dim_t
            z_gt = z_gt.reshape([-1, self.dim_full])
            z = z_gt[::self.dim_t, :]

        else:
            self.dim_t = self.z_gt.shape[0]      
            z = z_gt[0, :]
            z = torch.unsqueeze(z, 0)
            


        # Forward pass
        x_all =  z_gt @ self.Ud
        

        
        
        z_sae =  x_all @ self.Ud.t()
        

        print('Current GPU memory allocated after eval: ', torch.cuda.memory_allocated() / 1024 ** 3, 'GB')

        #z_norm = self.SAE.normalize(z)

        x = z @ self.Ud
        
        #print(x_all)
        
        
#         if self.sys_name == 'GC_SVD':
            


#             scale_factor = (1.85 - 0.15) / (self.max_value_col1 -self.min_value_col1)
#             shift_factor = 0.15 -self.min_value_col1 * scale_factor
#             x[:,0] = x[:,0] * scale_factor + shift_factor

#             scale_factor = (4.4066+4.1807) / (self.max_value_col2 -self.min_value_col2)
#             shift_factor = -4.1807-self.min_value_col2* scale_factor
#             x[:,1] = x[:,1] * scale_factor + shift_factor

#             scale_factor = (3.2 - 0.7) / (self.max_value_col3 -self.min_value_col3)
#             shift_factor = 0.7-self.min_value_col3* scale_factor
#             x[:,2] = x[:,2] * scale_factor + shift_factor

#             scale_factor = (3.2 - 0.7) / (self.max_value_col4 -self.min_value_col4)
#             shift_factor = 0.7-self.min_value_col4* scale_factor
#             x[:,3] = x[:,3] * scale_factor + shift_factor

            

        if self.dtype == 'float':
            x_net = torch.zeros(x_all.shape).float()
            x_net_all = torch.zeros(x_all.shape).float()

        elif self.dtype == 'double':
            x_net = torch.zeros(x_all.shape).double()
            x_net_all = torch.zeros(x_all.shape).double()

        
        
        if (self.sys_name == 'GC_SVD') or (self.sys_name == 'VC_SPNN_SVD'):
            x_net[::self.dim_t, :] = x
        else:
            x_net[0,:] = x

            
        #print(x_net.shape)
        if self.device == 'gpu':
            x_net = x_net.to(torch.device('cuda'))
            x_net_all = x_net_all.to(torch.device('cuda'))


        
        #with torch.no_grad():
        dE, M = self.net.netE(x)
        #     print(dE.shape)
        # print(M.shape)
        dS, L = self.net.netS(x)
        
#         print(L.shape)
#         print(M.shape)
#         print(dE.shape)
#         print(dS.shape)
        
        x.requires_grad_(False)
        

        if (self.sys_name == 'GC_SVD') or (self.sys_name == 'VC_SPNN_SVD'):
            if self.dtype == 'float':
                dSdt_net = torch.zeros(x_all.shape[0]).float()
                dEdt_net = torch.zeros(x_all.shape[0]).float()
            elif self.dtype == 'double':
                dSdt_net = torch.zeros(x_all.shape[0]).double()
                dEdt_net = torch.zeros(x_all.shape[0]).double()
        

            dE = dE.unsqueeze(1)
            #print(dE.shape)
            dS = dS.unsqueeze(1)


            dEdt = torch.sum(dE.squeeze()* ((dE @ L) + (dS @ M)).squeeze(),1)
            dSdt = torch.sum(dS.squeeze()* ((dE @ L) + (dS @ M)).squeeze(),1)
            

            dEdt_net[::self.dim_t] = dEdt
            dSdt_net[::self.dim_t] = dSdt
        
        else: 
            
            if self.dtype == 'float':
                dSdt_net = torch.zeros(x_all.shape).float()
                dEdt_net = torch.zeros(x_all.shape).float()
            elif self.dtype == 'double':
                dSdt_net = torch.zeros(x_all.shape).double()
                dEdt_net = torch.zeros(x_all.shape).double()
        


            dEdt = dE @ ((dE @ L).squeeze() + (dS @ M).squeeze())
            dSdt = dS @ ((dE @ L).squeeze() + (dS @ M).squeeze())
            
            dEdt_net[0, :] = dEdt
            dSdt_net[0, :] = dSdt

            

            

        for snapshot in range(self.dim_t - 1):

            # Structure-Preserving Neural Network

            # x1_net = self.net(x)
            # print(x1_net.shape)
            #with torch.no_grad():
#             dE, M = self.net.netE(x)
#             dS, L = self.net.netS(x)
#             print(M)
            

            x1_net = self.net.integrator2(x)
            


            if  (self.sys_name == 'GC_SVD') or (self.sys_name == 'VC_SPNN_SVD'):
                x_net[snapshot + 1::self.dim_t, :] = x1_net
            else:
                x_net[snapshot + 1, :] = x1_net
                

            x = x1_net
            
#             min_value_col1 = torch.min(x[:, 0])
#             max_value_col1 = torch.max(x[:, 0])

#             min_value_col2 = torch.min(x[:, 1])
#             max_value_col2 = torch.max(x[:, 1])

#             min_value_col3 = torch.min(x[:, 2])
#             max_value_col3 = torch.max(x[:, 2])

#             min_value_col4 = torch.min(x[:, 3])
#             max_value_col4 = torch.max(x[:, 3])

#             print(min_value_col1)
#             print(max_value_col1)

#             print(min_value_col2)
#             print(max_value_col2)

#             print(min_value_col3)
#             print(max_value_col3)

#             print(min_value_col4)
#             print(max_value_col4)
            
                     
            
            
            #with torch.no_grad():

            dE, M = self.net.netE(x)
            #     print(dE.shape)
            # print(M.shape)
            dS, L = self.net.netS(x)
            
            dE = dE.detach()
            dS = dS.detach()
            M = M.detach()
            L = L.detach()

            
            if  (self.sys_name == 'GC_SVD') or (self.sys_name == 'VC_SPNN_SVD'):
                dE = dE.unsqueeze(1)

                dS = dS.unsqueeze(1)

                dEdt = torch.sum(dE.squeeze() * ((dE @ L) + (dS @ M)).squeeze(), 1)
                dSdt = torch.sum(dS.squeeze() * ((dE @ L) + (dS @ M)).squeeze(), 1)
                dEdt_net[snapshot+1::self.dim_t] = dEdt
                dSdt_net[snapshot+1::self.dim_t] = dSdt
            else:
                dEdt = dE @ ((dE @ L).squeeze() + (dS @ M).squeeze())
                dSdt = dS @ ((dE @ L).squeeze() + (dS @ M).squeeze())
                dEdt_net[snapshot + 1, :] = dEdt
                dSdt_net[snapshot + 1, :] = dSdt
        
        x_net = x_net.detach()
        
        x_gfinn = x_net

        
#         if self.sys_name == 'GC_SVD':
            
            
#             pid = 0
#             plot_name = '[GC_SVD] Test AE Latent Variables_'+self.AE_name
#             plot_latent_visco(x_gfinn[pid*self.dim_t:(pid+1)*self.dim_t], self.dataset.dt, plot_name, self.output_dir)

#             plot_name = '[GC_SVD] Test True AE Latent Variables_'+self.AE_name
#             plot_latent_visco(x_all[pid*self.dim_t:(pid+1)*self.dim_t], self.dataset.dt, plot_name, self.output_dir)

            
#             scale_factor = (1.85 - 0.15) / (self.max_value_col1 -self.min_value_col1)
#             shift_factor = 0.15 -self.min_value_col1 * scale_factor

#             x_gfinn[:,0] = (x_gfinn[:,0] - shift_factor) / scale_factor
#             #x_all[:,0] = (x_all[:,0] - shift_factor) / scale_factor


#             scale_factor = (4.4066+4.1807) / (self.max_value_col2 -self.min_value_col2)
#             shift_factor = -4.1807-self.min_value_col2* scale_factor   
# #             inverse_scale_factor = 1.0 / scale_factor
# #             inverse_shift_factor = -shift_factor / scale_factor
# #             x_gfinn[:,1] = (x_gfinn[:,1] - inverse_shift_factor) / inverse_scale_factor
# #             x_all[:,1] = (x_all[:,1] - inverse_shift_factor) / inverse_scale_factor
#             x_gfinn[:,1] = (x_gfinn[:,1] - shift_factor) / scale_factor
#             #x_all[:,1] = (x_all[:,1] - shift_factor) / scale_factor
            

                
#             scale_factor = (3.2 - 0.7) / (self.max_value_col3 -self.min_value_col3)
#             shift_factor = 0.7-self.min_value_col3* scale_factor
# #             inverse_scale_factor = 1.0 / scale_factor
# #             inverse_shift_factor = -shift_factor / scale_factor         
# #             x_gfinn[:,2] = (x_gfinn[:,2] - inverse_shift_factor) / inverse_scale_factor
# #             x_all[:,2] = (x_all[:,2] - inverse_shift_factor) / inverse_scale_factor
#             x_gfinn[:,2] = (x_gfinn[:,2] - shift_factor) / scale_factor
#             #x_all[:,2] = (x_all[:,2] - shift_factor) / scale_factor
            
            
#             scale_factor = (3.2 - 0.7) / (self.max_value_col4 -self.min_value_col4)
#             shift_factor = 0.7-self.min_value_col4* scale_factor 
# #             inverse_scale_factor = 1.0 / scale_factor
# #             inverse_shift_factor = -shift_factor / scale_factor
# #             x_gfinn[:,3] = (x_gfinn[:,3] - inverse_shift_factor) / inverse_scale_factor
# #             x_all[:,3] = (x_all[:,3] - inverse_shift_factor) / inverse_scale_factor
#             x_gfinn[:,3] = (x_gfinn[:,3] - shift_factor) / scale_factor
#             x_all[:,3] = (x_all[:,3] - shift_factor) / scale_factor
            

        z_gfinn = x_gfinn @ self.Ud.t()

        z_gfinn_all = x_net_all @ self.Ud.t()
        
        
        #check
        x_tt = self.z_gt_tt @ self.Ud
        x1_tt = self.z1_gt_tt @ self.Ud
        x1_sae_tt = self.net.integrator2(x_tt)
        
        z1_sae_tt = x1_sae_tt @ self.Ud.t()
        z1_tt = x1_tt @ self.Ud.t()
        
        

        # Load Ground Truth and Compute MSE
        #z_gt = self.z_gt
        print_mse(z_gfinn, z_gt, self.sys_name)
#         print(z_gfinn)
#         print(x_gfinn)
#         print(z_gt)

        print_mse(z1_sae_tt,z1_tt, self.sys_name)
        
        print(torch.norm(x_all-x_gfinn, p=2)/torch.norm(x_all, p=2))
        print_mse(z_sae, z_gt, self.sys_name)


        # Plot results
        if (self.save_plots):
            #plot_name = 'SPNN Full Integration (Latent)'
            #plot_latent(x_net, self.x_trunc, dEdt_net, dSdt_net, self.dt, plot_name, self.output_dir, self.sys_name)
            if (self.sys_name == 'GC_SVD') or (self.sys_name == 'VC_SPNN_SVD'):
                pid = 0
#                 print((pid+1)*self.dim_t)
                plot_name = 'Energy_Entropy_Derivatives_' +self.AE_name
                
                plot_latent(dEdt_net[pid*self.dim_t:(pid+1)*self.dim_t], dSdt_net[pid*self.dim_t:(pid+1)*self.dim_t], self.dt, plot_name, self.output_dir, self.sys_name)
                plot_name = 'Full Integration_'+self.AE_name
                #print(self.sys_name)
                plot_results(z_gfinn[pid*self.dim_t:(pid+1)*self.dim_t,:], z_gt[pid*self.dim_t:(pid+1)*self.dim_t,:], self.dt, plot_name, self.output_dir, self.sys_name)

                plot_name = 'AE Reduction Only_'+self.AE_name
                plot_results(z_sae[pid*self.dim_t:(pid+1)*self.dim_t,:], z_gt[pid*self.dim_t:(pid+1)*self.dim_t,:], self.dt, plot_name, self.output_dir, self.sys_name)
                
                plot_name = 'GFINNs 1 step'+self.AE_name
                plot_results(z1_sae_tt[pid*(self.dim_t-1):(pid+1)*(self.dim_t-1),:], z1_tt[pid*(self.dim_t-1):(pid+1)*(self.dim_t-1),:], self.dt, plot_name, self.output_dir, self.sys_name)

            
            else:
                plot_name = 'Energy_Entropy_Derivatives_' +self.AE_name
                plot_latent(dEdt_net, dSdt_net, self.dt, plot_name, self.output_dir, self.sys_name)

                plot_name = 'Full Integration_'+self.AE_name

    #             print(z_gfinn.shape)
    #             print(z_gt.shape)

                plot_results(z_gfinn, z_gt, self.dt, plot_name, self.output_dir, self.sys_name)

                plot_name = 'AE Reduction Only_'+self.AE_name
                plot_results(z_sae, z_gt, self.dt, plot_name, self.output_dir, self.sys_name)
            

            if self.sys_name == 'viscoelastic':
                # Plot latent variables
                if (self.save_plots == True):
                    plot_name = '[VC] Latent Variables_' + self.AE_name
                    plot_latent_visco(x_gfinn, self.dataset.dt, plot_name, self.output_dir)

            elif self.sys_name == '1DBurgers':

                # Plot latent variables
                if (self.save_plots == True):
                    plot_name = '[1DBurgers] Latent Variables_' + self.AE_name
                    plot_latent_visco(x_gfinn, self.dataset.dt, plot_name, self.output_dir)
                    
            elif self.sys_name == 'GC_SVD_concat':
                # Plot latent variables
                if (self.save_plots == True):
                    plot_name = '[GC_SVD_concat] Latent Variables_' + self.AE_name
                    plot_latent_visco(x_gfinn, self.dataset.dt, plot_name, self.output_dir)
                    
                
              
            elif self.sys_name == 'GC':
                # Plot latent variables
                if (self.save_plots == True):
                    plot_name = '[GC] Latent Variables_' + self.AE_name
                    plot_latent_visco(x_gfinn, self.dataset.dt, plot_name, self.output_dir)
            
            
            elif self.sys_name == 'GC_SVD':
                # Plot latent variables
                if (self.save_plots == True):
                    pid = 0
                    plot_name = '[GC_SVD] AE Latent Variables_'+self.AE_name
                    plot_latent_visco(x_gfinn[pid*self.dim_t:(pid+1)*self.dim_t], self.dataset.dt, plot_name, self.output_dir)
                    
                    plot_name = '[GC_SVD] True AE Latent Variables_'+self.AE_name
                    plot_latent_visco(x_all[pid*self.dim_t:(pid+1)*self.dim_t], self.dataset.dt, plot_name, self.output_dir)
                    
            elif self.sys_name == 'VC_SPNN_SVD':
                    pid = 0
                    plot_name = '[VC_SVD] AE Latent Variables_'+self.AE_name
                    plot_latent_visco(x_gfinn[pid*self.dim_t:(pid+1)*self.dim_t], self.dataset.dt, plot_name, self.output_dir)
                    
                    plot_name = '[VC_SVD] True AE Latent Variables_'+self.AE_name
                    plot_latent_visco(x_all[pid*self.dim_t:(pid+1)*self.dim_t], self.dataset.dt, plot_name, self.output_dir)



#             elif self.sys_name == 'rolling_tire':
#                 x_q, x_v, x_sigma = self.SAE.split_latent(x_gfinn)

#                 # Plot latent variables
#                 if (self.save_plots == True):
#                     plot_name = '[Rolling Tire] Latent Variables_' + self.AE_name
#                     plot_latent_tire(x_q, x_v, x_sigma, self.dataset.dt, plot_name, self.output_dir)

        print("\n[GFINNs Testing Finished]\n")
