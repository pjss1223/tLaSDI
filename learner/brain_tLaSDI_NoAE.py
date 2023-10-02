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



class Brain_tLaSDI_NoAE:
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
        self.train_traj, self.test_traj = split_dataset(self.sys_name, self.dataset.z.shape[0]) # valid only for GC_SVD

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



        if self.sys_name == "GC_SVD" or self.sys_name == "VC_SPNN_SVD":
            

            x_gt = self.dataset.z.reshape([-1,self.dataset.dim_z])
            


            
            x_gt_traj = self.dataset.z


            
            x_gt_tr = x_gt_traj[self.train_traj,:-1,:]
            x_gt_tt = x_gt_traj[self.test_traj,:-1,:]
            

            
            x1_gt_tr = x_gt_traj[self.train_traj,1:,:]
            x1_gt_tt = x_gt_traj[self.test_traj,1:,:]
            

            
            x_gt_tr = x_gt_tr.reshape([-1,self.dataset.dim_z])
            x_gt_tt = x_gt_tt.reshape([-1,self.dataset.dim_z])
            
            
            

            x1_gt_tr = x1_gt_tr.reshape([-1,self.dataset.dim_z])
            x1_gt_tt = x1_gt_tt.reshape([-1,self.dataset.dim_z])
            

            self.x_gt = x_gt
            self.x_gt_tt_all = x_gt_traj[self.test_traj,:,:].reshape([-1,self.dataset.dim_z]) 
            
            
            
        else:
            z_gt_tr = self.dataset.z[self.train_snaps, :]
            z_gt_tt = self.dataset.z[self.test_snaps, :]
            dz_gt_tr = self.dataset.dz[self.train_snaps, :]



            dz_gt_tt = self.dataset.dz[self.test_snaps, :]



            z1_gt_tr = self.dataset.z[self.train_snaps+1, :]
            z1_gt_tt = self.dataset.z[self.test_snaps+1, :]

            z_gt = self.dataset.z
            
            
        
        self.x_data = Data(x_gt_tr,x1_gt_tr,x_gt_tt,x1_gt_tt,self.device)
        
        
        
        self.dataset.dz = None
        

        
        prev_lr = self.__optimizer.param_groups[0]['lr']
        for i in range(self.iterations + 1):
                        
#             print(z_gt_tr_norm.shape)
#             print(z_gt_tr_norm.size(0))
                
        

            
            X_train,y_train, _ = self.x_data.get_batch(self.batch_size)
            
            
#             path = './data/'

#             torch.save(y_train,path + '/y_train_data_GC.p')
            
#             print(y_train)
            
#             print(X_train)
            
#             print(X_train)
#             print(y_train)
                    
#             dE, M = self.net.netE(X_train)
#             dS, L = self.net.netS(X_train)
#             dE = dE.unsqueeze(1)
        
#             dS = dS.unsqueeze(1)
#             print(dE.shape)
#             print(dS.shape)
#             print((dE @ M).squeeze().shape)
            
#             print((dE @ M).squeeze())
#             print((dS @ L).squeeze())
            #print((dS).squeeze())

            ### check degeneracy
#             dE, M = self.net.netE(X_train)
#             dS, L = self.net.netS(X_train)
#             dE = dE.unsqueeze(1)
#             dS = dS.unsqueeze(1)
#             dEM = dE @ M
#             dSL = dS @ L
            
#             print(M.shape)
#             print(L.shape)
        
#             print(dE.shape)
#             print(dEM.shape)
#             print('dEM')
#             print(dEM)
#             print('dSL')
#             print(dSL)


            loss_GFINNs = self.__criterion(X_train, y_train)
            



            loss = loss_GFINNs


            Loss_early = 1e-8
            
            


            if i % self.print_every == 0 or i == self.iterations:
                

                X_test,y_test, _ = self.x_data.get_batch_test(self.batch_size_test)
                
#                 print(x_gt_tt.shape) #8000 4

                
                
                
                #with torch.no_grad():
                loss_no_batch = self.__criterion2(x_gt_tr, x1_gt_tr)
                
                
                #with torch.no_grad():
                loss_test = self.__criterion(x_gt_tt, x1_gt_tt)


                # print('{:<9}a loss: %.4e{:<25}Test loss: %.4e{:<25}'.format(i, loss.item(), loss_test.item()), flush=True)
                print(' ADAM || It: %05d, Loss: %.4e,  Test loss: %.4e' %(i, loss.item(), loss_test.item()))
                if torch.any(torch.isnan(loss)):
                    self.encounter_nan = True
                    print('Encountering nan, stop training', flush=True)
                    return None
                if self.save:
                    if not os.path.exists('model'): os.mkdir('model')
                    if self.path == None:
                        torch.save(self.net, 'model/model{}.pkl'.format(i))
                    else:
                        if not os.path.isdir('model/' + self.path): os.makedirs('model/' + self.path)
                        torch.save(self.net, 'model/{}/model{}.pkl'.format(self.path, i))
                if self.callback is not None:
                    output = self.callback(self.data, self.net)
                    loss_history.append([i, loss_no_batch.item(), loss_test.item(), *output])
                else:
                    loss_history.append([i, loss_no_batch.item(), loss_test.item()])
               #     loss_AE_GFINNs_history.append([i, loss_AE_GFINNs.item(), loss_AE_GFINNs_test.item(), *output])
                if loss_no_batch <= Loss_early:
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
                loss.backward(retain_graph=False)
                #loss.backward()
                self.__optimizer.step()

                self.__scheduler.step()
                
        self.loss_history = np.array(loss_history)
        
        

        

        self.dataset.z = None
        
        return self.loss_history


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
            else:
                self.best_model = torch.load('model/{}/model{}.pkl'.format(self.path, iteration))
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
                X_train, y_train,_ = self.x_data.get_batch(self.batch_size)

                X_test, y_test, _ = self.x_data.get_batch_test(self.batch_size_test)

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
#         self.data.device = self.device
#         self.data.dtype = self.dtype
        self.net.device = self.device
        self.net.dtype = self.dtype
        self.__init_optimizer()
        self.__init_criterion()

    def __init_optimizer(self):
        print(self.net.parameters())
        if self.optimizer == 'adam':
            params = [
                {'params': self.net.parameters(), 'lr': self.lr, 'weight_decay': self.weight_decay_GFINNs},
            ]
            #self.__optimizer = torch.optim.Adam(list(self.net.parameters())+list(self.SAE.parameters()), lr=self.lr, weight_decay=self.weight_decay)
            self.__optimizer = torch.optim.Adam(params)
#             self.__scheduler = torch.optim.lr_scheduler.MultiStepLR(self.__optimizer, milestones=self.miles_lr,gamma=self.gamma_lr)
            self.__scheduler = torch.optim.lr_scheduler.StepLR(self.__optimizer, step_size=self.miles_lr, gamma=self.gamma_lr)
        else:
            raise NotImplementedError

    def __init_criterion(self):
        #print(self.net.criterion)
        if isinstance(self.net, LossNN):
            self.__criterion = self.net.criterion
            self.__criterion2 = self.net.criterion2
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

        x_gt = self.x_gt_tt_all

        x = x_gt[::self.dim_t, :]
        
        
        if self.dtype == 'float':
            x_net = torch.zeros(x_gt.shape).float()
            x_net_all = torch.zeros(x_gt.shape).float()

        elif self.dtype == 'double':
            x_net = torch.zeros(x_gt.shape).double()
            x_net_all = torch.zeros(x_gt.shape).double()

        
        
        if (self.sys_name == 'GC_SVD') or (self.sys_name == 'VC_SPNN_SVD'):
            x_net[::self.dim_t, :] = x
        else:
            x_net[0,:] = x

            
        #print(x_net.shape)
        if self.device == 'gpu':
            x_net = x_net.to(torch.device('cuda'))
            x_net_all = x_net_all.to(torch.device('cuda'))



#         #with torch.no_grad():
#         dE, M = self.net.netE(x)
#         #     print(dE.shape)
#         # print(M.shape)
#         dS, L = self.net.netS(x)
        
# #         print(x.shape)

        
        
#         if self.sys_name == 'GC_SVD':
#             if self.dtype == 'float':
#                 dSdt_net = torch.zeros(x_all.shape[0]).float()
#                 dEdt_net = torch.zeros(x_all.shape[0]).float()
#             elif self.dtype == 'double':
#                 dSdt_net = torch.zeros(x_all.shape[0]).double()
#                 dEdt_net = torch.zeros(x_all.shape[0]).double()
        

#             dE = dE.unsqueeze(1)
#             #print(dE.shape)
#             dS = dS.unsqueeze(1)
            
# #             print(L.shape)
# #             print(dE.shape)


#             dEdt = torch.sum(dE.squeeze()* ((dE @ L) + (dS @ M)).squeeze(),1)
#             dSdt = torch.sum(dS.squeeze()* ((dE @ L) + (dS @ M)).squeeze(),1)
            
# #             print(dEdt.shape)
# #             print(dEdt_net.shape)
            

            
#             dEdt_net[::self.dim_t] = dEdt
#             dSdt_net[::self.dim_t] = dSdt
        
#         else: 
            
#             if self.dtype == 'float':
#                 dSdt_net = torch.zeros(x_all.shape).float()
#                 dEdt_net = torch.zeros(x_all.shape).float()
#             elif self.dtype == 'double':
#                 dSdt_net = torch.zeros(x_all.shape).double()
#                 dEdt_net = torch.zeros(x_all.shape).double()
        


#             dEdt = dE @ ((dE @ L).squeeze() + (dS @ M).squeeze())
#             dSdt = dS @ ((dE @ L).squeeze() + (dS @ M).squeeze())
            
#             dEdt_net[0, :] = dEdt
#             dSdt_net[0, :] = dSdt



        for snapshot in range(self.dim_t - 1):
            
            x1_net = self.net.integrator2(self.net(x))

            if (self.sys_name == 'GC_SVD') or (self.sys_name == 'VC_SPNN_SVD'):
                x_net[snapshot + 1::self.dim_t, :] = x1_net
            else:
                x_net[snapshot + 1, :] = x1_net

            x = x1_net

            #with torch.no_grad():

#             dE, M = self.net.netE(x)
#             #     print(dE.shape)
#             # print(M.shape)
#             dS, L = self.net.netS(x)
            
#             dE = dE.detach()
#             dS = dS.detach()
#             M = M.detach()
#             L = L.detach()

            

            # dEdt = dE @ M
            # dSdt = dS @ L

            # print(dSdt.shape)
              
            
#             if self.sys_name == 'GC_SVD':
#                 dE = dE.unsqueeze(1)

#                 dS = dS.unsqueeze(1)

#                 dEdt = torch.sum(dE.squeeze() * ((dE @ L) + (dS @ M)).squeeze(), 1)
#                 dSdt = torch.sum(dS.squeeze() * ((dE @ L) + (dS @ M)).squeeze(), 1)
#                 dEdt_net[snapshot+1::self.dim_t] = dEdt
#                 dSdt_net[snapshot+1::self.dim_t] = dSdt
#             else:
#                 dEdt = dE @ ((dE @ L).squeeze() + (dS @ M).squeeze())
#                 dSdt = dS @ ((dE @ L).squeeze() + (dS @ M).squeeze())
#                 dEdt_net[snapshot + 1, :] = dEdt
#                 dSdt_net[snapshot + 1, :] = dSdt

        x_net = x_net.detach()
        
        x_gfinn = x_net
        


        print_mse(x_gfinn, x_gt, self.sys_name)

        if self.sys_name == 'GC_SVD':
            q_gt = x_gt[:,0]
            p_gt = x_gt[:,1]
            s1_gt = x_gt[:,2]
            s2_gt = x_gt[:,3]

            q_net = x_gfinn[:,0]
            p_net = x_gfinn[:,1]
            s1_net = x_gfinn[:,2]
            s2_net = x_gfinn[:,3]

            q_mse = torch.mean(torch.sqrt(torch.sum((q_gt - q_net) ** 2, 0) / torch.sum(q_gt ** 2, 0)))
            p_mse = torch.mean(torch.sqrt(torch.sum((p_gt - p_net) ** 2, 0) / torch.sum(p_gt ** 2, 0)))
            s1_mse = torch.mean(torch.sqrt(torch.sum((s1_gt - s1_net) ** 2, 0) / torch.sum(s1_gt ** 2, 0)))
            s2_mse = torch.mean(torch.sqrt(torch.sum((s2_gt - s2_net) ** 2, 0) / torch.sum(s2_gt ** 2, 0)))

            # Print MSE
            print('Position MSE = {:1.2e}'.format(q_mse))
            print('Momentum MSE = {:1.2e}'.format(p_mse))
            print('Entropy1 MSE = {:1.2e}'.format(s1_mse))
            print('Entropy2 MSE = {:1.2e}'.format(s2_mse))
        
        if self.sys_name == 'VC_SPNN_SVD':
            q1_gt = x_gt[:,0]
            q2_gt = x_gt[:,1]
            v_gt = x_gt[:,2]
            e_gt = x_gt[:,3]
            tau_gt = x_gt[:,4]

            q1_net = x_gfinn[:,0]
            q2_net = x_gfinn[:,1]
            v_net = x_gfinn[:,2]
            e_net = x_gfinn[:,3]
            tau_net = x_gfinn[:,4]

            q1_mse = torch.mean(torch.sqrt(torch.sum((q1_gt - q1_net) ** 2, 0) / torch.sum(q1_gt ** 2, 0)))
            q2_mse = torch.mean(torch.sqrt(torch.sum((q2_gt - q2_net) ** 2, 0) / torch.sum(q2_gt ** 2, 0)))
            v_mse = torch.mean(torch.sqrt(torch.sum((v_gt - v_net) ** 2, 0) / torch.sum(v_gt ** 2, 0)))
            e_mse = torch.mean(torch.sqrt(torch.sum((e_gt - e_net) ** 2, 0) / torch.sum(e_gt ** 2, 0)))
            tau_mse = torch.mean(torch.sqrt(torch.sum((tau_gt - tau_net) ** 2, 0) / torch.sum(tau_gt ** 2, 0)))
            
#             print(q1_net)
#             print(q1_gt)

            # Print MSE
            print('Position1 MSE = {:1.2e}'.format(q1_mse))
            print('Position2 MSE = {:1.2e}'.format(q2_mse))
            print('Velocity MSE = {:1.2e}'.format(v_mse))
            print('Energy MSE = {:1.2e}'.format(e_mse))
            print('Stress MSE = {:1.2e}'.format(tau_mse))



        # Plot results
        if (self.save_plots):
   
            if self.sys_name == 'GC_SVD':
                # Plot latent variables
                if (self.save_plots == True):
                    pid = 3
                    plot_name = '[GC_NoAE] AE Latent Variables_'+self.AE_name
                    plot_latent_visco(x_gfinn[pid*self.dim_t:(pid+1)*self.dim_t], self.dataset.dt, plot_name, self.output_dir)
                    
                    plot_name = '[GC_NoAE] True AE Latent Variables_'+self.AE_name
                    plot_latent_visco(x_gt[pid*self.dim_t:(pid+1)*self.dim_t], self.dataset.dt, plot_name, self.output_dir)
                    
                    
                    N = x_gt[pid*self.dim_t:(pid+1)*self.dim_t].shape[0]
                    t_vec = np.linspace(self.dataset.dt,N*self.dataset.dt,N)

                    fig, axes = plt.subplots(1,4, figsize=(20, 5))
                    ax1, ax2, ax3, ax4 = axes.flatten()
                    plot_name = '[GC_NoAE] Approximation results' +self.AE_name
                    fig.suptitle(plot_name)
                    
                    #print(q_net.shape) # 8020
                    ax1.plot(t_vec, q_net[pid*self.dim_t:(pid+1)*self.dim_t].detach().cpu(),'b')
                    ax1.plot(t_vec, q_gt[pid*self.dim_t:(pid+1)*self.dim_t].detach().cpu(),'k--')
                    l1, = ax1.plot([],[],'k--')
                    l2, = ax1.plot([],[],'b')
                    ax1.legend((l1, l2), ('GT','Net'))
                    ax1.set_ylabel('$q$ [-]')
                    ax1.set_xlabel('$t$ [s]')
                    ax1.grid()

                    ax2.plot(t_vec, p_net[pid*self.dim_t:(pid+1)*self.dim_t].detach().cpu(),'b')
                    ax2.plot(t_vec, p_gt[pid*self.dim_t:(pid+1)*self.dim_t].detach().cpu(),'k--')
                    l1, = ax2.plot([],[],'k--')
                    l2, = ax2.plot([],[],'b')
                    ax2.legend((l1, l2), ('GT','Net'))
                    ax2.set_ylabel('$p$ [-]')
                    ax2.set_xlabel('$t$ [s]')
                    ax2.grid()

                    ax3.plot(t_vec, s1_net[pid*self.dim_t:(pid+1)*self.dim_t].detach().cpu(),'b')
                    ax3.plot(t_vec, s1_gt[pid*self.dim_t:(pid+1)*self.dim_t].detach().cpu(),'k--')
                    l1, = ax3.plot([],[],'k--')
                    l2, = ax3.plot([],[],'b')
                    ax3.legend((l1, l2), ('GT','Net'))
                    ax3.set_ylabel('$S_1$ [-]')
                    ax3.set_xlabel('$t$ [s]')
                    ax3.grid()

                    ax4.plot(t_vec, s2_net[pid*self.dim_t:(pid+1)*self.dim_t].detach().cpu(),'b')
                    ax4.plot(t_vec, s2_gt[pid*self.dim_t:(pid+1)*self.dim_t].detach().cpu(),'k--')
                    l1, = ax4.plot([],[],'k--')
                    l2, = ax4.plot([],[],'b')
                    ax4.legend((l1, l2), ('GT','Net'))
                    ax4.set_ylabel('$S_2$ [-]')
                    ax4.set_xlabel('$t$ [s]')
                    ax4.grid()

                    save_dir = os.path.join(self.output_dir, plot_name)
                    
                    plt.savefig(save_dir)
                    plt.clf()
            if self.sys_name == 'VC_SPNN_SVD':
                # Plot latent variables
                if (self.save_plots == True):
                    
                    pid = 3
                    plot_name = '[VC_NoAE] AE Latent Variables_'+self.AE_name
                    plot_latent_visco(x_gfinn[pid*self.dim_t:(pid+1)*self.dim_t], self.dataset.dt, plot_name, self.output_dir)
                    
                    plot_name = '[VC_NoAE] True AE Latent Variables_'+self.AE_name
                    plot_latent_visco(x_gt[pid*self.dim_t:(pid+1)*self.dim_t], self.dataset.dt, plot_name, self.output_dir)
                    
                    
                    N = x_gt[pid*self.dim_t:(pid+1)*self.dim_t].shape[0]
                    t_vec = np.linspace(self.dataset.dt,N*self.dataset.dt,N)

                    fig, axes = plt.subplots(1,4, figsize=(20, 5))
                    ax1, ax2, ax3, ax4 = axes.flatten()
                    plot_name = '[VC_NoAE] Approximation results' +self.AE_name
                    fig.suptitle(plot_name)
                    
                    #print(q_net.shape) # 8020
                    pid = [0, 5, 9, 16]
                    ax1.plot(t_vec, q1_net[pid[0]*self.dim_t:(pid[0]+1)*self.dim_t].detach().cpu(),'b')
                    ax1.plot(t_vec, q1_net[pid[1]*self.dim_t:(pid[1]+1)*self.dim_t].detach().cpu(),'b')
                    ax1.plot(t_vec, q1_net[pid[2]*self.dim_t:(pid[2]+1)*self.dim_t].detach().cpu(),'b')
                    ax1.plot(t_vec, q1_net[pid[3]*self.dim_t:(pid[3]+1)*self.dim_t].detach().cpu(),'b')
                    
                    ax1.plot(t_vec, q1_gt[pid[0]*self.dim_t:(pid[0]+1)*self.dim_t].detach().cpu(),'k--')
                    ax1.plot(t_vec, q1_gt[pid[1]*self.dim_t:(pid[1]+1)*self.dim_t].detach().cpu(),'k--')
                    ax1.plot(t_vec, q1_gt[pid[2]*self.dim_t:(pid[2]+1)*self.dim_t].detach().cpu(),'k--')
                    ax1.plot(t_vec, q1_gt[pid[3]*self.dim_t:(pid[3]+1)*self.dim_t].detach().cpu(),'k--')
                    l1, = ax1.plot([],[],'k--')
                    l2, = ax1.plot([],[],'b')
                    ax1.legend((l1, l2), ('GT','Net'))
                    ax1.set_ylabel('$q1$ [-]')
                    ax1.set_xlabel('$t$ [s]')
                    ax1.grid()

#                     ax2.plot(t_vec, q2_net[pid*self.dim_t:(pid+1)*self.dim_t].detach().cpu(),'b')
#                     ax2.plot(t_vec, q2_gt[pid*self.dim_t:(pid+1)*self.dim_t].detach().cpu(),'k--')
#                     l1, = ax2.plot([],[],'k--')
#                     l2, = ax2.plot([],[],'b')
#                     ax2.legend((l1, l2), ('GT','Net'))
#                     ax2.set_ylabel('$q2$ [-]')
#                     ax2.set_xlabel('$t$ [s]')
#                     ax2.grid()
                    ax2.plot(t_vec, v_net[pid[0]*self.dim_t:(pid[0]+1)*self.dim_t].detach().cpu(),'b')
                    ax2.plot(t_vec, v_net[pid[1]*self.dim_t:(pid[1]+1)*self.dim_t].detach().cpu(),'b')
                    ax2.plot(t_vec, v_net[pid[2]*self.dim_t:(pid[2]+1)*self.dim_t].detach().cpu(),'b')
                    ax2.plot(t_vec, v_net[pid[3]*self.dim_t:(pid[3]+1)*self.dim_t].detach().cpu(),'b')
                    
                    ax2.plot(t_vec, v_gt[pid[0]*self.dim_t:(pid[0]+1)*self.dim_t].detach().cpu(),'k--')
                    ax2.plot(t_vec, v_gt[pid[1]*self.dim_t:(pid[1]+1)*self.dim_t].detach().cpu(),'k--')
                    ax2.plot(t_vec, v_gt[pid[2]*self.dim_t:(pid[2]+1)*self.dim_t].detach().cpu(),'k--')
                    ax2.plot(t_vec, v_gt[pid[3]*self.dim_t:(pid[3]+1)*self.dim_t].detach().cpu(),'k--')
                    l1, = ax2.plot([],[],'k--')
                    l2, = ax2.plot([],[],'b')
                    ax2.legend((l1, l2), ('GT','Net'))
                    ax2.set_ylabel('$v$ [-]')
                    ax2.set_xlabel('$t$ [s]')
                    ax2.grid()

                    ax3.plot(t_vec, e_net[pid[0]*self.dim_t:(pid[0]+1)*self.dim_t].detach().cpu(),'b')
                    ax3.plot(t_vec, e_net[pid[1]*self.dim_t:(pid[1]+1)*self.dim_t].detach().cpu(),'b')
                    ax3.plot(t_vec, e_net[pid[2]*self.dim_t:(pid[2]+1)*self.dim_t].detach().cpu(),'b')
                    ax3.plot(t_vec, e_net[pid[3]*self.dim_t:(pid[3]+1)*self.dim_t].detach().cpu(),'b')
                    
                    ax3.plot(t_vec, e_gt[pid[0]*self.dim_t:(pid[0]+1)*self.dim_t].detach().cpu(),'k--')
                    ax3.plot(t_vec, e_gt[pid[1]*self.dim_t:(pid[1]+1)*self.dim_t].detach().cpu(),'k--')
                    ax3.plot(t_vec, e_gt[pid[2]*self.dim_t:(pid[2]+1)*self.dim_t].detach().cpu(),'k--')
                    ax3.plot(t_vec, e_gt[pid[3]*self.dim_t:(pid[3]+1)*self.dim_t].detach().cpu(),'k--')
                    l1, = ax3.plot([],[],'k--')
                    l2, = ax3.plot([],[],'b')
                    ax3.legend((l1, l2), ('GT','Net'))
                    ax3.set_ylabel('$e$ [-]')
                    ax3.set_xlabel('$t$ [s]')
                    ax3.grid()
                    
                    ax4.plot(t_vec, tau_net[pid[0]*self.dim_t:(pid[0]+1)*self.dim_t].detach().cpu(),'b')
                    ax4.plot(t_vec, tau_net[pid[1]*self.dim_t:(pid[1]+1)*self.dim_t].detach().cpu(),'b')
                    ax4.plot(t_vec, tau_net[pid[2]*self.dim_t:(pid[2]+1)*self.dim_t].detach().cpu(),'b')
                    ax4.plot(t_vec, tau_net[pid[3]*self.dim_t:(pid[3]+1)*self.dim_t].detach().cpu(),'b')
                    
                    ax4.plot(t_vec, tau_gt[pid[0]*self.dim_t:(pid[0]+1)*self.dim_t].detach().cpu(),'k--')
                    ax4.plot(t_vec, tau_gt[pid[1]*self.dim_t:(pid[1]+1)*self.dim_t].detach().cpu(),'k--')
                    ax4.plot(t_vec, tau_gt[pid[2]*self.dim_t:(pid[2]+1)*self.dim_t].detach().cpu(),'k--')
                    ax4.plot(t_vec, tau_gt[pid[3]*self.dim_t:(pid[3]+1)*self.dim_t].detach().cpu(),'k--')
                    l1, = ax4.plot([],[],'k--')
                    l2, = ax4.plot([],[],'b')
                    ax4.legend((l1, l2), ('GT','Net'))
                    ax4.set_ylabel('$\\tau$ [-]')
                    ax4.set_xlabel('$t$ [s]')
                    ax4.grid()

                    save_dir = os.path.join(self.output_dir, plot_name)
                    
                    plt.savefig(save_dir)
                    plt.clf()





        print("\n[GFINNs Testing Finished]\n")
