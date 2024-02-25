"""
@author: jpzxshi & zen
"""
import os
import time
import numpy as np
import torch

from data import Data
from .nn import LossNN
from .utils import timing, cross_entropy_loss

#from utilities.plot_gfinns import plot_results, plot_latent

import torch
import torch.optim as optim
import numpy as np

#from model import SparseAutoEncoder, StackedSparseAutoEncoder
from dataset_sim import load_dataset, split_dataset
from utilities.plot import plot_latent_dynamics, plot_latent, plot_test_results
from utilities.utils import print_mse, truncate_latent
import matplotlib.pyplot as plt

from learner.utils import mse, wasserstein, div, grad
from tqdm import tqdm



class Brain_tLaSDI_SAE_sep:
    '''Runner based on torch.
    '''
    brain = None

    @classmethod
    def Init(cls, ROM_model,AE, net, data_type,x_trunc,latent_idx,latent_dim_max, dt, z_gt, sys_name, output_dir, save_plots, criterion, optimizer, lr,
             iterations, lbfgs_steps, AE_name,dset_dir,output_dir_AE,save_plots_AE,
             lambda_dx,lambda_dz,lambda_int,miles_lr = [30000],gamma_lr = 1e-1, path=None, load_path=None, batch_size=None,
             batch_size_test=None, weight_decay_GFINNs=0, print_every=1000, save=False, load = False,  callback=None, dtype='float',
             device='cpu',trunc_period=1):
        cls.brain = cls( ROM_model,AE, net,data_type,x_trunc,latent_idx,latent_dim_max, dt, z_gt, sys_name, output_dir, save_plots, criterion,
                         optimizer, lr, weight_decay_GFINNs, iterations, lbfgs_steps,AE_name,dset_dir,output_dir_AE,save_plots_AE,lambda_dx,lambda_dz,lambda_int, miles_lr, gamma_lr,path,load_path, batch_size,
                         batch_size_test, print_every, save, load, callback, dtype, device,trunc_period)

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

    def __init__(self,ROM_model, AE, net,data_type,x_trunc,latent_idx, latent_dim_max, dt,z_gt,sys_name, output_dir,save_plots, criterion, optimizer, lr, weight_decay_GFINNs, iterations, lbfgs_steps,AE_name,dset_dir,output_dir_AE,save_plots_AE,
             lambda_dx,lambda_dz,lambda_int,miles_lr, gamma_lr,path,load_path, batch_size,batch_size_test, print_every, save, load, callback, dtype, device,trunc_period):
        #self.data = data
        self.net = net
        self.sys_name = sys_name
        self.output_dir = output_dir
        self.save_plots = save_plots
        self.x_trunc = x_trunc.detach()
        self.latent_idx = latent_idx
        self.dt = dt
        self.z_gt = z_gt
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr
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
        self.weight_decay_GFINNs = weight_decay_GFINNs 
        self.ROM_model = ROM_model
        
        self.miles_lr = miles_lr
        self.gamma_lr = gamma_lr
        self.data_type = data_type
        self.latent_dim_max = latent_dim_max
        
        
        if self.load:
            path = './outputs/' + self.load_path
            loss_history_value= torch.load( path + '/loss_history_value.p')
            self.lr = loss_history_value['lr_final']

        else:    
            self.lr = lr

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self.save_plots_AE = save_plots_AE


        if self.load:
            path = './outputs/' + self.load_path
            self.SAE = torch.load( path + '/model_best_AE.pkl')
            self.net = torch.load( path + '/model_best.pkl')
        else:
            self.SAE = AE
            if self.device == 'gpu':
                self.SAE = self.SAE.to(torch.device('cuda'))

        for param in self.SAE.parameters():
            param.requires_grad = False
            
#         self.x_trunc.requires_grad = False



        print(sum(p.numel() for p in self.SAE.parameters() if p.requires_grad))
        print(sum(p.numel() for p in self.net.parameters() if p.requires_grad))




        # Dataset Parameters
        self.dset_dir = dset_dir
        self.dataset = load_dataset(self.sys_name, self.dset_dir,self.device,self.dtype)
        self.dt = self.dataset.dt
        self.dim_t = self.dataset.dim_t
        
        
        z = self.dataset.z

        z_norm = self.SAE.normalize(z)
        # Forward pass
        _, x = self.SAE(z_norm)

        self.x_trunc, self.latent_idx = truncate_latent(x)
        

        self.train_snaps, self.test_snaps = split_dataset(self.sys_name, self.dim_t-1,self.data_type)

        self.lambda_dx = lambda_dx
        self.lambda_dz = lambda_dz
        self.lambda_int = lambda_int

        self.loss_history = None
        self.encounter_nan = False
        self.best_model = None

        self.__optimizer = None
        self.__criterion = None

    @timing
    def run(self):
        self.__init_brain()
        print('Training...', flush=True)
        if self.load:
            path = './outputs/' + self.load_path
            loss_history_value= torch.load( path + '/loss_history_value.p')
            loss_history = loss_history_value['loss_history']

            i_loaded = loss_history[-1][0]

        else:
            loss_history = []

            i_loaded = 0

        z_gt_tr = self.dataset.z[self.train_snaps, :]
        z_gt_tt = self.dataset.z[self.test_snaps, :]
        dz_gt_tr = self.dataset.dz[self.train_snaps, :]


        dz_gt_tt = self.dataset.dz[self.test_snaps, :]


        z1_gt_tr = self.dataset.z[self.train_snaps+1, :]
        z1_gt_tt = self.dataset.z[self.test_snaps+1, :]

        z_gt_tr_norm = self.SAE.normalize(z_gt_tr)
        z_gt_tt_norm = self.SAE.normalize(z_gt_tt)

        z1_gt_tr_norm = self.SAE.normalize(z1_gt_tr)
        z1_gt_tt_norm = self.SAE.normalize(z1_gt_tt)

        z_gt_norm = self.SAE.normalize(self.dataset.z)

        dz_gt_tr_norm_tmp = self.SAE.normalize(dz_gt_tr)
        dz_gt_tt_norm_tmp = self.SAE.normalize(dz_gt_tt)
        
        self.z_data = Data(z_gt_tr_norm,z1_gt_tr_norm,z_gt_tt_norm,z1_gt_tt_norm)

        
        prev_lr = self.__optimizer.param_groups[0]['lr']
#         for i in range(self.iterations + 1):

#         print(self.x_trunc.requires_grad)
        
        X_tmp, y_tmp= self.x_trunc[self.train_snaps,:], self.x_trunc[self.train_snaps+1,:]
    
    
        for i in tqdm(range(self.iterations + 1)):

            z_gt_tr_norm,z1_gt_tr_norm, mask_tr = self.z_data.get_batch(self.batch_size)
            z_gt_tt_norm,z1_gt_tt_norm, mask_tt = self.z_data.get_batch_test(self.batch_size_test)
            
            dz_gt_tr_norm = dz_gt_tr_norm_tmp[mask_tr]
            dz_gt_tt_norm = dz_gt_tt_norm_tmp[mask_tt]
            
            X_train = X_tmp[mask_tr]
            y_train = y_tmp[mask_tr]
            
            loss_GFINNs = self.__criterion(self.net(X_train), y_train)

            if  (self.lambda_dz == 0 and self.lambda_dx == 0):
                
                loss_dx = torch.tensor(0, dtype=torch.float64)
                loss_dz = torch.tensor(0, dtype=torch.float64)
            else:

                #new part with JVP
                dx_train = self.net.f(X_train)
                

                _, dx_data_train, dz_train_dec, idx_trunc = self.SAE.JVP_SAE(z_gt_tr_norm, X_train, dz_gt_tr_norm, dx_train,  self.trunc_period, self.latent_idx,self.latent_dim_max,self.dtype,self.device)
                

                loss_dx = torch.mean((dx_train - dx_data_train) ** 2)

        
                
                loss_dz = torch.mean((dz_gt_tr_norm[:, idx_trunc] - dz_train_dec) ** 2)
                

            loss = self.lambda_int*loss_GFINNs+self.lambda_dx*loss_dx+self.lambda_dz*loss_dz



            #print(loss) #tensor(0.0008, grad_fn=<MseLossBackward0>)
            Loss_early = 1e-10


            if i == 0 or (i+i_loaded) % self.print_every == 0 or i == self.iterations:



                test_init = min(self.test_snaps)
                test_final = max(self.test_snaps)

                self.dim_t_tt = len(self.test_snaps)+1 #includes the last training snapshot


                x = self.x_trunc[test_init-1,:]
                x = torch.unsqueeze(x, 0)

                if self.dtype == 'float':
                    x_net_test = torch.zeros(self.dim_t_tt+1, x.shape[1]).float()

                elif self.dtype == 'double':
                    x_net_test = torch.zeros(self.dim_t_tt+1, x.shape[1]).double()

                x_net_test[0,:] = x


                if self.device == 'gpu':
                    x_net_test = x_net_test.to(torch.device('cuda'))


                for snapshot in range(self.dim_t_tt):


                    x1_net = self.net.integrator2(x)

                    x_net_test[snapshot + 1, :] = x1_net

                    x = x1_net


                if self.dtype == 'float':
                    x_gfinn_test = torch.zeros([self.dim_t_tt+1, self.latent_dim_max]).float()
#             x_net_all = torch.zeros(self.x_trunc.shape).float()

                elif self.dtype == 'double':
                    x_gfinn_test = torch.zeros([self.dim_t_tt+1, self.latent_dim_max]).double()

                if self.device == 'gpu':
                    x_gfinn_test = x_gfinn_test.to(torch.device('cuda'))

                x_gfinn_test[:, self.latent_idx] = x_net_test


                z_gfinn_norm = self.SAE.decode(x_gfinn_test)



                # Decode latent vector

                loss_test = torch.mean(torch.sqrt(torch.sum((self.dataset.z[test_init-1:test_final+2,:] - z_gfinn_norm) ** 2,1))/torch.sqrt(torch.sum((self.dataset.z[test_init-1:test_final+2,:]) ** 2,1)))


                # print('{:<9}a loss: %.4e{:<25}Test loss: %.4e{:<25}'.format(i, loss.item(), loss_test.item()), flush=True)
                print(' ADAM || It: %05d, Loss: %.4e, loss_GFINNs: %.4e, loss_dx: %.4e, loss_dz: %.4e, Test: %.4e' %
                      (i+i_loaded, loss.item(),loss_GFINNs.item(),loss_dx.item(),loss_dz.item(), loss_test.item()))
                
                if torch.any(torch.isnan(loss)):
                    self.encounter_nan = True
                    print('Encountering nan, stop training', flush=True)
                    return None
                if self.save:
                    if not os.path.exists('model'): os.mkdir('model')
                    if self.path == None:
                        torch.save(self.net, 'model/model{}.pkl'.format(i+i_loaded))
                        torch.save(self.SAE, 'model/AE_model{}.pkl'.format(i+i_loaded))
                    else:
                        if not os.path.isdir('model/' + self.path): os.makedirs('model/' + self.path)
                        torch.save(self.net, 'model/{}/model{}.pkl'.format(self.path, i+i_loaded))
                        torch.save(self.SAE, 'model/{}/AE_model{}.pkl'.format(self.path, i+i_loaded))
                if self.callback is not None:
                    output = self.callback(self.data, self.net)
                    loss_history.append([i+i_loaded, loss.item(), loss_test.item(), *output])
#                     loss_GFINNs_history.append([i, loss_GFINNs.item(), loss_GFINNs_test.item(), *output])
#                     loss_dx_history.append([i, loss_dx.item(), loss_dx_test.item(), *output])
#                     loss_dz_history.append([i, loss_dz.item(), loss_dz_test.item(), *output])
#                     loss_pred_history.append([i, loss.item(), loss_pred_test.item(), *output])

             #       loss_AE_GFINNs_history.append([i, loss_AE_GFINNs.item(), loss_AE_GFINNs_test.item(), *output])
                else:
                    loss_history.append([i+i_loaded, loss.item(), loss_test.item()])
#                     loss_GFINNs_history.append([i, loss_GFINNs.item(), loss_GFINNs_test.item()])
#                     loss_dx_history.append([i, loss_dx.item(), loss_dx_test.item()])
#                     loss_dz_history.append([i, loss_dz.item(), loss_dz_test.item()])
#                     loss_pred_history.append([i, loss.item(), loss_pred_test.item()])
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
                self.__optimizer.zero_grad()
                loss.backward(retain_graph=False)
                self.__optimizer.step()
                self.__scheduler.step()
                
        lr_final = self.__optimizer.param_groups[0]['lr']
        path = './outputs/' + self.path
        if not os.path.isdir(path): os.makedirs(path)
        torch.save({'loss_history':loss_history, 'lr_final':lr_final}, path + '/loss_history_value.p')

    
                
        self.loss_history = np.array(loss_history)

        return self.loss_history



    def restore(self):
        if self.loss_history is not None and self.save == True:
            best_loss_index = np.argmin(self.loss_history[:, 1])
            iteration = int(self.loss_history[best_loss_index, 0])
            loss_train = self.loss_history[best_loss_index, 1]
            loss_test = self.loss_history[best_loss_index, 2]

            print('BestADAM It: %05d, Loss: %.4e, Test: %.4e' %
                  (iteration, loss_train, loss_test))
            if self.path == None:
                self.best_model = torch.load('model/model{}.pkl'.format(iteration))
                self.best_model_AE = torch.load('model/AE_model{}.pkl'.format(iteration))
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
                X_train, y_train, _ = self.data.get_batch(None)

                X_test, y_test, _ = self.data.get_batch_test(None)

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
            plt.savefig(path + '/loss_'+self.AE_name+self.sys_name+'.png')
            p1.remove()
            p2.remove()

#             p3,=plt.plot(self.loss_GFINNs_history[:,0], self.loss_GFINNs_history[:,1],'-')
#             p4,=plt.plot(self.loss_GFINNs_history[:,0], self.loss_GFINNs_history[:,2],'--')
#             plt.legend(['train loss (GFINNs)', 'test loss (GFINNs)'])  # , '$\hat{u}$'])
#             plt.yscale('log')
#             plt.savefig(path + '/loss_GFINNs_'+self.AE_name+self.sys_name+'.png')
#             p3.remove()
#             p4.remove()

            
#             p13,=plt.plot(self.loss_history[:,0], self.loss_history[:,1],'-')
#             p14,=plt.plot(self.loss_GFINNs_history[:,0], self.loss_GFINNs_history[:,1],'-')
#             p15,=plt.plot(self.loss_pred_history[:,0], self.loss_pred_history[:,2],'o')
#             plt.legend(['$\mathcal{L}$','$\mathcal{L}_{int}$','rel. l2 error'], loc='best')  # , '$\hat{u}$'])
#             plt.yscale('log')
#             plt.savefig(path + '/loss_all_pred_'+self.AE_name+self.sys_name+'.png')
#             p13.remove()
#             p14.remove()
#             p15.remove()

            
#             p16,=plt.plot(self.loss_history[:,0], self.loss_history[:,1],'-')
#             p17,=plt.plot(self.loss_pred_history[:,0], self.loss_pred_history[:,2],'o')
#             plt.legend(['train loss', 'rel. l2 error'])  # , '$\hat{u}$'])
#             plt.yscale('log')
#             plt.savefig(path + '/loss_pred_'+self.AE_name+self.sys_name+'.png')
#             p16.remove()
#             p17.remove()
            
            p3, = plt.plot(self.loss_history[:,0], self.loss_history[:,2],'-')
            plt.legend(['rel. l2 error'], loc='best')
            plt.yscale('log')
            plt.savefig(path + '/test_error_'+self.AE_name+self.sys_name+'.png')
            p3.remove()

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
#             self.__optimizer = torch.optim.Adam(list(self.net.parameters())+list(self.SAE.parameters()), lr=self.lr, weight_decay=self.weight_decay_GFINNs)
            params = [
                {'params': self.net.parameters(), 'lr': self.lr, 'weight_decay': self.weight_decay_GFINNs},
#                 {'params': self.SAE.parameters(), 'lr': self.lr_AE, 'weight_decay': self.weight_decay_AE}
            ]

            self.__optimizer = torch.optim.AdamW(params)
            if self.sys_name == 'rolling_tire':
                self.__scheduler = torch.optim.lr_scheduler.MultiStepLR(self.__optimizer, milestones=self.miles_lr,gamma=self.gamma_lr)
            else:
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

    ##frim spnn
    def test(self):
        print("\n[GFNN Testing Started]\n")


        self.net = self.best_model
        self.SAE = self.best_model_AE

        self.dim_t = self.z_gt.shape[0]



        z_gt_norm = self.SAE.normalize(self.z_gt)

        # Forward pass
        z_sae_norm, _ = self.SAE(z_gt_norm)

        z_sae = self.SAE.denormalize(z_sae_norm)


#         _, x = self.SAE(z)
        
        x = self.x_trunc[0,:]


        if self.dtype == 'float':
            x_net = torch.zeros(self.x_trunc.shape).float()
#             x_net_all = torch.zeros(self.x_trunc.shape).float()

        elif self.dtype == 'double':
            x_net = torch.zeros(self.x_trunc.shape).double()
#             x_net_all = torch.zeros(self.x_trunc.shape).double()
        if self.device == 'gpu':
            x_net = x_net.to(torch.device('cuda'))
#             x_net_all = x_net_all.to(torch.device('cuda'))
            

        x_net[0,:] = x



        if self.dtype == 'float':
            dSdt_net = torch.zeros(self.x_trunc.shape).float()
            dEdt_net = torch.zeros(self.x_trunc.shape).float()
        elif self.dtype == 'double':
            dSdt_net = torch.zeros(self.x_trunc.shape).double()
            dEdt_net = torch.zeros(self.x_trunc.shape).double()
            

        dE, M = self.net.netE(x)
        #     print(dE.shape)
        # print(M.shape)
        dS, L = self.net.netS(x)

        dEdt = dE @ ((dE @ L).squeeze() + (dS @ M).squeeze())
        dSdt = dS @ ((dE @ L).squeeze() + (dS @ M).squeeze())
        #
        # print(dE.shape)
        # print(((dE @ L).squeeze() + (dS @ M).squeeze()).shape)
        #
        # dEdt = dE @ M
        # dSdt = dS @ L

        dEdt_net[0, :] = dEdt
        dSdt_net[0, :] = dSdt

        # print(x_net.shape)

        for snapshot in range(self.dim_t - 1):

            x1_net = self.net.integrator2(self.net(x))

            x_net[snapshot + 1, :] = x1_net

            x = x1_net

            dE, M = self.net.netE(x)

            dS, L = self.net.netS(x)

            dEdt = dE @ ((dE @ L).squeeze() + (dS @ M).squeeze())
            dSdt = dS @ ((dE @ L).squeeze() + (dS @ M).squeeze())


            dEdt_net[snapshot + 1, :] = dEdt
            dSdt_net[snapshot + 1, :] = dSdt


        # Decode latent vector
        
        if self.dtype == 'float':
            x_gfinn = torch.zeros([self.dim_t, self.latent_dim_max]).float()
#             x_net_all = torch.zeros(self.x_trunc.shape).float()

        elif self.dtype == 'double':
            x_gfinn = torch.zeros([self.dim_t, self.latent_dim_max]).double()
            
        if self.device == 'gpu':
            x_gfinn = x_gfinn.to(torch.device('cuda'))
        
        x_gfinn[:, self.latent_idx] = x_net
             
            
        z_gfinn_norm = self.SAE.decode(x_gfinn)
        z_gfinn = self.SAE.denormalize(z_gfinn_norm)

        self.dim_t = self.z_gt.shape[0]
        self.dim_t_tt = len(self.test_snaps)+1 #includes the last training snapshot
    
        self.dim_t_tr = len(self.train_snaps)
                
        test_init = min(self.test_snaps)
        test_final = max(self.test_snaps)

        
        z_gt_norm = self.SAE.normalize(self.z_gt)
        


        print('Current GPU memory allocated after eval: ', torch.cuda.memory_allocated() / 1024 ** 3, 'GB')

        
        
        x = self.x_trunc[test_init-1,:]
#         print(x.shape)
#         print(self.x_trunc.shape)
        x = torch.unsqueeze(x, 0)
    
    
        if self.dtype == 'float':
            x_net_test = torch.zeros(self.dim_t_tt+1,x.shape[1]).float()
#             x_net_all = torch.zeros(self.x_trunc.shape).float()

        elif self.dtype == 'double':
            x_net_test = torch.zeros(self.dim_t_tt+1,x.shape[1]).double()
#             x_net_all = torch.zeros(self.x_trunc.shape).double()
        if self.device == 'gpu':
            x_net_test = x_net_test.to(torch.device('cuda'))
        
        x_net_test[0,:] = x


        for snapshot in range(self.dim_t_tt):


            x1_net = self.net.integrator2(x)

            x_net_test[snapshot + 1, :] = x1_net

            x = x1_net

        
        if self.dtype == 'float':
            x_gfinn_test = torch.zeros(self.dim_t_tt+1, self.latent_dim_max).float()
        elif self.dtype == 'double':
            x_gfinn_test = torch.zeros(self.dim_t_tt+1, self.latent_dim_max).double()

        
        if self.device == 'gpu':
            x_gfinn_test = x_gfinn_test.to(torch.device('cuda'))


#         print(self.dim_t_tt+1)
        x_gfinn_test[:,self.latent_idx] = x_net_test

        # Decode latent vector
        z_gfinn_test_norm = self.SAE.decode(x_gfinn_test)
        z_gfinn_test = self.SAE.denormalize(z_gfinn_test_norm)
        
        

        # Load Ground Truth and Compute MSE
        z_gt = self.z_gt
        
                
        
        print('prediction from last training snap')
        

        print_mse(z_gfinn_test, z_gt[test_init-1:test_final+2,:], self.sys_name)

        
        print('prediction error only for testing part')
        print_mse(z_gfinn[self.test_snaps,:], z_gt[self.test_snaps,:], self.sys_name)
     
        print('prediction error in entire time domain')
        print_mse(z_gfinn, z_gt, self.sys_name)
        

        
        print('prediction error only for AE part for test data')
        print_mse(z_sae[self.test_snaps,:], z_gt[self.test_snaps,:], self.sys_name)
        
        test_ratio = len(self.test_snaps)/self.z_gt.shape[0]
        
        path = './outputs/' + self.sys_name
        torch.save({'z_gfinn_test':z_gfinn_test, 'z_gt':z_gt}, path + '/TA_ROM_GT.p')

        # Plot results
        if (self.save_plots):
            #plot_name = 'SPNN Full Integration (Latent)'
            #plot_latent(x_net, self.x_trunc, dEdt_net, dSdt_net, self.dt, plot_name, self.output_dir, self.sys_name)
            plot_name = 'Energy_Entropy_Derivatives_' +self.AE_name
            plot_latent(dEdt_net, dSdt_net, self.dt, plot_name, self.output_dir, self.sys_name)

            #only valid for missing interval cases
            plot_name = 'GFINNs prediction_test'+self.AE_name
            plot_test_results(z_gfinn_test[1:,:], z_gt[test_init:test_final+2,:], self.dt, plot_name, self.output_dir, test_final,self.dim_t_tt,self.sys_name,self.ROM_model)



            if self.sys_name == 'viscoelastic':
                # Plot latent variables
                if (self.save_plots == True):
                    plot_name = '[VC] Latent Variables_' + self.AE_name
                    plot_latent_dynamics(x_gfinn, self.dataset.dt, plot_name, self.output_dir)
                    
            elif self.sys_name == 'GC':
                # Plot latent variables
                if (self.save_plots == True):
                    plot_name = '[GC] Latent Variables_' + self.AE_name
                    plot_latent_dynamics(x_gfinn, self.dataset.dt, plot_name, self.output_dir)



        print("\n[GFINNs Testing Finished]\n")
