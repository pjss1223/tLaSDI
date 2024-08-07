"""
@author: jpzxshi & zen
"""
import os
import time

from data import Data
from .nn import LossNN
from .utils import timing, cross_entropy_loss

import torch
import numpy as np
from tqdm import tqdm

from model import AutoEncoder
from dataset_sim import load_dataset, split_dataset
from utilities.plot import plot_test_results
from utilities.utils import print_mse
import matplotlib.pyplot as plt

class Brain_tLaSDI:
    '''Runner based on torch.
    '''
    brain = None

    @classmethod
    def Init(cls, ROM_model, net, sys_name, output_dir, save_plots, criterion, optimizer, lr,
             iterations, AE_name,dset_dir,output_dir_AE,layer_vec_AE,
             activation_AE,lr_AE,lambda_r_AE,lambda_jac_AE,lambda_dx,lambda_dz, lr_scheduler_type = 'StepLR', miles_lr=90000,gamma_lr=0.1, path=None, load_path=None, batch_size=None,
             weight_decay_AE = 0, weight_decay_GFINNs = 0, print_every=1000, save=False, load = False,  callback=None, dtype='double',
             device='cpu',trunc_period=1):
        cls.brain = cls( ROM_model,net, sys_name, output_dir, save_plots, criterion,
                         optimizer, lr, iterations,AE_name,dset_dir,output_dir_AE,layer_vec_AE,
                        activation_AE,lr_AE,lambda_r_AE,lambda_jac_AE,lambda_dx,lambda_dz,lr_scheduler_type, miles_lr,gamma_lr, path,load_path, batch_size, weight_decay_AE, weight_decay_GFINNs, print_every, save, load, callback, dtype, device,trunc_period)

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

    def __init__(self,ROM_model, net, sys_name, output_dir,save_plots, criterion, optimizer, lr, iterations, AE_name,dset_dir,output_dir_AE,layer_vec_AE,
             activation_AE,lr_AE,lambda_r_AE,lambda_jac_AE,lambda_dx,lambda_dz,lr_scheduler_type, miles_lr,gamma_lr, path,load_path, batch_size,
             weight_decay_AE, weight_decay_GFINNs, print_every, save, load, callback, dtype, device,trunc_period):
        self.net = net
        self.sys_name = sys_name
        self.output_dir = output_dir
        self.save_plots = save_plots
        self.criterion = criterion
        self.optimizer = optimizer
        self.ROM_model = ROM_model
        
        self.weight_decay_GFINNs = weight_decay_GFINNs
        self.weight_decay_AE = weight_decay_AE
        self.iterations = iterations
        self.path = path
        self.load_path = load_path
        self.batch_size = batch_size
        self.print_every = print_every
        
        self.lr_scheduler_type = lr_scheduler_type
        
        self.save = save
        self.load = load
        self.callback = callback
        
        self.dtype = dtype
        self.device = device
        self.dtype_torch = torch.float32 if dtype == 'float' else torch.float64 
        self.device_torch = torch.device("cuda") if device == 'gpu' else torch.device("cpu")

        self.AE_name = AE_name
        self.output_dir_AE = output_dir_AE
        self.trunc_period = trunc_period
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

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        if self.load:
            path = './outputs/' + self.load_path
            self.AE = torch.load( path + '/model_best_AE.pkl')
            self.net = torch.load( path + '/model_best.pkl')
        else:
            self.AE = AutoEncoder(layer_vec_AE, activation_AE).to(dtype=self.dtype_torch, device=self.device_torch)

        # print(sum(p.numel() for p in self.AE.parameters() if p.requires_grad))
        # print(sum(p.numel() for p in self.net.parameters() if p.requires_grad))
        #
        # Dataset Parameters
        self.dset_dir = dset_dir
        self.dataset = load_dataset(self.sys_name, self.dset_dir, self.device, self.dtype)
        self.dt = self.dataset.dt
        self.dim_t = self.dataset.dim_t
        self.z_gt = self.dataset.z

        self.train_snaps, self.test_snaps = split_dataset(self.sys_name, self.dim_t-1)

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
        
        if self.load:
            path = './outputs/' + self.load_path
            loss_history_value= torch.load( path + '/loss_history_value.p')
            loss_history = loss_history_value['loss_history']
            loss_GFINNs_history = loss_history_value['loss_GFINNs_history']
            loss_AE_recon_history = loss_history_value['loss_AE_recon_history']
            loss_AE_jac_history = loss_history_value['loss_AE_jac_history']
            loss_dx_history = loss_history_value['loss_dx_history']
            loss_dz_history = loss_history_value['loss_dz_history']
            elapsed_time =  loss_history_value['elapsed_time']
            i_loaded = loss_history[-1][0]
            loaded_time = elapsed_time[-1][0]

        else:
            loss_history = []
            loss_GFINNs_history = []
            loss_AE_recon_history = []
            loss_AE_jac_history = []
            loss_dx_history = []
            loss_dz_history = []
            elapsed_time =[]
            i_loaded = 0
            loaded_time = 0

        z_gt_tr = self.dataset.z[self.train_snaps, :]
        z_gt_tt = self.dataset.z[self.test_snaps, :]
        dz_gt_tr = self.dataset.dz[self.train_snaps, :]

        z1_gt_tr = self.dataset.z[self.train_snaps+1, :]
        z1_gt_tt = self.dataset.z[self.test_snaps+1, :]

        z_gt_tr_norm = self.AE.normalize(z_gt_tr)
        z_gt_tt_norm = self.AE.normalize(z_gt_tt)

        z1_gt_tr_norm = self.AE.normalize(z1_gt_tr)
        z1_gt_tt_norm = self.AE.normalize(z1_gt_tt)

        dz_gt_tr_norm_tmp = self.AE.normalize(dz_gt_tr)

        self.z_data = Data(z_gt_tr_norm,z1_gt_tr_norm,z_gt_tt_norm,z1_gt_tt_norm,self.device)

        self.dataset.dz = None

        prev_lr = self.__optimizer.param_groups[0]['lr']

        best_loss = float('inf')  # Initialize the best loss as infinity
        best_model = None
        best_model_AE = None

        start_time = time.time()
            
        for i in tqdm(range(self.iterations + 1)):

            z_gt_tr_norm,z1_gt_tr_norm, mask_tr = self.z_data.get_batch(self.batch_size)
            
            dz_gt_tr_norm = dz_gt_tr_norm_tmp[mask_tr]

            # regular in terms of training data
            z_sae_tr_norm, X_train = self.AE(z_gt_tr_norm)
            loss_AE_recon = torch.mean((z_sae_tr_norm - z_gt_tr_norm) ** 2)
            
            _, y_train = self.AE(z1_gt_tr_norm)

            loss_GFINNs = self.__criterion(self.net(X_train), y_train)

            if  ((self.lambda_jac == 0 and self.lambda_dx == 0) and self.lambda_dz == 0): 
                loss_AE_jac = torch.tensor(0, dtype=torch.float64)
                loss_dx = torch.tensor(0, dtype=torch.float64)
                loss_dz = torch.tensor(0, dtype=torch.float64)
                
            else:
                dx_train = self.net.f(X_train)

                dz_train, dx_data_train, dz_train_dec , idx_trunc = self.AE.JVP(z_gt_tr_norm, X_train, dz_gt_tr_norm, dx_train,  self.trunc_period)

                loss_dx = torch.mean((dx_train - dx_data_train) ** 2)
        
                loss_AE_jac =  torch.mean((dz_train - dz_gt_tr_norm[:,idx_trunc]) ** 2)
                
                loss_dz = torch.mean((dz_gt_tr_norm[:, idx_trunc] - dz_train_dec) ** 2)

            loss = loss_GFINNs+self.lambda_r*loss_AE_recon+self.lambda_dx*loss_dx+self.lambda_dz*loss_dz+self.lambda_jac*loss_AE_jac
            
            Loss_early = 1e-10

            if i == 0 or (i+i_loaded) % self.print_every == 0 or i == self.iterations:
                #prediction loss
                test_init = min(self.test_snaps)
                test_final = max(self.test_snaps)

                self.dim_t_tt = len(self.test_snaps)+1 #includes the last training snapshot

                z_gt_norm = self.AE.normalize(self.z_gt)

                z = z_gt_norm[test_init-1, :]

                z = torch.unsqueeze(z, 0)

                _, x = self.AE(z)

                if self.dtype == 'float':
                    x_tlasdi_test = torch.zeros(self.dim_t_tt+1, x.shape[1]).float()

                elif self.dtype == 'double':
                    x_tlasdi_test = torch.zeros(self.dim_t_tt+1, x.shape[1]).double()

                x_tlasdi_test[0,:] = x

                if self.device == 'gpu':
                    x_tlasdi_test = x_tlasdi_test.to(torch.device('cuda'))

                for snapshot in range(self.dim_t_tt):

                    x1_net = self.net.integrator2(x)

                    x_tlasdi_test[snapshot + 1, :] = x1_net

                    x = x1_net

                # Decode latent vector
                z_tlasdi_norm = self.AE.decode(x_tlasdi_test)

                loss_test = torch.mean(torch.sqrt(torch.sum((self.z_gt[test_init-1:test_final+2,:] - z_tlasdi_norm) ** 2,1))/torch.sqrt(torch.sum((self.z_gt[test_init-1:test_final+2,:]) ** 2,1)))

                current_time = time.time()
                elapsed_time_tmp = current_time - start_time  + loaded_time  

                elapsed_time.append([elapsed_time_tmp])            

                print(' ADAM || It: %05d, Loss: %.4e, loss_GFINNs: %.4e, loss_AE_recon: %.4e, loss_jac: %.4e, loss_dx: %.4e, loss_dz: %.4e, Test: %.4e' %
                  (i+i_loaded, loss.item(),loss_GFINNs.item(),loss_AE_recon.item(),loss_AE_jac.item(),loss_dx.item(),loss_dz.item(),loss_test.item()))

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_model = self.net
                    best_model_AE = self.AE

                if torch.any(torch.isnan(loss)):
                    self.encounter_nan = True
                    print('Encountering nan, stop training', flush=True)
                    return None

                if self.callback is not None:
                    output = self.callback(self.data, self.net)

                    loss_history.append([i+i_loaded, loss.item(), loss_test.item(), *output])

                    loss_GFINNs_history.append([i+i_loaded, loss_GFINNs.item(), *output])
                    loss_AE_recon_history.append([i+i_loaded, loss_AE_recon.item(), *output])
                    loss_AE_jac_history.append([i+i_loaded, loss_AE_jac.item(), *output])
                    loss_dx_history.append([i+i_loaded, loss_dx.item(), *output])
                    loss_dz_history.append([i+i_loaded, loss_dz.item(), *output])
                else:

                    loss_history.append([i+i_loaded, loss.item(), loss_test.item()])

                    loss_GFINNs_history.append([i+i_loaded, loss_GFINNs.item()])
                    loss_AE_recon_history.append([i+i_loaded, loss_AE_recon.item()])
                    loss_AE_jac_history.append([i+i_loaded, loss_AE_jac.item()])
                    loss_dx_history.append([i+i_loaded, loss_dx.item()])
                    loss_dz_history.append([i+i_loaded, loss_dz.item()])

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
            
            if i < self.iterations:
                self.__optimizer.zero_grad()
                loss.backward(retain_graph=False)
                self.__optimizer.step()

                if current_lr > 1e-5:
                    self.__scheduler.step()
        
        lr_final = self.__optimizer.param_groups[0]['lr']
        lr_AE_final = self.__optimizer.param_groups[1]['lr']
        path = './outputs/' + self.path
        if not os.path.isdir(path): os.makedirs(path)
        torch.save({'loss_history':loss_history, 'loss_GFINNs_history':loss_GFINNs_history,'loss_AE_recon_history':loss_AE_recon_history,'loss_AE_jac_history':loss_AE_jac_history,'loss_dx_history':loss_dx_history,'loss_dz_history':loss_dz_history, 'lr_final':lr_final,'lr_AE_final':lr_AE_final,'elapsed_time':elapsed_time,'optimizer_state_dict': self.__optimizer.state_dict()}, path + '/loss_history_value.p')

        self.loss_history = np.array(loss_history)
        self.loss_GFINNs_history = np.array(loss_GFINNs_history)
        self.loss_AE_recon_history = np.array(loss_AE_recon_history)
        self.loss_AE_jac_history = np.array(loss_AE_jac_history)
        self.loss_dx_history = np.array(loss_dx_history)
        self.loss_dz_history = np.array(loss_dz_history)
        self.elapsed_time = np.array(elapsed_time)

        self.best_model = best_model
        self.best_model_AE = best_model_AE

        self.dataset.z = None
        
        return self.loss_history, self.loss_GFINNs_history, self.loss_AE_recon_history, self.loss_dx_history, self.loss_AE_jac_history

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
        if loss_history:

            p1,=plt.plot(self.loss_history[:,0], self.loss_history[:,1],'-')
            p2,=plt.plot(self.loss_GFINNs_history[:,0], self.loss_GFINNs_history[:,1],'-')
            p3,=plt.plot(self.loss_AE_recon_history[:,0], self.loss_AE_recon_history[:,1],'-')
            p4,=plt.plot(self.loss_AE_jac_history[:,0], self.loss_AE_jac_history[:,1],'-')
            p5,=plt.plot(self.loss_dx_history[:,0], self.loss_dx_history[:,1],'-')
            p6,=plt.plot(self.loss_dz_history[:,0], self.loss_dz_history[:,1],'-')
            p7,=plt.plot(self.loss_history[:,0], self.loss_history[:,2],'o')
            plt.legend(['$\mathcal{L}$','$\mathcal{L}_{int}$','$\mathcal{L}_{rec}$','$\mathcal{L}_{jac}$','$\mathcal{L}_{con}$', '$\mathcal{L}_{approx}$','rel. l2 error'], loc='best',ncol=3)  # , '$\hat{u}$'])
            plt.yscale('log')
            plt.savefig(path + '/loss_all_pred_'+self.AE_name+self.sys_name+'.png')
            p1.remove()
            p2.remove()
            p3.remove()
            p4.remove()
            p5.remove()
            p6.remove()
            p7.remove()
            
            p8, = plt.plot(self.loss_history[:,0], self.loss_history[:,2],'-')
            plt.legend(['rel. l2 error'], loc='best')
            plt.yscale('log')
            plt.savefig(path + '/test_error_'+self.AE_name+self.sys_name+'.png')
            p8.remove()

            p9, = plt.plot(self.elapsed_time, self.loss_history[:,2],'-')
            plt.legend(['rel. l2 error'], loc='best')
            plt.yscale('log')
            plt.xscale('log')
            plt.savefig(path + '/test_error_wall_time'+self.AE_name+self.sys_name+'.png')
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
        self.net.device = self.device
        self.net.dtype = self.dtype
        self.__init_optimizer()
        self.__init_criterion()

    def __init_optimizer(self):
        
        if self.optimizer == 'adam':
            params = [
                {'params': self.net.parameters(), 'lr': self.lr, 'weight_decay': self.weight_decay_GFINNs},
                {'params': self.AE.parameters(), 'lr': self.lr_AE, 'weight_decay': self.weight_decay_AE}
            ]

            self.__optimizer = torch.optim.AdamW(params)
            
            if self.load:
                path = './outputs/' + self.load_path
                loss_history_value= torch.load( path + '/loss_history_value.p')
                
                self.__optimizer.load_state_dict(loss_history_value['optimizer_state_dict'])
            

            if self.lr_scheduler_type == 'StepLR': 
                self.__scheduler = torch.optim.lr_scheduler.StepLR(self.__optimizer, step_size=self.miles_lr, gamma=self.gamma_lr)
            elif self.lr_scheduler_type == 'MultiStepLR':
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

    def test(self):
        print("\n[tLaSDI Testing Started]\n")

        self.net = self.best_model
        self.AE = self.best_model_AE

        self.dim_t = self.z_gt.shape[0]

        # Forward pass

        self.dim_t_tt = len(self.test_snaps)+1
    
        self.dim_t_tr = len(self.train_snaps)
                
        test_init = min(self.test_snaps)
        test_final = max(self.test_snaps)

        z_gt_norm = self.AE.normalize(self.z_gt)

        z = z_gt_norm[test_init-1, :]

        z = torch.unsqueeze(z, 0)

        _, x = self.AE(z)

        x_tlasdi_test = torch.zeros(self.dim_t_tt+1, x.shape[1]).to(dtype=self.dtype_torch, device=self.device_torch)

        x_tlasdi_test[0,:] = x

        for snapshot in range(self.dim_t_tt):

            x1_net = self.net.integrator2(x)

            x_tlasdi_test[snapshot + 1, :] = x1_net

            x = x1_net

        # Decode latent vector
        z_tlasdi_test_norm = self.AE.decode(x_tlasdi_test)
        z_tlasdi_test = self.AE.denormalize(z_tlasdi_test_norm)

        print('prediction from last training snap')

        print_mse(z_tlasdi_test, self.z_gt[test_init-1:test_final+2,:], self.sys_name)

        # Plot results
        if (self.save_plots):

            plot_name = 'tLaSDI prediction_test'+self.AE_name
            plot_test_results(z_tlasdi_test[1:,:], self.z_gt[test_init:test_final+2,:], self.dt, plot_name, self.output_dir, test_final,self.dim_t_tt,self.sys_name,self.ROM_model)

        print("\n[tLaSDI Testing Finished]\n")
