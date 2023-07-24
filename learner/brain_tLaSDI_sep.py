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

#from model import SparseAutoEncoder, StackedSparseAutoEncoder
from dataset_sim import load_dataset, split_dataset
from utilities.plot import plot_results, plot_latent_visco, plot_latent_tire, plot_latent
from utilities.utils import print_mse, all_latent
import matplotlib.pyplot as plt

from learner.utils import mse, wasserstein, div, grad



class Brain_tLaSDI_sep:
    '''Runner based on torch.
    '''
    brain = None

    @classmethod
    def Init(cls, AE, net, dt, z_gt, sys_name, output_dir, save_plots, criterion, optimizer, lr,
             iterations, lbfgs_steps, AE_name,dset_dir,output_dir_AE,save_plots_AE,
             lambda_dx,lambda_dz,miles_lr = [30000],gamma_lr = 1e-1, path=None, load_path=None, batch_size=None,
             batch_size_test=None, weight_decay_GFINNs=0, print_every=1000, save=False, load = False,  callback=None, dtype='float',
             device='cpu',trunc_period=1):
        cls.brain = cls( AE, net, dt, z_gt, sys_name, output_dir, save_plots, criterion,
                         optimizer, lr, weight_decay_GFINNs, iterations, lbfgs_steps,AE_name,dset_dir,output_dir_AE,save_plots_AE,lambda_dx,lambda_dz,miles_lr, gamma_lr,path,load_path, batch_size,
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

    def __init__(self, AE, net, dt,z_gt,sys_name, output_dir,save_plots, criterion, optimizer, lr, weight_decay_GFINNs, iterations, lbfgs_steps,AE_name,dset_dir,output_dir_AE,save_plots_AE,
             lambda_dx,lambda_dz,miles_lr, gamma_lr,path,load_path, batch_size,batch_size_test, print_every, save, load, callback, dtype, device,trunc_period):
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
        self.weight_decay = weight_decay_GFINNs
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
        
        self.miles_lr = miles_lr
        self.gamma_lr = gamma_lr

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self.save_plots_AE = save_plots_AE


        if self.load:
            path = './outputs/' + self.load_path
            #self.SAE = torch.load( path + '/model_best_AE.pkl')
            self.net = torch.load( path + '/model_best.pkl')
        else:
            self.SAE = AE
            if self.device == 'gpu':
                self.SAE = self.SAE.to(torch.device('cuda'))

            for param in self.SAE.parameters():
                param.requires_grad = False

            # if self.sys_name == 'viscoelastic':
            #     #self.SAE = SparseAutoEncoder(layer_vec_SAE, activation_SAE).float()
            #     self.SAE = SparseAutoEncoder(layer_vec_SAE, activation_SAE).double()
            #     if self.device =='gpu':
            #         self.SAE = self.SAE.to(torch.device('cuda'))
            #
            # elif self.sys_name == '1DBurgers':
            #     #self.SAE = SparseAutoEncoder(layer_vec_SAE, activation_SAE).float()
            #     self.SAE = SparseAutoEncoder(layer_vec_SAE, activation_SAE).double()
            #     if self.device =='gpu':
            #         self.SAE = self.SAE.to(torch.device('cuda'))
            #
            # elif self.sys_name == 'rolling_tire':
            #     #self.SAE = StackedSparseAutoEncoder(layer_vec_SAE_q, layer_vec_SAE_v, layer_vec_SAE_sigma,
            #     #                                    activation_SAE).float()
            #     self.SAE = StackedSparseAutoEncoder(layer_vec_SAE_q, layer_vec_SAE_v, layer_vec_SAE_sigma,
            #                                         activation_SAE).double()
            #     if self.device =='gpu':
            #         self.SAE = self.SAE.to(torch.device('cuda'))

        print(sum(p.numel() for p in self.SAE .parameters() if p.requires_grad))
        print(sum(p.numel() for p in self.net.parameters() if p.requires_grad))




        # Dataset Parameters
        self.dset_dir = dset_dir
        self.dataset = load_dataset(self.sys_name, self.dset_dir,self.device,self.dtype)
        self.dt = self.dataset.dt
        self.dim_t = self.dataset.dim_t


        self.train_snaps, self.test_snaps = split_dataset(self.sys_name, self.dim_t-1)


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
        #loss_AE_jac_history = []
        loss_dx_history = []
        loss_dz_history = []

        z_gt_tr = self.dataset.z[self.train_snaps, :]
        z_gt_tt = self.dataset.z[self.test_snaps, :]
        dz_gt_tr = self.dataset.dz[self.train_snaps, :]

       # z_gt_tr = z_gt_tr.requires_grad_(True)
#         z_gt_tt = z_gt_tt.requires_grad_(True)


#         dz_gt_tr = dz_gt_tr.requires_grad_(True)


        dz_gt_tt = self.dataset.dz[self.test_snaps, :]
#         dz_gt_tt = dz_gt_tt.requires_grad_(True)
#         z_gt_tt = z_gt_tt.requires_grad_(True)



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
        for i in range(self.iterations + 1):
                        
#             print(z_gt_tr_norm.shape)
#             print(z_gt_tr_norm.size(0))
            
            z_gt_tr_norm,z1_gt_tr_norm, mask_tr = self.z_data.get_batch(self.batch_size)
            z_gt_tt_norm,z1_gt_tt_norm, mask_tt = self.z_data.get_batch_test(self.batch_size_test)
            
            dz_gt_tr_norm = dz_gt_tr_norm_tmp[mask_tr]
            dz_gt_tt_norm = dz_gt_tt_norm_tmp[mask_tt]

            z_sae_tr_norm, x = self.SAE(z_gt_tr_norm)
            z_sae_tt_norm, x_tt = self.SAE(z_gt_tt_norm)

            z1_sae_tr_norm, x1 = self.SAE(z1_gt_tr_norm)
            z1_sae_tt_norm, x1_tt = self.SAE(z1_gt_tt_norm)

            self.data = Data(x, x1, x_tt, x1_tt)
            self.data.device = self.device
            self.data.dtype = self.dtype

            X_train, y_train= x, x1

            loss_GFINNs = self.__criterion(self.net(X_train), y_train)
            
            


            if  (self.lambda_dz == 0 and self.lambda_dx == 0):
                
                loss_dx = torch.tensor(0)
                loss_dz = torch.tensor(0)
            else:
                _, J_e, J_d,idx_trunc = self.SAE.jacobian_norm_trunc_wo_jac_loss(z_gt_tr_norm, x, self.trunc_period)


            
                dx_train = self.net.f(X_train)


                dz_gt_tr_norm = dz_gt_tr_norm.unsqueeze(2)


                dx_data_train = J_e @ dz_gt_tr_norm[:,idx_trunc]
                dx_data_train = dx_data_train.squeeze()

                # print(dx_train.shape)
                # print(grad(self.SAE.decode(X_train),X_train).shape)
                dx_train = dx_train.unsqueeze(2)
                #dz_train = grad(self.SAE.decode(X_train),X_train)@dx_train
                #dz_train = J_d @ dx_train


                dx_train = dx_train.squeeze()
                #dz_train = dz_train.squeeze()

                dz_gt_tr_norm = dz_gt_tr_norm.squeeze()




                loss_dx = torch.mean((dx_train - dx_data_train) ** 2)
                #loss_dz = torch.mean((dz_train - dz_gt_tr_norm[:,idx_trunc]) ** 2)


            loss = loss_GFINNs+self.lambda_dx*loss_dx



            #print(loss) #tensor(0.0008, grad_fn=<MseLossBackward0>)
            Loss_early = 1e-10


            if i % self.print_every == 0 or i == self.iterations:
                X_test, y_test = x_tt, x1_tt

                dx_test = self.net.f(X_test)
                dz_gt_tt_norm = dz_gt_tt_norm.unsqueeze(2)

                #loss_AE_jac_test, J_e, J_d, idx_trunc = self.SAE.jacobian_norm_trunc(z_gt_tt_norm, x_tt)

                _,J_e, J_d, idx_trunc = self.SAE.jacobian_norm_trunc_wo_jac_loss(z_gt_tt_norm, x_tt,self.trunc_period)


                #dx_data_test = grad(self.SAE.encode(z_gt_tt_norm), z_gt_tt_norm) @ dz_gr_tt_norm

                dx_data_test = J_e @ dz_gt_tt_norm[:,idx_trunc]
                dx_data_test = dx_data_test.squeeze()
                dz_gt_tt_norm = dz_gt_tt_norm.squeeze()

                dx_test = dx_test.unsqueeze(2)
                #dz_test = grad(self.SAE.decode(X_test), X_test) @ dx_test
                #dz_test = J_d @ dx_test

                dx_test = dx_test.squeeze()
                #dz_test = dz_test.squeeze()


                loss_GFINNs_test = self.__criterion(self.net(X_test), y_test)
                # loss_AE_recon_test = torch.mean((z_sae_tt_norm - z_gt_tt_norm) ** 2)

                loss_dx_test = torch.mean((dx_test - dx_data_test) ** 2)
                #loss_dz_test = torch.mean((dz_test - dz_gt_tt_norm[:,idx_trunc]) ** 2)
                #loss_AE_GFINNs_test = torch.mean((z_sae_gfinns_tt_norm - z_gt_tt_norm1) ** 2)

                loss_test = loss_GFINNs_test+loss_dx_test

                # print('{:<9}a loss: %.4e{:<25}Test loss: %.4e{:<25}'.format(i, loss.item(), loss_test.item()), flush=True)
                print(' ADAM || It: %05d, Loss: %.4e, loss_GFINNs: %.4e, loss_dx: %.4e, Test: %.4e' %
                      (i, loss.item(),loss_GFINNs.item(),loss_dx.item(), loss_test.item()))
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
                    loss_history.append([i, loss.item(), loss_test.item(), *output])
                    loss_GFINNs_history.append([i, loss_GFINNs.item(), loss_GFINNs_test.item(), *output])
                    loss_dx_history.append([i, loss_dx.item(), loss_dx_test.item(), *output])
                   # loss_dz_history.append([i, loss_dz.item(), loss_dz_test.item(), *output])
             #       loss_AE_GFINNs_history.append([i, loss_AE_GFINNs.item(), loss_AE_GFINNs_test.item(), *output])
                else:
                    loss_history.append([i, loss.item(), loss_test.item()])
                    loss_history.append([i, loss.item(), loss_test.item()])
                    loss_GFINNs_history.append([i, loss_GFINNs.item(), loss_GFINNs_test.item()])
                    loss_dx_history.append([i, loss_dx.item(), loss_dx_test.item()])
                    #loss_dz_history.append([i, loss_dz.item(), loss_dz_test.item()])
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
                #print(loss)
                loss.backward(retain_graph=True)
                #loss.backward()
                self.__optimizer.step()
                self.__scheduler.step()
                
        self.loss_history = np.array(loss_history)
        self.loss_GFINNs_history = np.array(loss_GFINNs_history)
        self.loss_dx_history = np.array(loss_dx_history)
        #self.loss_dz_history = np.array(loss_dz_history)

        _, x_de = self.SAE(z_gt_norm)
        if self.sys_name == 'viscoelastic':
            # Plot latent variables
            if (self.save_plots == True):
                plot_name = '[VC] AE Latent Variables_' + self.AE_name
                plot_latent_visco(x_de, self.dataset.dt, plot_name, self.output_dir)

        elif self.sys_name == '1DBurgers':

            # Plot latent variables
            if (self.save_plots == True):
                plot_name = '[1DBurgers] AE Latent Variables_' + self.AE_name
                plot_latent_visco(x_de, self.dataset.dt, plot_name, self.output_dir)

        elif self.sys_name == 'rolling_tire':
            x_q, x_v, x_sigma = self.SAE.split_latent(x_de)

            # Plot latent variables
            if (self.save_plots == True):
                plot_name = '[Rolling Tire] AE Latent Variables_' + self.AE_name
                plot_latent_tire(x_q, x_v, x_sigma, self.dataset.dt, plot_name, self.output_dir)

        # print('Done!', flush=True)
        return self.loss_history, self.loss_GFINNs_history,self.loss_dx_history



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
                    loss.backward(retain_graph=True)
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

            p3,=plt.plot(self.loss_GFINNs_history[:,0], self.loss_GFINNs_history[:,1],'-')
            p4,=plt.plot(self.loss_GFINNs_history[:,0], self.loss_GFINNs_history[:,2],'--')
            plt.legend(['train loss (GFINNs)', 'test loss (GFINNs)'])  # , '$\hat{u}$'])
            plt.yscale('log')
            plt.savefig(path + '/loss_GFINNs_'+self.AE_name+self.sys_name+'.png')
            p3.remove()
            p4.remove()

            # p5,=plt.plot(self.loss_AE_recon_history[:,0], self.loss_AE_recon_history[:,1],'-')
            # p6,=plt.plot(self.loss_AE_recon_history[:,0], self.loss_AE_recon_history[:,2],'--')
            # plt.legend(['train loss (AE recon)', 'test loss (AE recon)'])  # , '$\hat{u}$'])
            # plt.yscale('log')
            # plt.savefig(path + '/loss_AE_recon_'+self.AE_name+self.sys_name+'.png')
            # p5.remove()
            # p6.remove()

            # p7,=plt.plot(self.loss_AE_jac_history[:,0], self.loss_AE_jac_history[:,1],'-')
            # p8,=plt.plot(self.loss_AE_jac_history[:,0], self.loss_AE_jac_history[:,2],'--')
            # plt.legend(['train loss (AE jac)', 'test loss (AE jac)'])  # , '$\hat{u}$'])
            # plt.yscale('log')
            # plt.savefig(path + '/loss_AE_jac_'+self.AE_name+self.sys_name+'.png')
            # p7.remove()
            # p8.remove()

            p9,=plt.plot(self.loss_dx_history[:,0], self.loss_dx_history[:,1],'-')
            p10,=plt.plot(self.loss_dx_history[:,0], self.loss_dx_history[:,2],'--')
            plt.legend(['train loss (dx)', 'test loss (dx)'])  # , '$\hat{u}$'])
            plt.yscale('log')
            plt.savefig(path + '/loss_dx_'+self.AE_name+self.sys_name+'.png')
            p9.remove()
            p10.remove()

#             p11,=plt.plot(self.loss_dz_history[:,0], self.loss_dz_history[:,1],'-')
#             p12,=plt.plot(self.loss_dz_history[:,0], self.loss_dz_history[:,2],'--')
#             plt.legend(['train loss (dz)', 'test loss (dz)'])  # , '$\hat{u}$'])
#             plt.yscale('log')
#             plt.savefig(path + '/loss_dz_'+self.AE_name+self.sys_name+'.png')
#             p11.remove()
#             p12.remove()

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
            self.__optimizer = torch.optim.Adam(list(self.net.parameters())+list(self.SAE.parameters()), lr=self.lr, weight_decay=self.weight_decay_GFINNs)
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


        self.net = self.best_model
        self.SAE = self.best_model_AE

        self.dim_t = self.z_gt.shape[0]



        z_gt_norm = self.SAE.normalize(self.z_gt)
        z = z_gt_norm[0, :]
        z = torch.unsqueeze(z, 0)


        # Forward pass
        z_sae_norm, x_all = self.SAE(z_gt_norm)

        z_sae = self.SAE.denormalize(z_sae_norm)

        #z_norm = self.SAE.normalize(z)

        _, x = self.SAE(z)


        if self.dtype == 'float':
            x_net = torch.zeros(x_all.shape).float()
            x_net_all = torch.zeros(x_all.shape).float()

        elif self.dtype == 'double':
            x_net = torch.zeros(x_all.shape).double()
            x_net_all = torch.zeros(x_all.shape).double()
            

        x_net[0,:] = x


        x_net_all[0,:] = x
        x_net_all[1:,:] = self.net.integrator2(self.net(x_all[:-1,:]))

        #print(x_net.shape)
        if self.device == 'gpu':
          x_net = x_net.to(torch.device('cuda'))
          x_net_all = x_net_all.to(torch.device('cuda'))

        if self.dtype == 'float':
            dSdt_net = torch.zeros(x_all.shape).float()
            dEdt_net = torch.zeros(x_all.shape).float()
        elif self.dtype == 'double':
            dSdt_net = torch.zeros(x_all.shape).double()
            dEdt_net = torch.zeros(x_all.shape).double()
            

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
            # Structure-Preserving Neural Network

            # x1_net = self.net(x)
            # print(x1_net.shape)
            x1_net = self.net.integrator2(self.net(x))
            # x1_net = self.net.criterion(self.net(x), self.dt)

            # dEdt, dSdt = self.SPNN.get_thermodynamics(x)

            # Save results and Time update
            x_net[snapshot + 1, :] = x1_net
            # dEdt_net[snapshot] = dEdt
            # dSdt_net[snapshot] = dSdt
            x = x1_net

            dE, M = self.net.netE(x)
            #     print(dE.shape)
            # print(M.shape)
            dS, L = self.net.netS(x)

            dEdt = dE @ ((dE @ L).squeeze() + (dS @ M).squeeze())
            dSdt = dS @ ((dE @ L).squeeze() + (dS @ M).squeeze())

            # dEdt = dE @ M
            # dSdt = dS @ L

            # print(dSdt.shape)

            dEdt_net[snapshot + 1, :] = dEdt
            dSdt_net[snapshot + 1, :] = dSdt

        x_gfinn = x_net

        # Decode latent vector
        z_gfinn_norm = self.SAE.decode(x_gfinn)
        z_gfinn = self.SAE.denormalize(z_gfinn_norm)

        z_gfinn_all_norm = self.SAE.decode(x_net_all)
        z_gfinn_all = self.SAE.denormalize(z_gfinn_all_norm)

        # Load Ground Truth and Compute MSE
        z_gt = self.z_gt
        print_mse(z_gfinn, z_gt, self.sys_name)
        print_mse(z_gfinn_all, z_gt, self.sys_name)
        print_mse(z_sae, z_gt, self.sys_name)


        # Plot results
        if (self.save_plots):
            #plot_name = 'SPNN Full Integration (Latent)'
            #plot_latent(x_net, self.x_trunc, dEdt_net, dSdt_net, self.dt, plot_name, self.output_dir, self.sys_name)
            plot_name = 'Energy_Entropy_Derivatives_' +self.AE_name
            plot_latent(dEdt_net, dSdt_net, self.dt, plot_name, self.output_dir, self.sys_name)

            plot_name = 'GFINNs Full Integration_'+self.AE_name
            #print(self.sys_name)
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

            elif self.sys_name == 'rolling_tire':
                x_q, x_v, x_sigma = self.SAE.split_latent(x_gfinn)

                # Plot latent variables
                if (self.save_plots == True):
                    plot_name = '[Rolling Tire] Latent Variables_' + self.AE_name
                    plot_latent_tire(x_q, x_v, x_sigma, self.dataset.dt, plot_name, self.output_dir)

        print("\n[GFINNs Testing Finished]\n")
