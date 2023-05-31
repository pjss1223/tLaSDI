"""
@author: jpzxshi & zen
"""
import os
import time
import numpy as np
import torch

from data2 import Data
from .nn import LossNN
from .utils import timing, cross_entropy_loss, mse

#from utilities.plot_gfinns import plot_results #, plot_latent

import torch
import torch.optim as optim
import numpy as np

from model import SparseAutoEncoder, StackedSparseAutoEncoder
from dataset_sim_parain import load_dataset, split_dataset
from utilities.plot import plot_results, plot_latent_visco, plot_latent_tire, plot_latent
from utilities.utils import print_mse, all_latent
import matplotlib.pyplot as plt

class Brain_test_parain_sim:
    '''Runner based on torch.
    '''
    brain = None

    @classmethod
    def Init(cls,  net, dt, z_gt, sys_name, output_dir, save_plots, criterion, optimizer, lr,
             iterations, lbfgs_steps, AE_name,dset_dir,output_dir_AE,save_plots_AE,layer_vec_SAE,layer_vec_SAE_q,layer_vec_SAE_v,layer_vec_SAE_sigma,
             activation_SAE,lr_SAE,miles_SAE,gamma_SAE,lambda_r_SAE, path=None, batch_size=None,
             batch_size_test=None, weight_decay=0, print_every=1000, save=False, callback=None, dtype='float',
             device='cpu'):
        cls.brain = cls( net, dt, z_gt, sys_name, output_dir, save_plots, criterion,
                         optimizer, lr, weight_decay, iterations, lbfgs_steps,AE_name,dset_dir,output_dir_AE,save_plots_AE,layer_vec_SAE,
                         layer_vec_SAE_q,layer_vec_SAE_v,layer_vec_SAE_sigma,activation_SAE,lr_SAE,miles_SAE,gamma_SAE,lambda_r_SAE, path, batch_size,
                         batch_size_test, print_every, save, callback, dtype, device)

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

    def __init__(self,  net, dt,z_gt,sys_name, output_dir,save_plots, criterion, optimizer, lr, weight_decay, iterations, lbfgs_steps,AE_name,dset_dir,output_dir_AE,save_plots_AE,layer_vec_SAE,layer_vec_SAE_q,layer_vec_SAE_v,layer_vec_SAE_sigma,
             activation_SAE,lr_SAE,miles_SAE,gamma_SAE,lambda_r_SAE, path, batch_size,
                 batch_size_test, print_every, save, callback, dtype, device):
        #self.data = data
        self.net = net
        #print(self.net.netE.fnnB.modus['LinMout'].weight)
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
        self.weight_decay = weight_decay
        self.iterations = iterations
        self.lbfgs_steps = lbfgs_steps
        self.path = path
        self.batch_size = batch_size
        self.batch_size_test = batch_size_test
        self.print_every = print_every
        self.save = save
        self.callback = callback
        self.dtype = dtype
        self.device = device
        self.AE_name = AE_name
        #self.dset_dir = dset_dir
        self.output_dir_AE = output_dir_AE
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self.save_plots_AE = save_plots_AE
        if self.sys_name == 'viscoelastic':
            #self.SAE = SparseAutoEncoder(layer_vec_SAE, activation_SAE).float()
            self.SAE = SparseAutoEncoder(layer_vec_SAE, activation_SAE).double()
            if self.device =='gpu':
                self.SAE = self.SAE.to(torch.device('cuda'))

        elif self.sys_name == '1DBurgers':
            #self.SAE = SparseAutoEncoder(layer_vec_SAE, activation_SAE).float()
            self.SAE = SparseAutoEncoder(layer_vec_SAE, activation_SAE).double()
            if self.device =='gpu':
                self.SAE = self.SAE.to(torch.device('cuda'))

        elif self.sys_name == 'rolling_tire':
            #self.SAE = StackedSparseAutoEncoder(layer_vec_SAE_q, layer_vec_SAE_v, layer_vec_SAE_sigma,
            #                                    activation_SAE).float()
            self.SAE = StackedSparseAutoEncoder(layer_vec_SAE_q, layer_vec_SAE_v, layer_vec_SAE_sigma,
                                                activation_SAE).double()
            if self.device =='gpu':
                self.SAE = self.SAE.to(torch.device('cuda'))





        # Dataset Parameters
        if self.sys_name == '1DBurgers':
            self.total_paras = 256
        elif self.sys_name == '2DBurgers':
            self.total_paras = 0

        train_paras = int(0.75 * self.total_paras)

        # Random split
        indices = np.arange(self.total_paras)
        np.random.shuffle(indices)
        path = './outputs/'

        torch.save(indices, path + '/1DBG_para_data_split_indices.p')
        # torch.save(indices, path + '/2DBG_para_data_split_indices.p')

        # if sys_name == '1DBurgers':
        #     indices = torch.load(path + '/1DBG_para_data_split_indices.p')
        #
        # elif sys_name == '2DBurgers':
        #     indices = torch.load(path + '/2DBG_para_data_split_indices.p')

        train_indices = indices[:train_paras]
        test_indices = indices[train_paras:self.total_paras]

        self.train_indices = train_indices
        self.test_indices = test_indices
        self.dset_dir = dset_dir


        self.lambda_r = lambda_r_SAE

        # self.dataset = load_dataset(self.sys_name, self.dset_dir,self.device,0)
        # self.dt = self.dataset.dt
        # self.dim_t = self.dataset.dim_t
        # self.mu = self.dataset.mu



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

        for i in range(self.iterations + 1):

            loss_GFINNs = 0
            loss_AE = 0
            for para_i in self.train_indices:
                self.dataset = load_dataset(self.sys_name, self.dset_dir, self.device, para_i)
                self.dt = self.dataset.dt
                self.dim_t = self.dataset.dim_t
                self.mu = self.dataset.mu[para_i]
                self.mu_tmp = self.mu.T.repeat(self.dim_t,1)
                self.dim_mu = self.mu.shape[0]

                z_gt_tr = self.dataset.z[:self.dim_t - 1, :]
                #z_gt_tt = self.dataset.z[self.test_snaps, :]

                z1_gt_tr = self.dataset.z[1:self.dim_t, :]
                #z1_gt_tt = self.dataset.z[self.test_snaps+1, :]

                z_gt_tr_norm = self.SAE.normalize(z_gt_tr)
                #z_gt_tt_norm = self.SAE.normalize(z_gt_tt)

                z1_gt_tr_norm = self.SAE.normalize(z1_gt_tr)
                #z1_gt_tt_norm = self.SAE.normalize(z1_gt_tt)

                z_gt_norm = self.SAE.normalize(self.dataset.z)

                z_sae_tr_norm, x = self.SAE(z_gt_tr_norm)
                #z_sae_tt_norm, x_tt = self.SAE(z_gt_tt_norm)

                z1_sae_tr_norm, x1 = self.SAE(z1_gt_tr_norm)
                #z1_sae_tt_norm, x1_tt = self.SAE(z1_gt_tt_norm)

                x_sae_tr, x1_sae_tr = x, x1
                #x_sae_tt, x1_sae_tt = x_tt, x1_tt


                #MODIFY this part if you want batches
                # data = Data(x_sae_tr, x1_sae_tr, x_sae_tt, x1_sae_tt)
                # data.device = self.device
                # data.dtype = self.dtype

                #X_train, y_train = data.get_batch(self.batch_size)

                X_train, y_train = x, x1

                X_train = torch.cat((X_train, self.mu_tmp[:-1,:]), 1)
                y_train = torch.cat((y_train, self.mu_tmp[:-1,:]), 1)

                #print(X_train.shape) #1000, 12
                #print(self.net(X_train).shape) #1000, 12

            # print(X_train)
            # print(y_train)

                loss_GFINNs_tmp = self.__criterion(self.net(X_train), y_train)


                X_train1 = self.net.integrator2(self.net(X_train))

                # z_sae_gfinns_tr_norm = self.SAE.decode(X_train1)
                # z_gt_tr_norm1 = self.dataset.z[self.train_snaps+1, :]

                # z_gt_tt_norm1 = self.dataset.z[self.test_snaps[:-1] + 1, :]


                loss_AE_tmp = torch.mean((z_sae_tr_norm - z_gt_tr_norm) ** 2)

                loss_GFINNs = loss_GFINNs+ loss_GFINNs_tmp
                loss_AE = loss_AE+ loss_AE_tmp
           # loss_AE_GFINNs = torch.mean((z_sae_gfinns_tr_norm - z_gt_tr_norm1) ** 2)
            loss_GFINNs = loss_GFINNs/self.dim_mu
            loss_AE = loss_AE/self.dim_mu

            loss = loss_GFINNs+self.lambda_r*loss_AE#+loss_AE_GFINNs



            #print(loss) #tensor(0.0008, grad_fn=<MseLossBackward0>)
            Loss_early = 1e-10


            if i % self.print_every == 0 or i == self.iterations:

                loss_GFINNs_test = 0
                loss_AE_test = 0
                for para_i in self.test_indices:
                    self.dataset = load_dataset(self.sys_name, self.dset_dir, self.device, para_i)
                    self.dt = self.dataset.dt
                    self.dim_t = self.dataset.dim_t
                    self.mu = self.dataset.mu[para_i]
                    self.mu_tmp = self.mu.T.repeat(self.dim_t, 1)

                    z_gt_tt = self.dataset.z[:self.dim_t - 1, :]

                    z1_gt_tt = self.dataset.z[1:self.dim_t, :]

                    z_gt_tt_norm = self.SAE.normalize(z_gt_tt)

                    z1_gt_tt_norm = self.SAE.normalize(z1_gt_tt)

                    z_gt_norm = self.SAE.normalize(self.dataset.z)

                    z_sae_tt_norm, x_tt = self.SAE(z_gt_tt_norm)

                    #z1_sae_tt_norm, x1_tt = self.SAE(z1_gt_tt_norm)


                    # MODIFY this part if you want batches
                    # data = Data(x_sae_tr, x1_sae_tr, x_sae_tt, x1_sae_tt)
                    # data.device = self.device
                    # data.dtype = self.dtype

                    # X_train, y_train = data.get_batch(self.batch_size)

                    X_test, y_test = x_tt, x_tt

                    X_test = torch.cat((X_test, self.mu_tmp[:-1,:]), 1)
                    y_test = torch.cat((y_test, self.mu_tmp[:-1,:]), 1)



                    loss_GFINNs_test_tmp = self.__criterion(self.net(X_test), y_test)
                    loss_AE_test_tmp = torch.mean((z_sae_tt_norm - z_gt_tt_norm) ** 2)

                    loss_GFINNs_test = loss_GFINNs_test+loss_GFINNs_test_tmp
                    loss_AE_test = loss_AE_test+loss_AE_test_tmp
                #loss_AE_GFINNs_test = torch.mean((z_sae_gfinns_tt_norm - z_gt_tt_norm1) ** 2)
                loss_GFINNs_test = loss_GFINNs_test / self.dim_mu
                loss_AE_test = loss_AE_test / self.dim_mu
                loss_test = loss_GFINNs_test+loss_AE_test#+loss_AE_GFINNs_test

                # print('{:<9}a loss: %.4e{:<25}Test loss: %.4e{:<25}'.format(i, loss.item(), loss_test.item()), flush=True)
                print(' ADAM || It: %05d, Loss: %.4e, Test: %.4e' %
                      (i, loss.item(), loss_test.item()))
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
             #    if self.callback is not None:
             #        output = self.callback(data, self.net)
             #        loss_history.append([i, loss.item(), loss_test.item(), *output])
             #        loss_GFINNs_history.append([i, loss_GFINNs.item(), loss_GFINNs_test.item(), *output])
             #        loss_AE_history.append([i, loss_AE.item(), loss_AE_test.item(), *output])
             # #       loss_AE_GFINNs_history.append([i, loss_AE_GFINNs.item(), loss_AE_GFINNs_test.item(), *output])
                if self.callback is None:
                    loss_history.append([i, loss.item(), loss_test.item()])
                    loss_history.append([i, loss.item(), loss_test.item()])
                    loss_GFINNs_history.append([i, loss_GFINNs.item(), loss_GFINNs_test.item()])
                    loss_AE_history.append([i, loss_AE.item(), loss_AE_test.item()])
               #     loss_AE_GFINNs_history.append([i, loss_AE_GFINNs.item(), loss_AE_GFINNs_test.item(), *output])
                if loss <= Loss_early:
                    print('Stop training: Loss under %.2e' % Loss_early)
                    break
            if i < self.iterations:
                self.__optimizer.zero_grad()
                #print(loss)
                loss.backward(retain_graph=True)
                #loss.backward()
                self.__optimizer.step()
        self.loss_history = np.array(loss_history)
        self.loss_GFINNs_history = np.array(loss_GFINNs_history)
        self.loss_AE_history = np.array(loss_AE_history)

        _, x_de = self.SAE(z_gt_norm)
        if self.sys_name == 'viscoelastic':
            # Plot latent variables
            if (self.save_plots == True):
                plot_name = '[VC] AE Latent Variables_'+self.AE_name
                plot_latent_visco(x_de, self.dataset.dt, plot_name, self.output_dir)

        elif self.sys_name == '1DBurgers':

            # Plot latent variables
            if (self.save_plots == True):
                plot_name = '[1DBurgers] AE Latent Variables_'+self.AE_name
                plot_latent_visco(x_de, self.dataset.dt, plot_name, self.output_dir)

        elif self.sys_name == 'rolling_tire':
            x_q, x_v, x_sigma = self.SAE.split_latent(x_de)

            # Plot latent variables
            if (self.save_plots == True):
                plot_name = '[Rolling Tire] AE Latent Variables_'+self.AE_name
                plot_latent_tire(x_q, x_v, x_sigma, self.dataset.dt, plot_name, self.output_dir)



        # print('Done!', flush=True)
        return self.loss_history, self.loss_GFINNs_history, self.loss_AE_history



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

        #use this for lbfgs
        # if self.lbfgs_steps != 0:
        #     def closure():
        #         if torch.is_grad_enabled():
        #             optim.zero_grad()
        #
        #
        #         # X_train, y_train = data.get_batch(None)
        #         #
        #         # X_test, y_test = data.get_batch_test(None)
        #
        #
        #         loss = self.best_model.criterion(self.best_model(X_train), y_train)
        #         loss_test = self.best_model.criterion(self.best_model(X_test), y_test)
        #         # print('Train loss: {:<25}Test loss: {:<25}'.format(loss.item(), loss_test.item()), flush=True)
        #         it = self.it + 1
        #         if it % self.print_every == 0 or it == self.lbfgs_steps:
        #             print('L-BFGS|| It: %05d, Loss: %.4e, Test: %.4e' %
        #                   (it, loss.item(), loss_test.item()))
        #         self.it = it
        #         if loss.requires_grad:
        #             loss.backward(retain_graph=True)
        #         return loss
        #
        #     optim.step(closure)
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
            path = './outputs/' + self.AE_name+'_'+ self.path
        if not os.path.isdir(path): os.makedirs(path)

        if best_model:
            torch.save(self.best_model, path + '/model_best.pkl')
        if loss_history:
            np.savetxt(path + '/loss.txt', self.loss_history)
            p1,=plt.plot(self.loss_history[:,0], self.loss_history[:,1],'-')
            p2,= plt.plot(self.loss_history[:,0], self.loss_history[:,2],'--')
            plt.legend(['train loss', 'test loss'])  # , '$\hat{u}$'])
            plt.yscale('log')
            plt.savefig(path + '/loss_'+self.AE_name+'.png')
            p1.remove()
            p2.remove()

            p3,=plt.plot(self.loss_GFINNs_history[:,0], self.loss_GFINNs_history[:,1],'-')
            p4,=plt.plot(self.loss_GFINNs_history[:,0], self.loss_GFINNs_history[:,2],'--')
            plt.legend(['train loss (GFINNs)', 'test loss (GFINNs)'])  # , '$\hat{u}$'])
            plt.yscale('log')
            plt.savefig(path + '/loss_GFINNs_'+self.AE_name+'.png')
            p3.remove()
            p4.remove()

            p5,=plt.plot(self.loss_AE_history[:,0], self.loss_AE_history[:,1],'-')
            p6,=plt.plot(self.loss_AE_history[:,0], self.loss_AE_history[:,2],'--')
            plt.legend(['train loss (AE)', 'test loss (AE)'])  # , '$\hat{u}$'])
            plt.yscale('log')
            plt.savefig(path + '/loss_AE_'+self.AE_name+'.png')
            p5.remove()
            p6.remove()

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

    ##from spnn
    def test(self):
        print("\n[GFNN Testing Started]\n")

        self.dim_t = self.z_gt.shape[0]

        loss_test_final = 0
        for para_i in self.test_indices:
            self.dataset = load_dataset(self.sys_name, self.dset_dir, self.device, para_i)
            self.dt = self.dataset.dt
            self.dim_t = self.dataset.dim_t
            self.mu = self.dataset.mu[para_i]
            self.mu_tmp = self.mu.T.repeat(self.dim_t, 1)


            z_gt_norm = self.SAE.normalize(self.dataset.z)
            z = z_gt_norm[0, :]
            z = torch.unsqueeze(z, 0)




            # Forward pass
            z_sae_norm, x_all = self.SAE(z_gt_norm)

            z_sae = self.SAE.denormalize(z_sae_norm)

            #z_norm = self.SAE.normalize(z)

            _, x = self.SAE(z)

            # print(self.mu.shape)
            # print(x.shape)
            # print(x_all.shape)
            # print(self.mu_tmp)
            # print(self.mu_tmp.shape)
            # print(torch.unsqueeze(self.mu, 0).shape)
            self.mu = torch.unsqueeze(self.mu, 0)

            x_all = torch.cat((x_all, self.mu_tmp), 1)
            x = torch.cat((x, self.mu), 1)


            x_net = torch.zeros(x_all.shape).double()

            x_net_all = torch.zeros(x_all.shape).double()




            x_net[0,:] = x


            x_net_all[0,:] = x
            x_net_all[1:,:] = self.net.integrator2(self.net(x_all[:-1,:]))

            if self.device == 'gpu':
              x_net = x_net.to(torch.device('cuda'))
              x_net_all = x_net_all.to(torch.device('cuda'))

            dSdt_net = torch.zeros(x_all.shape).double()
            dEdt_net = torch.zeros(x_all.shape).double()

            dE, M = self.net.netE(x)
              #     print(dE.shape)
              # print(M.shape)
            dS, L = self.net.netS(x)

            dEdt = dE @((dE @ L).squeeze() + (dS @ M).squeeze())
            dSdt = dS @ ((dE @ L).squeeze() + (dS @ M).squeeze())
            #
            # print(dE.shape)
            # print(((dE @ L).squeeze() + (dS @ M).squeeze()).shape)
            #
            # dEdt = dE @ M
            # dSdt = dS @ L



            dEdt_net[0, :] = dEdt
            dSdt_net[0, :] = dSdt

            #print(x_net.shape)


            for snapshot in range(self.dim_t - 1):
                # Structure-Preserving Neural Network

                #x1_net = self.net(x)
                #print(x1_net.shape)
                # print(x.shape)
                # print(self.net(x).shape)
                x1_net = self.net.integrator2(self.net(x))
                #x1_net = self.net.criterion(self.net(x), self.dt)




                #dEdt, dSdt = self.SPNN.get_thermodynamics(x)

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


                #print(dSdt.shape)

                dEdt_net[snapshot+1, :] = dEdt
                dSdt_net[snapshot+1, :] = dSdt



            # Detruncate
            # x_gfinn = torch.zeros([self.dim_t, self.SAE.dim_latent])
            # x_gfinn[:, latent_idx] = x_net

            x_gfinn = x_net


            # Decode latent vector
            z_gfinn_norm = self.SAE.decode(x_gfinn[:,:-self.dim_mu])
            z_gfinn = self.SAE.denormalize(z_gfinn_norm)

            z_gfinn_all_norm = self.SAE.decode(x_net_all[:,:-self.dim_mu])
            z_gfinn_all = self.SAE.denormalize(z_gfinn_all_norm)

            # Load Ground Truth and Compute MSE
            z_gt = self.z_gt
            #print_mse(z_gfinn, z_gt, self.sys_name)
            loss_test_final_tmp = mse(z_gfinn,z_gt)

            loss_test_final = loss_test_final+loss_test_final_tmp

        loss_test_final = loss_test_final/self.dim_mu

        print(' Final test error: %.4e' % loss_test_final)

            # print_mse(z_gfinn_all, z_gt, self.sys_name)
            # print_mse(z_sae, z_gt, self.sys_name)


        # Plot results
        if (self.save_plots):
            #plot_name = 'SPNN Full Integration (Latent)'
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
                    plot_name = '[RT] Latent Variables_' + self.AE_name
                    plot_latent_tire(x_q, x_v, x_sigma, self.dataset.dt, plot_name, self.output_dir)

        print("\n[GFINNs Testing Finished]\n")
