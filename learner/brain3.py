"""
@author: jpzxshi & zen
"""
import os
import time
import numpy as np
import torch

from .nn import LossNN
from .utils import timing, cross_entropy_loss


#from ..dataset import load_dataset, split_dataset
from utilities.plot import plot_results, plot_latent
from utilities.utils import print_mse



class Brain:
    '''Runner based on torch.
    '''
    brain = None

    def Init(cls, data, net, x_trunc,latent_idx,dt,z_gt,sys_name,output_dir,save_plots, criterion, optimizer, lr, iterations, lbfgs_steps, path=None, batch_size=None,
             batch_size_test=None, weight_decay=0, print_every=1000, save=False, callback=None, dtype='float',
             device='cpu'):
        cls.brain = cls(data, net,x_trunc, latent_idx,dt, z_gt,sys_name,output_dir,save_plots ,criterion, optimizer, lr, weight_decay, iterations, lbfgs_steps, path, batch_size,
                        batch_size_test, print_every, save, callback, dtype, device)

    @classmethod
    def Run(cls):
        cls.brain.run()

    @classmethod
    def Restore(cls):
        cls.brain.restore()

    @classmethod
    def Output(cls, data=True, best_model=True, loss_history=True, info=None, **kwargs):
        cls.brain.output(data, best_model, loss_history, info, **kwargs)

    @classmethod
    def Loss_history(cls):
        return cls.brain.loss_history

    @classmethod
    def Encounter_nan(cls):
        return cls.brain.encounter_nan

    @classmethod
    def Best_model(cls):
        return cls.brain.best_model

    def __init__(self, data, net, x_trunc,latent_idx, dt,z_gt,sys_name,output_dir,save_plots, criterion, optimizer, lr, weight_decay, iterations, lbfgs_steps, path, batch_size,
                 batch_size_test, print_every, save, callback, dtype, device):
        self.data = data
        self.net = net
        self.sys_name = sys_name
        self.output_dir = output_dir
        self.save_plots = save_plots
        self.x_trunc = x_trunc
        self.latent_idx = latent_idx
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

        # self.dataset = load_dataset(args)
        # self.dt = self.dataset.dt
        # self.dim_t = self.dataset.dim_t

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
        for i in range(self.iterations + 1):
            X_train, y_train = self.data.get_batch(self.batch_size)
            # print(X_train.shape) [100, 10]
            loss = self.__criterion(self.net(X_train), y_train)
            if i % self.print_every == 0 or i == self.iterations:
                X_test, y_test = self.data.get_batch_test(self.batch_size_test)
                loss_test = self.__criterion(self.net(X_test), y_test)
                # print('{:<9}Train loss: %.4e{:<25}Test loss: %.4e{:<25}'.format(i, loss.item(), loss_test.item()), flush=True)
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
                if self.callback is not None:
                    output = self.callback(self.data, self.net)
                    loss_history.append([i, loss.item(), loss_test.item(), *output])
                else:
                    loss_history.append([i, loss.item(), loss_test.item()])
            if i < self.iterations:
                self.__optimizer.zero_grad()
                loss.backward()
                self.__optimizer.step()
        self.loss_history = np.array(loss_history)
        # print('Done!', flush=True)
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
                X_train, y_train = self.data.get_batch(None)

                X_test, y_test = self.data.get_batch_test(None)
                loss = self.best_model.criterion(self.best_model(X_train), y_train)
                loss_test = self.best_model.criterion(self.best_model(X_test), y_test)
                # print('Train loss: {:<25}Test loss: {:<25}'.format(loss.item(), loss_test.item()), flush=True)
                it = self.it + 1
                if it % self.print_every == 0 or it == self.lbfgs_steps:
                    print('L-BFGS|| It: %05d, Loss: %.4e, Test: %.4e' %
                          (it, loss.item(), loss_test.item()))
                self.it = it
                if loss.requires_grad:
                    loss.backward()
                return loss

            optim.step(closure)
        print('Done!', flush=True)
        return self.best_model

    def output(self, data, best_model, loss_history, info, **kwargs):
        if self.path is None:
            path = './outputs/' + time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        else:
            path = './outputs/' + self.path
        if not os.path.isdir(path): os.makedirs(path)
        if data:
            def save_data(fname, data):
                if isinstance(data, dict):
                    np.savez_compressed(path + '/' + fname, **data)
                else:
                    np.save(path + '/' + fname, data)

            save_data('X_train', self.data.X_train_np)
            save_data('y_train', self.data.y_train_np)
            save_data('X_test', self.data.X_test_np)
            save_data('y_test', self.data.y_test_np)
        if best_model:
            torch.save(self.best_model, path + '/model_best.pkl')
        if loss_history:
            np.savetxt(path + '/loss.txt', self.loss_history)
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
        self.device = self.device
        self.data.dtype = self.dtype
        self.net.device = self.device
        self.net.dtype = self.dtype
        self.__init_optimizer()
        self.__init_criterion()

    def __init_optimizer(self):
        if self.optimizer == 'adam':
            self.__optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
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
    def test(self, SAE, latent_idx):
        print("\n[SPNN Testing Started]\n")
        self.dim_t = self.x_trunc.shape[0]

        x_net = torch.zeros(self.x_trunc.shape)
        #dEdt_net, dSdt_net = torch.zeros(self.dim_t), torch.zeros(self.dim_t)

        # Initial conditions z(t=0)
        x_net[0, :] = self.x_trunc[0, :]
        x = self.x_trunc[0, :]

        for snapshot in range(self.dim_t - 1):
            # Structure-Preserving Neural Network
            x1_net = self.net(x)
            #dEdt, dSdt = self.SPNN.get_thermodynamics(x)

            # Save results and Time update
            x_net[snapshot + 1, :] = x1_net
            # dEdt_net[snapshot] = dEdt
            # dSdt_net[snapshot] = dSdt
            x = x1_net

        # Detruncate
        x_gfinn = torch.zeros([self.dim_t, SAE.dim_latent])
        x_gfinn[:, latent_idx] = x_net

        # Decode latent vector
        z_gfinn_norm = SAE.decode(x_gfinn)
        z_gfinn = SAE.denormalize(z_gfinn_norm)

        # Load Ground Truth and Compute MSE
        z_gt = self.z_gt
        print_mse(z_gfinn, z_gt, self.sys_name)

        # Plot results
        if (self.save_plots):
            #plot_name = 'SPNN Full Integration (Latent)'
            #plot_latent(x_net, self.x_trunc, dEdt_net, dSdt_net, self.dt, plot_name, self.output_dir, self.sys_name)
            plot_name = 'SPNN Full Integration'
            plot_results(z_gfinn, z_gt, self.dt, plot_name, self.output_dir, self.sys_name)

        print("\n[SPNN Testing Finished]\n")
