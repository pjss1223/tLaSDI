"""solver.py"""

import os

import torch
import torch.optim as optim
import numpy as np

from model import SparseAutoEncoder, StackedSparseAutoEncoder
from dataset import load_dataset, split_dataset
from utilities.plot import plot_results, plot_latent_visco, plot_latent_tire
from utilities.utils import print_mse, all_latent
import matplotlib.pyplot as plt


class AE_Solver_jac(object):
    def __init__(self, args, AE_name,layer_vec_SAE,layer_vec_SAE_q,layer_vec_SAE_v,layer_vec_SAE_sigma):
        # Study Case
        self.sys_name = args.sys_name
        self.device = args.device

        # Dataset Parameters
        self.dset_dir = args.dset_dir
        self.dataset = load_dataset(args)
        self.dt = self.dataset.dt
        self.dim_t = self.dataset.dim_t
        self.trunc_period = args.trunc_period
        self.AE_name = AE_name
        self.dtype = args.dtype
        self.device = args.device
        self.data_type = args.data_type
        
#         self.num_para = 64
        self.batch_size = args.batch_size_AE
    
        self.path = self.sys_name + args.net + AE_name  

        self.train_snaps, self.test_snaps = split_dataset(self.sys_name, self.dim_t-1, self.data_type)
        if self.sys_name == '1DBurgers':
            self.train_snaps, self.test_snaps = split_dataset(self.sys_name, self.num_para,self.data_type)
            

        # Training Parameters
        self.max_epoch = args.max_epoch_SAE
        self.lambda_r = args.lambda_r_SAE
        self.lambda_jac = args.lambda_jac_SAE
        self.loss_history = None
        self.loss_history_recon = None
        self.loss_history_jac = None
        


        if self.sys_name == 'viscoelastic' or self.sys_name == 'GC':
            if args.dtype == 'double':
#                 print(layer_vec_SAE)
#                 print(args.activation_SAE)
                self.SAE = SparseAutoEncoder(layer_vec_SAE, args.activation_SAE).double()
#                 net_parameters = self.SAE.parameters()

#                 for param in net_parameters:
#                     print(param)

            elif args.dtype == 'float':
                self.SAE = SparseAutoEncoder(layer_vec_SAE, args.activation_SAE).float()
            

                
            if self.device == 'gpu':
                self.SAE = self.SAE.to(torch.device('cuda'))

        elif self.sys_name == '1DBurgers':
            if args.dtype == 'double':
                self.SAE = SparseAutoEncoder(layer_vec_SAE, args.activation_SAE).double()
            elif args.dtype == 'float':
                self.SAE = SparseAutoEncoder(layer_vec_SAE, args.activation_SAE).float()
                
            if self.device == 'gpu':
                self.SAE = self.SAE.to(torch.device('cuda'))
                
        elif self.sys_name == 'rolling_tire':
            if args.dtype == 'double':
                self.SAE = StackedSparseAutoEncoder(layer_vec_SAE_q, layer_vec_SAE_v, layer_vec_SAE_sigma,
                                                args.activation_SAE, args.dtype).double()
            elif args.dtype == 'float':
                self.SAE = StackedSparseAutoEncoder(layer_vec_SAE_q, layer_vec_SAE_v, layer_vec_SAE_sigma,
                                                args.activation_SAE, args.dtype).float()
                
            if self.device == 'gpu':
                self.SAE = self.SAE.to(torch.device('cuda'))

#         self.optim = optim.Adam(self.SAE.parameters(), lr=args.lr_SAE, weight_decay=1e-4)

        params = [
                {'params': self.SAE.parameters(), 'lr': args.lr_SAE, 'weight_decay':args.weight_decay_AE} #args.weight_decay_AE}
            ]
            
        self.optim = torch.optim.AdamW(params)

#         self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=args.miles_SAE,
                                                             # gamma=args.gamma_SAE)
        if self.sys_name == 'rolling_tire':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=args.miles_SAE,gamma=args.gamma_SAE)
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=args.miles_SAE, gamma=args.gamma_SAE)


        if (args.train_SAE == False):
            # Load pretrained nets
            load_name = self.AE_name+'_' + self.sys_name + '.pt'
            load_path = os.path.join(self.dset_dir, load_name)
            self.SAE.load_state_dict(torch.load(load_path))

        self.output_dir = args.output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self.save_plots = args.save_plots
        


    # Train AE Algorithm
    def train(self):

        print("\n[AE Training Started]\n")

        # Training data
        
        if self.sys_name == '1DBurgers':
            path = './data/'
            z_data = torch.load(path + '/1DBG_Z_data_para_400_300.p')
#             z_gt= z_data['z']
#             dz_gt = z_data['dz']
            z_gt= z_data['z_tr']
            dz_gt = z_data['dz_tr']
            if self.dtype == 'float':
                z_gt = z_gt.to(torch.float32)
                dz_gt = dz_gt.to(torch.float32)
            if self.device == 'gpu':
                z_gt = z_gt.to(torch.device("cuda"))
                dz_gt = dz_gt.to(torch.device("cuda"))

        else:
            z_gt = self.dataset.z[self.train_snaps, :]
            dz_gt = self.dataset.dz[self.train_snaps, :]

        

        epoch = 1
        loss_history_recon = []
        loss_history_jac = []
        loss_history = []
        

        
        while (epoch <= self.max_epoch):



            z_gt_norm = self.SAE.normalize(z_gt)
            z_gt_norm = z_gt_norm.requires_grad_(True)


            dz_gt_norm = self.SAE.normalize(dz_gt)
            dz_gt_norm = dz_gt_norm.requires_grad_(True)

            z_sae_norm, x = self.SAE(z_gt_norm)


            dz_train, dx_data_train,  idx_trunc = self.SAE.JVP_AE(z_gt_norm, x, dz_gt_norm,  self.trunc_period)
                


        
            loss_jac =  torch.mean((dz_train - dz_gt_norm[:,idx_trunc]) ** 2)

            

            # Loss function
            loss_reconst = torch.mean((z_sae_norm - z_gt_norm) ** 2)
#             print(loss_reconst)

            loss = self.lambda_r*loss_reconst+self.lambda_jac*loss_jac #+ self.lambda_r * loss_sparsity

            if epoch % 100 == 0 or epoch == self.max_epoch:
    
                loss_reconst_mean = loss_reconst.item() / len(self.train_snaps)
                loss_jac_mean = loss_jac.item() / len(self.train_snaps)
            

                print("Epoch [{}/{}], Reconst Loss: {:1.6e} (Train), Jacobian Loss: {:1.6e} (Train) "
                  .format(epoch, int(self.max_epoch), loss_reconst,loss_jac))
                

                if not os.path.exists('model'): os.mkdir('model')
                if self.path == None:
                    torch.save(self.SAE, 'model/AE_model{}.pkl'.format(epoch))
                else:
                    if not os.path.isdir('model/' + self.path): os.makedirs('model/' + self.path)
                    torch.save(self.SAE, 'model/{}/AE_model{}.pkl'.format(self.path, epoch))

#                 loss_history_recon.append([epoch, loss_reconst.item()])
#                 loss_history_jac.append([epoch, loss_jac.item()])
#                 loss_history.append([epoch, loss_reconst.item()+loss_jac.item()])
                
                loss_history.append([epoch, loss.item()])
                loss_history_recon.append([epoch, loss_reconst.item()])
                loss_history_jac.append([epoch, loss_jac.item()])
    
            # Backpropagation
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            self.scheduler.step()

            epoch += 1

        print("\n[SAE Training Finished]\n")
        print("[Train Set Evaluation]\n")
        

        
        z_gt_norm = self.SAE.normalize(z_gt)
        z_sae_norm, x = self.SAE(z_gt_norm)
        # Denormalize
        z_sae = self.SAE.denormalize(z_sae_norm)

        # Compute MSE
        
#         print(z_sae.shape)
#         print(z_gt.shape)

        print_mse(z_sae, z_gt, self.sys_name)

        # Save net
        file_name = self.AE_name + '_' + self.sys_name + '.pt'
        save_dir = os.path.join(self.output_dir, file_name)
        torch.save(self.SAE.state_dict(), save_dir)

        # Save loss plot
        self.loss_history = np.array(loss_history)
        self.loss_history_recon = np.array(loss_history_recon)
        self.loss_history_jac = np.array(loss_history_jac)



        loss_name = self.AE_name+ '_' + self.sys_name
        np.savetxt(os.path.join(self.output_dir, loss_name+'_loss.txt'), self.loss_history)
        p1, = plt.plot(self.loss_history[:, 0], self.loss_history[:, 1], '-')
        #plt.plot(self.loss_history[:, 0], self.loss_history[:, 2], '--')
        plt.legend(['train loss'])  # , '$\hat{u}$'])
        plt.yscale('log')
        plt.savefig(os.path.join(self.output_dir, loss_name+'_loss.png'))
        p1.remove()

        np.savetxt(os.path.join(self.output_dir, loss_name+'loss_recon.txt'), self.loss_history_recon)
        p2, = plt.plot(self.loss_history_recon[:, 0], self.loss_history_recon[:, 1], '-')
        #plt.plot(self.loss_history[:, 0], self.loss_history[:, 2], '--')
        plt.legend(['train loss (recon)'])  # , '$\hat{u}$'])
        plt.yscale('log')
        plt.savefig(os.path.join(self.output_dir, loss_name+'_loss_recon.png'))
        p2.remove()

        np.savetxt(os.path.join(self.output_dir, loss_name+'_loss_jac.txt'), self.loss_history_jac)
        p3, =plt.plot(self.loss_history_jac[:, 0], self.loss_history_jac[:, 1], '-')
        #plt.plot(self.loss_history[:, 0], self.loss_history[:, 2], '--')
        plt.legend(['train loss (jac)'])  # , '$\hat{u}$'])
        plt.yscale('log')
        plt.savefig(os.path.join(self.output_dir, loss_name+'_loss_jac.png'))
        p3.remove()
    # Test AE Algorithm
    def test(self):
        print("\n[AE Testing Started]\n")
        best_loss_index = np.argmin(self.loss_history[:, 1])
        iteration = int(self.loss_history[best_loss_index, 0])
        loss_train = self.loss_history[best_loss_index, 1]
        
        if self.path == None:
            self.best_model_AE = torch.load('model/AE_model{}.pkl'.format(iteration))
        else:
            self.best_model_AE = torch.load('model/{}/AE_model{}.pkl'.format(self.path, iteration))
                
        self.SAE = self.best_model_AE

        
        # Load data
        if self.sys_name == '1DBurgers':
            path = './data/'
            z_data = torch.load(path + '/1DBG_Z_data_para_400_300.p')
            z_gt= z_data['z']
            if self.dtype == 'float':
                z_gt = z_gt.to(torch.float32)
            if self.device == 'gpu':
                z_gt = z_gt.to(torch.device("cuda"))
        else:
            z_gt = self.dataset.z
            
        z_gt_norm = self.SAE.normalize(z_gt)

        # Forward pass
        z_sae_norm, _ = self.SAE(z_gt_norm)
        z_sae = self.SAE.denormalize(z_sae_norm)

        # Compute MSE
        
        print_mse(z_sae, z_gt, self.sys_name)
        
        print('error over test data')
        print(z_sae.shape)
        print(self.test_snaps)
        print_mse(z_sae[self.test_snaps,:], z_gt[self.test_snaps,:], self.sys_name)

        if (self.save_plots):
            plot_name = 'SAE Reduction Only'
            plot_results(z_sae, z_gt, self.dt, plot_name, self.output_dir, self.sys_name)

        print("\n[AE Testing Finished]\n")

    # Latent dimensionality detection
    def detect_dimensionality(self):
        # Load data
        if self.sys_name == '1DBurgers':
            path = './data/'
            z_data = torch.load(path + '/1DBG_Z_data_para_400_300.p')
            z= z_data['z']
            if self.dtype == 'float':
                z = z.to(torch.float32)
            if self.device == 'gpu':
                z = z.to(torch.device("cuda"))
        else:
            z = self.dataset.z

        z_norm = self.SAE.normalize(z)
        # Forward pass
        _, x = self.SAE(z_norm)

        if self.sys_name == 'viscoelastic':
            # Detect latent dimensionality
            x_trunc, latent_idx = all_latent(x)
            print('Latent Dimensionality: {}'.format(len(latent_idx)))

            # Plot latent variables
            if (self.save_plots == True):
                plot_name = '[VC] AE Latent Variables_'+self.AE_name
                plot_latent_visco(x, self.dataset.dt, plot_name, self.output_dir)

        elif self.sys_name == '1DBurgers':
            # Detect latent dimensionality
            x_trunc, latent_idx = all_latent(x)
            print('Latent Dimensionality: {}'.format(len(latent_idx)))

            # Plot latent variables
            if (self.save_plots == True):
                plot_name = '[1DBurgers] AE Latent Variables_'+self.AE_name
                plot_latent_visco(x, self.dataset.dt, plot_name, self.output_dir)

        elif self.sys_name == 'rolling_tire':
            x_q, x_v, x_sigma = self.SAE.split_latent(x)

            # Detect latent dimensionality
            x_q_trunc, latent_idx_q = all_latent(x_q)
            x_v_trunc, latent_idx_v = all_latent(x_v)
            x_sigma_trunc, latent_idx_sigma = all_latent(x_sigma)
            print('Latent Dimensionality:')
            print('  Position: {}'.format(len(latent_idx_q)))
            print('  Velocity: {}'.format(len(latent_idx_v)))
            print('  Stress Tensor: {}'.format(len(latent_idx_sigma)))

            x_trunc = torch.cat((x_q_trunc, x_v_trunc, x_sigma_trunc), 1)
            #print(x_trunc.shape)
            latent_idx_v = latent_idx_v + self.SAE.dim_latent_q
            latent_idx_sigma = latent_idx_sigma + self.SAE.dim_latent_q + self.SAE.dim_latent_v
            latent_idx = list(latent_idx_q) + list(latent_idx_v) + list(latent_idx_sigma)

            # Plot latent variables
            if (self.save_plots == True):
                plot_name = '[RT] AE Latent Variables_'+self.AE_name
                plot_latent_tire(x_q, x_v, x_sigma, self.dataset.dt, plot_name, self.output_dir)

        return x_trunc, latent_idx


if __name__ == '__main__':
    pass


