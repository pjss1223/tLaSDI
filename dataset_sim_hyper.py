"""dataset.py"""

import os
import numpy as np
import scipy.io

import torch
from torch.utils.data import Dataset
import pickle
import matplotlib.pyplot as plt


def preprocess_data(data, vel):
    for i in range(len(data['data'])):
        if vel == 1:
            data['data'][i]['x'] = data['data'][i].pop('u')
            data['data'][i]['dx'] = data['data'][i].pop('du')
            data['data'][i].pop('v')
            data['data'][i].pop('dv')
        elif vel == 2:
            data['data'][i]['x'] = data['data'][i].pop('v')
            data['data'][i]['dx'] = data['data'][i].pop('dv')
            data['data'][i].pop('u')
            data['data'][i].pop('du')
        elif vel == 3:
            data['data'][i]['x'] = np.hstack((data['data'][i]['u'], data['data'][i]['v']))
            data['data'][i]['dx'] = np.hstack((data['data'][i]['du'], data['data'][i]['dv']))
    return data


class GroundTruthDataset(Dataset):
    def __init__(self, root_dir, sys_name, device, dtype):
        # Load Ground Truth simulations from Matlab

        if (sys_name == '1DBurgers'):
            # Load Ground Truth simulations from python
            #All data---------------------------------------------------------------------------
            # self.py_data = pickle.load(
            #     open(f"./data/database_1DBurgers.p", "rb"))
            #self.py_data = pickle.load(open(f"./data/database_1DBurgers_nmu64_nt400_nx301_tstop2.p", "rb"))
#             self.dt = 0.005
#             self.dx = 0.02
            self.py_data = pickle.load(open(f"./data/database_1DBurgers_nmu100_nt1000_nx601_tstop2.p", "rb"))
            self.dt = 0.002
            self.dx = 0.01
            
                #open(f"./data/database_1DBurgers_nmu64_nt300_nx101_tstop3.p", "rb"))
#             self.py_data = pickle.load(
#                 #open(f"./data/database_1DBurgers_nmu100_nt200_nx201.p", "rb"))
#                 open(f"./data/database_1DBurgers_nmu64_nt400_nx301_tstop2.p", "rb"))
#             # Load state variables
            #self.z = torch.from_numpy(self.py_data['data'][10]['x']).float()
            
            
            
            if dtype == 'double':
                self.z1 = torch.from_numpy(self.py_data['data'][0]['x']).double()
                self.dz = torch.from_numpy(self.py_data['data'][0]['dx']).double()
                self.mu = torch.from_numpy(np.array(self.py_data['param'])).double()
            elif dtype == 'float':
                self.z1 = torch.from_numpy(self.py_data['data'][0]['x']).float()
                self.dz = torch.from_numpy(self.py_data['data'][0]['dx']).float()
                self.mu = torch.from_numpy(np.array(self.py_data['param'])).float()

#                 t_vec = torch.from_numpy(self.py_data['data'][0]['t']).float()
                
#                 plot_name = 'test dx'
#                 fig, axes = plt.subplots(1,1, figsize=(5, 5))
#                 fig.suptitle(plot_name)

#                 axes.plot(t_vec, self.dz[:,150].detach().cpu())
#                 axes.set_ylabel('$dx$ [-]')
#                 axes.set_xlabel('$t$ [s]')
#                 axes.grid()

#                 save_dir = os.path.join('outputs', plot_name)
#                 plt.savefig(save_dir)
#                 plt.clf()
                
#                 plot_name = 'test x'
#                 fig, axes = plt.subplots(1,1, figsize=(5, 5))
#                 fig.suptitle(plot_name)

#                 axes.plot(t_vec, self.z1[:,150].detach().cpu())
#                 axes.set_ylabel('$dx$ [-]')
#                 axes.set_xlabel('$t$ [s]')
#                 axes.grid()

#                 save_dir = os.path.join('outputs', plot_name)
#                 plt.savefig(save_dir)
#                 plt.clf()


                

            #print(self.dz.shape) #101 101

            # Extract relevant dimensions and lengths of the problem
            #self.dt = 0.01

            
#             print(self.z1.shape) #401 301
            
            self.dim_t = self.z1.shape[0]
            self.dim_z = self.z1.shape[1]
            self.len = self.dim_t - 1
            self.dim_mu = self.mu.shape[1]
            
            if device == 'gpu':
                self.dz = self.dz.to(torch.device("cuda"))
                self.mu = self.mu.to(torch.device("cuda"))
            
        elif (sys_name == '2DBurgers'):
            # Load Ground Truth simulations from python
            #All data---------------------------------------------------------------------------
            vel = 3 # 1 u 2 v 3 u and v
            self.py_data = pickle.load(open(f"./data/database_2DBurgers_nmu64_nt100_nx40_tstop1.p", "rb"))            
            self.py_data = preprocess_data(self.py_data, vel)
            # Extract relevant dimensions and lengths of the problem
            self.dt = 0.01
            self.dx = 0.15
            self.nx = 40
            self.tstop = 2
            
            self.Re = 10000
            
            
            if dtype == 'double':
                self.z1 = torch.from_numpy(self.py_data['data'][10]['x']).double()
                self.dz = torch.from_numpy(self.py_data['data'][10]['dx']).double()
                self.mu = torch.from_numpy(np.array(self.py_data['param'])).double()
            elif dtype == 'float':
                self.z1 = torch.from_numpy(self.py_data['data'][10]['x']).float()
                self.dz = torch.from_numpy(self.py_data['data'][10]['dx']).float()
                self.mu = torch.from_numpy(np.array(self.py_data['param'])).float()


                        
            self.dim_t = self.z1.shape[0]
            self.dim_z = self.z1.shape[1]
            self.len = self.dim_t - 1
            self.dim_mu = self.mu.shape[1]

            if device == 'gpu':
                self.dz = self.dz.to(torch.device("cuda"))
                self.mu = self.mu.to(torch.device("cuda"))
         #   --------------------------------------------------------------------------------------------

        #     # HALF of data -------------------------------------------------------------------------------
        #     # Load Ground Truth simulations from python
        #     self.py_data = pickle.load(
        #         open(f"./data/database_1DBurgers.p", "rb"))
        #
        #     # Load state variables
        #     #self.z = torch.from_numpy(self.py_data['data'][10]['x']).float()
        #     self.z1 = torch.from_numpy(self.py_data['data'][0]['x']).double()
        #     self.dz = torch.from_numpy(self.py_data['data'][0]['dx']).double()
        #     self.mu = torch.from_numpy(np.array(self.py_data['param'])).double()
        #
        #     self.z1 = self.z1[0::2,0::2]
        #     self.dz = self.dz[0::2,0::2]
        #
        #     # Extract relevant dimensions and lengths of the problem
        #     self.dt = 0.002
        #     self.dim_t = self.z1.shape[0]
        #     self.dim_z = self.z1.shape[1]
        #     self.len = self.dim_t - 1
        #     self.dim_mu = self.mu.shape[1]
        #
        #
        #     if device == 'gpu':
        #         #self.z = self.z.to(torch.device("cuda"))
        #         self.dz = self.dz.to(torch.device("cuda"))
        #         self.mu = self.mu.to(torch.device("cuda"))
        #     #-----------------------------------------------------------------------------------------------------------
        else:
            self.mat_data = scipy.io.loadmat(root_dir)

            # Load state variables
            #self.z = torch.from_numpy(self.mat_data['Z']).float()
            
            
            if dtype == 'double':
                self.z = torch.from_numpy(self.mat_data['Z']).double()
                self.dz = torch.from_numpy(self.mat_data['dZ']).double()
            elif dtype == 'float':
                self.z = torch.from_numpy(self.mat_data['Z']).float()
                self.dz = torch.from_numpy(self.mat_data['dZ']).float()

            # Extract relevant dimensions and lengths of the problem
            self.dt = self.mat_data['dt'][0, 0]
            self.dim_t = self.z.shape[0]
            self.dim_z = self.z.shape[1]
            self.len = self.dim_t - 1

            if device == 'gpu':
                self.z = self.z.to(torch.device("cuda"))
                self.dz = self.dz.to(torch.device("cuda"))

    def __getitem__(self, snapshot):
        z = self.z[snapshot, :]
        return z

    def __len__(self):
        return self.len


def load_dataset(sys_name,dset_dir,device,dtype):
    # Dataset directory path
    if (sys_name == '1DBurgers'):
        sys_name = sys_name
        root_dir = os.path.join(dset_dir, 'database_' + sys_name + '.p')#not needed
    else:
        sys_name = sys_name
        root_dir = os.path.join(dset_dir, 'database_' + sys_name)

    # Create Dataset instance
    dataset = GroundTruthDataset(root_dir, sys_name,device,dtype)

    return dataset


def split_dataset(sys_name,total_snaps):
    # Train and test snapshots
    train_snaps = int(0.8 * total_snaps)

    # Random split
    #indices = np.arange(total_snaps)
    #np.random.shuffle(indices)
    path = './data/'

    #torch.save(indices,path + '/VC_data_split_indices.p')

    if sys_name == 'viscoelastic':
        indices  = torch.load(path + '/VC_data_split_indices.p')

    elif sys_name == '1DBurgers':
        indices = torch.load(path + '/BG_data_split_indices.p')



    train_indices = indices[:train_snaps]
    test_indices = indices[train_snaps:total_snaps]

    return train_indices, test_indices