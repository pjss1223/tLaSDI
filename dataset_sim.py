"""dataset.py"""

import os
import numpy as np
import scipy.io

import torch
from torch.utils.data import Dataset
import pickle


class GroundTruthDataset(Dataset):
    def __init__(self, root_dir, sys_name, device, dtype):
        # Load Ground Truth simulations from Matlab

        if (sys_name == '1DBurgers'):
            # Load Ground Truth simulations from python
            print('Current GPU memory allocated before data: ', torch.cuda.memory_allocated() / 1024 ** 3, 'GB')
            self.py_data = pickle.load(
                open(f"./data/database_1DBurgers.p", "rb"))
            print('Current GPU memory allocated after data: ', torch.cuda.memory_allocated() / 1024 ** 3, 'GB')
            # self.py_data = pickle.load(open(f" root_dir", "rb"))

            # Load state variables
            #self.z = torch.from_numpy(self.py_data['data'][10]['x']).float()
            if dtype == 'double':
                self.z = torch.from_numpy(self.py_data['data'][10]['x']).double()
                self.dz = torch.from_numpy(self.py_data['data'][10]['dx']).double()
                self.mu = torch.from_numpy(np.array(self.py_data['param'])).double()
            elif dtype == 'float':
                self.z = torch.from_numpy(self.py_data['data'][10]['x']).float()
                self.dz = torch.from_numpy(self.py_data['data'][10]['dx']).float()
                self.mu = torch.from_numpy(np.array(self.py_data['param'])).float()

            #print(self.py_data['data'][1,3]['x'])
            # parameter indices: 0-255

            # Extract relevant dimensions and lengths of the problem
            self.dt = 0.001
            self.dim_t = self.z.shape[0]
            self.dim_z = self.z.shape[1]
            self.len = self.dim_t - 1

            # self.mu = self.mu[1]
            # self.mu_tmp = self.mu.T.repeat(self.dim_t,1)

            # print(self.mu_tmp.shape)
            # print(torch.cat((self.z, self.mu_tmp),1).shape)
            #print(self.mu.T.repeat(self.dim_t,1,1)[:,:,2].shape) #1001,2
            #print(torch.cat((self.z, self.mu_tmp),1).shape)# 1001,1003
            
            if device == 'gpu':
                self.z = self.z.to(torch.device("cuda"))
                self.dz = self.dz.to(torch.device("cuda"))
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
        root_dir = os.path.join(dset_dir, 'database_' + sys_name + '.p')# not necessary
    elif (sys_name == 'rolling_tire'):
        sys_name = sys_name       
        root_dir = os.path.join(dset_dir, 'database_' + sys_name )
        #root_dir = os.path.join(dset_dir, 'database_' + sys_name + '_2') #reduced ratio 2
        #root_dir = os.path.join(dset_dir, 'database_' + sys_name + '_4') #reduce ratio 4
    else:
        sys_name = sys_name
        root_dir = os.path.join(dset_dir, 'database_' + sys_name)

    # Create Dataset instance
    print('Current GPU memory allocated before data: ', torch.cuda.memory_allocated() / 1024 ** 3, 'GB')
    dataset = GroundTruthDataset(root_dir, sys_name,device,dtype)
    print('Current GPU memory allocated after data: ', torch.cuda.memory_allocated() / 1024 ** 3, 'GB')

    return dataset


def split_dataset(sys_name,total_snaps):
    # Train and test snapshots
    
    train_snaps = int(0.8 * total_snaps)
    
    # Random split
    indices = np.arange(total_snaps)
    np.random.shuffle(indices)
    path = './data/'

    #torch.save(indices,path + '/RT_data_split_indices.p')

    if sys_name == 'viscoelastic':
        indices  = torch.load(path + '/VC_data_split_indices.p')
        train_indices = indices[:train_snaps]
        test_indices = indices[train_snaps:total_snaps]

    elif sys_name == '1DBurgers':
        indices = torch.load(path + '/BG_data_split_indices.p')
        train_indices = indices[:train_snaps]
        test_indices = indices[train_snaps:total_snaps]

#     elif sys_name == 'rolling_tire':
#         indices = torch.load(path + '/RT_data_split_indices.p')



    
    if sys_name == 'rolling_tire':
        
          ## manual selection
#         indices_tmp = np.arange(total_snaps)
#         test_indices = np.arange(0, total_snaps, 5)
#         train_indices = np.setdiff1d(indices_tmp,test_indices)
          
          # all indices for tr data
        train_indices = np.arange(total_snaps)
        test_indices = train_indices
        
#         #random selection
#         train_indices = indices[:train_snaps]
#         test_indices = indices[train_snaps:total_snaps]
    
#     print(indices.shape)
#     print(train_indices.shape)
#     print(test_indices.shape)
    #print(test_indices)
    
    return train_indices, test_indices