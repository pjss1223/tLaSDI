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

            self.py_data = pickle.load(open(f"./data/subsampled21_database_1DBurgers_nmu441_nt200_nx201_tstop2_aw1.p", "rb"))
            self.dt = 0.01
            self.dx = 0.03
            
            if dtype == 'double':
                self.z1 = torch.from_numpy(self.py_data['data'][0]['x']).double()
                self.dz = torch.from_numpy(self.py_data['data'][0]['dx']).double()
                self.mu = torch.from_numpy(np.array(self.py_data['param'])).double()
            elif dtype == 'float':
                self.z1 = torch.from_numpy(self.py_data['data'][0]['x']).float()
                self.dz = torch.from_numpy(self.py_data['data'][0]['dx']).float()
                self.mu = torch.from_numpy(np.array(self.py_data['param'])).float()
            
            self.dim_t = self.z1.shape[0]
            self.dim_z = self.z1.shape[1]
            self.len = self.dim_t - 1
            self.dim_mu = self.mu.shape[1]
            
            if device == 'gpu':
                self.dz = self.dz.to(torch.device("cuda"))
                self.mu = self.mu.to(torch.device("cuda"))
                
        elif (sys_name == '1DHeat'):
            
            self.py_data = pickle.load(open(f"./data/database_1DHeat_nmu441_nt200_nx201_tstop2_aw1.p", "rb"))
            self.dt = 0.01
            self.dx = 0.03
            
            if dtype == 'double':
                self.z1 = torch.from_numpy(self.py_data['data'][0]['x']).double()
                self.dz = torch.from_numpy(self.py_data['data'][0]['dx']).double()
                self.mu = torch.from_numpy(np.array(self.py_data['param'])).double()
            elif dtype == 'float':
                self.z1 = torch.from_numpy(self.py_data['data'][0]['x']).float()
                self.dz = torch.from_numpy(self.py_data['data'][0]['dx']).float()
                self.mu = torch.from_numpy(np.array(self.py_data['param'])).float()
            

            self.dim_t = self.z1.shape[0]
            self.dim_z = self.z1.shape[1]
            self.len = self.dim_t - 1
            self.dim_mu = self.mu.shape[1]
            
            if device == 'gpu':
                self.dz = self.dz.to(torch.device("cuda"))
                self.mu = self.mu.to(torch.device("cuda"))
            


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
    elif (sys_name == '1DHeat'):
        sys_name = sys_name
        root_dir = os.path.join(dset_dir, 'database_' + sys_name + '.p')#not needed
    else:
        sys_name = sys_name
        root_dir = os.path.join(dset_dir, 'database_' + sys_name)

    # Create Dataset instance
    dataset = GroundTruthDataset(root_dir, sys_name,device,dtype)

    return dataset


def split_dataset(sys_name,total_snaps,data_type):
    # Train and test snapshots
    
    if (sys_name == '1DBurgers'):
        if data_type == 'random':
            train_snaps = int(0.8 * total_snaps)
            # Random split
            indices = np.arange(total_snaps)
            np.random.shuffle(indices)

            train_indices = indices[:train_snaps]
            test_indices = indices[train_snaps:total_snaps]
        elif data_type == 'loaded':

            path = './data/'

            #torch.save(indices,path + '/VC_data_split_indices.p')

            if sys_name == 'viscoelastic':
                indices  = torch.load(path + '/VC_data_split_indices.p')

            elif sys_name == '1DBurgers':
                indices = torch.load(path + '/BG_data_split_indices.p')



    

    return train_indices, test_indices