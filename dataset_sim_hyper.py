"""dataset.py"""

import os
import numpy as np
import scipy.io

import torch
from torch.utils.data import Dataset
import pickle
import matplotlib.pyplot as plt


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
