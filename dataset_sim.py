"""dataset_sim.py"""

import os
import numpy as np
import scipy.io

import torch
from torch.utils.data import Dataset
import pickle


class GroundTruthDataset(Dataset):
    def __init__(self, root_dir, sys_name, device, dtype):
        # Load Ground Truth simulations from Matlab
                    
        self.mat_data = scipy.io.loadmat(root_dir)

        # Load state variables
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

    sys_name = sys_name
    root_dir = os.path.join(dset_dir, 'database_' + sys_name)

    # Create Dataset instance
    dataset = GroundTruthDataset(root_dir, sys_name,device,dtype)

    return dataset


def split_dataset(sys_name,total_snaps,data_type):
    
    train_snaps = int(0.8 * total_snaps)
    
    # Random split
    indices = np.arange(total_snaps)
    np.random.shuffle(indices)
    path = './data/'
    
    train_indices = indices[:train_snaps]
    test_indices = indices[train_snaps:total_snaps]


    if sys_name == 'viscoelastic':
        if data_type == 'last':
            # first 90% indices for tr data
            train_snaps = int(0.9 * total_snaps)
            indices = np.arange(total_snaps)
            train_indices = indices[:train_snaps]
            test_indices = indices[train_snaps:total_snaps]

        elif data_type == 'middle':
            #first 45 %, last 45 % as training
            train_snaps_part1_end = int(0.45 * total_snaps)
            test_end = int(0.55 * total_snaps)
            indices = np.arange(total_snaps)

            train_indices1 = indices[:train_snaps_part1_end]
            train_indices2 = indices[test_end:]
            train_indices = np.concatenate((train_indices1, train_indices2))
            test_indices = indices[train_snaps_part1_end:test_end]


    elif sys_name == 'GC':
        
        if data_type == 'last':
            
        #first 98 % as training data
            train_snaps = int(0.98 * total_snaps)
            indices = np.arange(total_snaps)
            train_indices = indices[:train_snaps]
            test_indices = indices[train_snaps:total_snaps]
            
        elif data_type == 'last10':
            
        #first 90 % as training data
            train_snaps = int(0.90 * total_snaps)
            indices = np.arange(total_snaps)
            train_indices = indices[:train_snaps]
            test_indices = indices[train_snaps:total_snaps]
            
        elif data_type == 'last5':
            
        #first 95 % as training dataq
            train_snaps = int(0.95 * total_snaps)
            indices = np.arange(total_snaps)
            train_indices = indices[:train_snaps]
            test_indices = indices[train_snaps:total_snaps]
            
        elif data_type == 'middle':
        #first 49 %, last 49 % as training data
            train_snaps_part1_end = int(0.49 * total_snaps)
            test_end = int(0.51 * total_snaps)
            indices = np.arange(total_snaps)

            train_indices1 = indices[:train_snaps_part1_end]
            train_indices2 = indices[test_end:]
            train_indices = np.concatenate((train_indices1, train_indices2))
            test_indices = indices[train_snaps_part1_end:test_end]


    return train_indices, test_indices