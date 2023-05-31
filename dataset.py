"""dataset.py"""

import os
import numpy as np
import scipy.io

import torch
from torch.utils.data import Dataset
import pickle

class GroundTruthDataset(Dataset):
    def __init__(self, root_dir,args):
        # Load Ground Truth simulations from Matlab
        self.device = args.device

        if (args.sys_name == '1DBurgers'):
            # Load Ground Truth simulations from python
            # self.py_data = pickle.load(open(f"/Users/sjpark/PycharmProjects/SAE_GFINNS/data/database_1DBurgers.p", "rb"))
            self.py_data = pickle.load(open(f"./data/database_1DBurgers.p", "rb"))

            #self.py_data = pickle.load(open(f" root_dir", "rb"))

            # Load state variables
            #self.z = torch.from_numpy(self.py_data['data'][10]['x']).float()
            self.z = torch.from_numpy(self.py_data['data'][10]['x']).double()
            #print(self.z.shape)
            # Extract relevant dimensions and lengths of the problem
            self.dt = 0.001
            self.dim_t = self.z.shape[0]
            self.dim_z = self.z.shape[1]
            self.len = self.dim_t - 1
            if self.device == 'gpu':
                self.z = self.z.to(torch.device("cuda"))
                self.dz = self.dz.to(torch.device("cuda"))
        else:
            self.mat_data = scipy.io.loadmat(root_dir)
        
            # Load state variables
            #self.z = torch.from_numpy(self.mat_data['Z']).float()
            self.z = torch.from_numpy(self.mat_data['Z']).double()
            # Extract relevant dimensions and lengths of the problem
            self.dt = self.mat_data['dt'][0,0]
            self.dim_t = self.z.shape[0]
            self.dim_z = self.z.shape[1]
            self.len = self.dim_t - 1
            if self.device == 'gpu':
                self.z = self.z.to(torch.device("cuda"))
                self.dz = self.dz.to(torch.device("cuda"))
    
    def __getitem__(self, snapshot):
        z = self.z[snapshot,:]
        return z

    def __len__(self):
        return self.len


def load_dataset(args):
    # Dataset directory path
    if (args.sys_name == '1DBurgers'):
        sys_name = args.sys_name
        root_dir = os.path.join(args.dset_dir, 'database_' + sys_name + '.p')
    else:
        sys_name = args.sys_name
        root_dir = os.path.join(args.dset_dir, 'database_' + sys_name)

    # Create Dataset instance
    dataset = GroundTruthDataset(root_dir,args)

    return dataset


def split_dataset(sys_name,total_snaps):
    # Train and test snapshots
    train_snaps = int(0.8*total_snaps)

    # Random split
    #indices = np.arange(total_snaps)
    #np.random.shuffle(indices)
    path = './outputs/'

    #torch.save(indices,path + '/VC_data_split_indices.p')

    if sys_name == 'viscoelastic':
        indices  = torch.load(path + '/VC_data_split_indices.p')

    elif sys_name == '1DBurgers':
        indices = torch.load(path + '/BG_data_split_indices.p')

    train_indices = indices[:train_snaps]
    test_indices = indices[train_snaps:total_snaps]

    return train_indices, test_indices