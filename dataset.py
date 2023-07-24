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
            if args.dtype == 'double':
                self.z = torch.from_numpy(self.py_data['data'][10]['x']).double()
            elif args.dtype == 'float':
                self.z = torch.from_numpy(self.py_data['data'][10]['x']).float()
                
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
            if args.dtype == 'double':
                self.z = torch.from_numpy(self.mat_data['Z']).double()
                self.dz = torch.from_numpy(self.mat_data['dZ']).double()
            elif args.dtype == 'float':
                self.z = torch.from_numpy(self.mat_data['Z']).float()
                self.dz = torch.from_numpy(self.mat_data['dZ']).float()
            # Extract relevant dimensions and lengths of the problem
            self.dt = self.mat_data['dt'][0, 0]
            self.dim_t = self.z.shape[0]
            self.dim_z = self.z.shape[1]
            self.len = self.dim_t - 1

            if args.device == 'gpu':
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
    elif (args.sys_name == 'rolling_tire'):
        sys_name = args.sys_name       
        #root_dir = os.path.join(args.dset_dir, 'database_' + sys_name)
        #root_dir = os.path.join(args.dset_dir, 'database_' + sys_name + '_2') #reduced ratio 2
        root_dir = os.path.join(args.dset_dir, 'database_' + sys_name + '_4') #reduce ratio 4
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
    path = './data/'

    #torch.save(indices,path + '/VC_data_split_indices.p')

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
        indices_tmp = np.arange(total_snaps)
        test_indices = np.arange(0, total_snaps, 5)
        train_indices = np.setdiff1d(indices_tmp,test_indices)

    return train_indices, test_indices