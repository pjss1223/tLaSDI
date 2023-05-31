"""dataset.py"""

import os
import numpy as np
import scipy.io

import torch
from torch.utils.data import Dataset
import pickle


class GroundTruthDataset(Dataset):
    def __init__(self, root_dir, sys_name, device):
        # Load Ground Truth simulations from Matlab

        if (sys_name == '1DBurgers'):
            # Load Ground Truth simulations from python
            #All data---------------------------------------------------------------------------
            # self.py_data = pickle.load(
            #     open(f"./data/database_1DBurgers.p", "rb"))
            self.py_data = pickle.load(
                #open(f"./data/database_1DBurgers_nmu100_nt200_nx201.p", "rb"))
                open(f"./data/database_1DBurgers_nmu64_nt100_nx101.p", "rb"))
            # Load state variables
            #self.z = torch.from_numpy(self.py_data['data'][10]['x']).float()
            self.z1 = torch.from_numpy(self.py_data['data'][0]['x']).double()
            self.dz = torch.from_numpy(self.py_data['data'][0]['dx']).double()
            self.mu = torch.from_numpy(np.array(self.py_data['param'])).double()

            #print(self.dz.shape) #101 101

            # Extract relevant dimensions and lengths of the problem
            self.dt = 0.01
            self.dim_t = self.z1.shape[0]
            self.dim_z = self.z1.shape[1]
            self.len = self.dim_t - 1
            self.dim_mu = self.mu.shape[1]


            if device == 'gpu':
                #self.z = self.z.to(torch.device("cuda"))
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
            self.z = torch.from_numpy(self.mat_data['Z']).double()
            self.dz = torch.from_numpy(self.mat_data['dZ']).double()

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


def load_dataset(sys_name,dset_dir,device):
    # Dataset directory path
    if (sys_name == '1DBurgers'):
        sys_name = sys_name
        root_dir = os.path.join(dset_dir, 'database_' + sys_name + '.p')
    else:
        sys_name = sys_name
        root_dir = os.path.join(dset_dir, 'database_' + sys_name)

    # Create Dataset instance
    dataset = GroundTruthDataset(root_dir, sys_name,device)

    return dataset


def split_dataset(sys_name,total_snaps):
    # Train and test snapshots
    train_snaps = int(0.8 * total_snaps)

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