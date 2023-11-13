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
#             self.py_data = pickle.load(
#                 open(f"./data/database_1DBurgers_nmu64_nt300_nx101_tstop3.p", "rb"))
#             self.py_data = pickle.load(
#                 open(f"./data/database_1DBurgers_nmu64_nt400_nx301_tstop2.p", "rb"))
            self.py_data = pickle.load(open(f"./data/database_1DBurgers_nmu100_nt1000_nx601_tstop2.p", "rb"))
            self.dt = 0.002
            self.dx = 0.01
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
            #self.dt = 0.01
#             self.dt = 0.005
            self.dim_t = self.z.shape[0]
#             print(self.dim_t)
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
        elif (sys_name == 'GC_SVD'):
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
            self.dim_t = self.z.shape[1]
            self.dim_z = self.z.shape[2]
            self.num_traj = self.z.shape[0]
            self.len = self.dim_t - 1
            
            if device == 'gpu':
                self.z = self.z.to(torch.device("cuda"))
                self.dz = self.dz.to(torch.device("cuda"))
                
        elif (sys_name == 'VC_SPNN_SVD'):
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
            self.dim_t = self.z.shape[1]
            self.dim_z = self.z.shape[2]
            self.num_traj = self.z.shape[0]
            self.len = self.dim_t - 1

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
#     elif (sys_name == 'rolling_tire'):
#         sys_name = sys_name       
#         root_dir = os.path.join(dset_dir, 'database_' + sys_name )
#         #root_dir = os.path.join(dset_dir, 'database_' + sys_name + '_2') #reduced ratio 2
#         #root_dir = os.path.join(dset_dir, 'database_' + sys_name + '_4') #reduce ratio 4
    else:
        sys_name = sys_name
        root_dir = os.path.join(dset_dir, 'database_' + sys_name)

    # Create Dataset instance
    dataset = GroundTruthDataset(root_dir, sys_name,device,dtype)

    return dataset


def split_dataset(sys_name,total_snaps,data_type):
    # Train and test snapshots
    
    train_snaps = int(0.8 * total_snaps)
    
    # Random split
    indices = np.arange(total_snaps)
    np.random.shuffle(indices)
    path = './data/'
    
    train_indices = indices[:train_snaps]
    test_indices = indices[train_snaps:total_snaps]

    #torch.save(indices,path + '/GC_data_split_indices.p')

#     #all indices for tr data
#     train_indices = np.arange(total_snaps)
#     test_indices = train_indices
    #test_indices = indices[train_snaps:total_snaps]
    
#           # manual selection
#     indices_tmp = np.arange(total_snaps)
#     test_indices = np.arange(0, total_snaps, 5)
#     train_indices = np.setdiff1d(indices_tmp,test_indices)

    if sys_name == 'viscoelastic':
        if data_type == 'last':
            # first 90% indices for tr data
            train_snaps = int(0.9 * total_snaps)
            indices = np.arange(total_snaps)
            train_indices = indices[:train_snaps]
            test_indices = indices[train_snaps:total_snaps]
            
        elif data_type == 'last85':
            ## first 90% indices for tr data
            train_snaps = int(0.85 * total_snaps)
            indices = np.arange(total_snaps)
            train_indices = indices[:train_snaps]
            test_indices = indices[train_snaps:total_snaps]
            
        elif data_type == 'last80':
            ## first 90% indices for tr data
            train_snaps = int(0.8 * total_snaps)
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

        
#         indices  = torch.load(path + '/VC_data_split_indices.p')
#         train_indices = indices[:train_snaps]
#         test_indices = indices[train_snaps:total_snaps]
        
#          # all indices for tr data
#         train_indices = np.arange(total_snaps)
#         #test_indices = train_indices
#         test_indices = indices[train_snaps:total_snaps]

#         ##first 80% indices for tr data
#         indices = np.arange(total_snaps)
#         train_indices = indices[:train_snaps]
#         test_indices = indices[train_snaps:total_snaps]

        
        
        
        ##Half of even/odd snapshots
#         indices = np.arange(total_snaps)
#         train_indices = indices[::2]
#         test_indices = indices[1::2]

    elif sys_name == 'GC':
        
        if data_type == 'last':
            
        #first 98 % as training
            train_snaps = int(0.98 * total_snaps)
            indices = np.arange(total_snaps)
            train_indices = indices[:train_snaps]
            test_indices = indices[train_snaps:total_snaps]
            
        elif data_type == 'last3':
            
        #first 97 % as training
            train_snaps = int(0.97 * total_snaps)
            indices = np.arange(total_snaps)
            train_indices = indices[:train_snaps]
            test_indices = indices[train_snaps:total_snaps]
            
        elif data_type == 'last2p5':
            
        #first 97 % as training
            train_snaps = int(0.975 * total_snaps)
            indices = np.arange(total_snaps)
            train_indices = indices[:train_snaps]
            test_indices = indices[train_snaps:total_snaps]
            
        elif data_type == 'middle':
        #first 49 %, last 49 % as training
            train_snaps_part1_end = int(0.49 * total_snaps)
            test_end = int(0.51 * total_snaps)
            indices = np.arange(total_snaps)

            train_indices1 = indices[:train_snaps_part1_end]
            train_indices2 = indices[test_end:]
            train_indices = np.concatenate((train_indices1, train_indices2))
            test_indices = indices[train_snaps_part1_end:test_end]


#         indices  = torch.load(path + '/GC_data_split_indices.p')
#         train_indices = indices[:train_snaps]
#         test_indices = indices[train_snaps:total_snaps]
        
#         #first 90 % as training
#         train_snaps = int(0.9 * total_snaps)
#         indices = np.arange(total_snaps)
#         train_indices = indices[:train_snaps]
#         test_indices = indices[train_snaps:total_snaps]

#         #first 98 % as training
#         train_snaps = int(0.98 * total_snaps)
#         indices = np.arange(total_snaps)
#         train_indices = indices[:train_snaps]
#         test_indices = indices[train_snaps:total_snaps]

#         #first 45 %, last 45 % as training
#         train_snaps_part1_end = int(0.45 * total_snaps)
#         test_end = int(0.55 * total_snaps)
#         indices = np.arange(total_snaps)
        
#         train_indices1 = indices[:train_snaps_part1_end]
#         train_indices2 = indices[test_end:]
#         train_indices = np.concatenate((train_indices1, train_indices2))
#         test_indices = indices[train_snaps_part1_end:test_end]

#         #first 46 %, last 46 % as training
#         train_snaps_part1_end = int(0.46 * total_snaps)
#         test_end = int(0.54 * total_snaps)
#         indices = np.arange(total_snaps)
        
#         train_indices1 = indices[:train_snaps_part1_end]
#         train_indices2 = indices[test_end:]
#         train_indices = np.concatenate((train_indices1, train_indices2))
#         test_indices = indices[train_snaps_part1_end:test_end]
        
#         #first 47 %, last 47 % as training
#         train_snaps_part1_end = int(0.47 * total_snaps)
#         test_end = int(0.53 * total_snaps)
#         indices = np.arange(total_snaps)
        
#         train_indices1 = indices[:train_snaps_part1_end]
#         train_indices2 = indices[test_end:]
#         train_indices = np.concatenate((train_indices1, train_indices2))
#         test_indices = indices[train_snaps_part1_end:test_end]
        

        
        
#         #first 49 %, last 49 % as training
#         train_snaps_part1_end = int(0.49 * total_snaps)
#         test_end = int(0.51 * total_snaps)
#         indices = np.arange(total_snaps)
        
#         train_indices1 = indices[:train_snaps_part1_end]
#         train_indices2 = indices[test_end:]
#         train_indices = np.concatenate((train_indices1, train_indices2))
#         test_indices = indices[train_snaps_part1_end:test_end]
        
#         print(train_indices.shape)
#         print(test_indices.shape)
#         print(train_indices)
#         print(test_indices)

#         #first 45 %, last 45 % as training
#         train_snaps_part1_end = int(0.45 * total_snaps)
#         test_end = int(0.55 * total_snaps)
#         indices = np.arange(total_snaps)
        
#         train_indices1 = indices[:train_snaps_part1_end]
#         train_indices2 = indices[test_end:]
#         train_indices = np.concatenate((train_indices1, train_indices2))
#         test_indices = indices[train_snaps_part1_end:test_end]


#         #first 95 % as training
#         train_snaps = int(0.95 * total_snaps)
#         indices = np.arange(total_snaps)
#         train_indices = indices[:train_snaps]
#         test_indices = indices[train_snaps:total_snaps]

#         # random selection 60%
#         train_snaps = int(0.6 * total_snaps)
#         indices = np.arange(total_snaps)
#         np.random.shuffle(indices)
#         train_indices = indices[:train_snaps]
#         test_indices = indices[train_snaps:total_snaps]
        
#         # random selection 70%
#         train_snaps = int(0.7 * total_snaps)
#         indices = np.arange(total_snaps)
#         np.random.shuffle(indices)
#         train_indices = indices[:train_snaps]
#         test_indices = indices[train_snaps:total_snaps]
        
#         #first 80 % as training
#         train_snaps = int(0.8 * total_snaps)
#         indices = np.arange(total_snaps)
#         train_indices = indices[:train_snaps]
#         test_indices = indices[train_snaps:total_snaps]

        
#         #random selection 80%
#         train_indices = indices[:train_snaps]
#         test_indices = indices[train_snaps:total_snaps]
        
        ##Half of even/odd snapshots
#         indices = np.arange(total_snaps)
#         train_indices = indices[::2]
#         test_indices = indices[1::2]

    elif (sys_name == '1DBurgers'):
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

    
    elif sys_name == 'rolling_tire':
        
          ## manual selection
#         indices_tmp = np.arange(total_snaps)
#         test_indices = np.arange(0, total_snaps, 5)
#         train_indices = np.setdiff1d(indices_tmp,test_indices)
          
# #           # all indices for tr data
#         train_indices = np.arange(total_snaps)
#         #test_indices = train_indices
#         test_indices = indices[train_snaps:total_snaps]
        
#         # random selection 90%
#         train_snaps = int(0.9 * total_snaps)
#         train_indices = indices[:train_snaps]
#         test_indices = indices[train_snaps:total_snaps]
        
#         #random selection 80%
#         train_indices = indices[:train_snaps]
#         test_indices = indices[train_snaps:total_snaps]

        # first 90% indices for tr data
        train_snaps = int(0.9 * total_snaps)
        indices = np.arange(total_snaps)
        train_indices = indices[:train_snaps]
        test_indices = indices[train_snaps:total_snaps]

    elif sys_name == 'GC_SVD':
#          ##       random selection 80%
#         train_indices = indices[:train_snaps]
#         test_indices = indices[train_snaps:total_snaps]
        
# #         ## manual selection
#         indices  = np.arange(total_snaps)
#         train_indices = indices[:train_snaps]
#         #train_indices = indices
#         test_indices = indices[train_snaps:total_snaps]
        
#                 # manual selection 60%
#         train_snaps = int(0.6 * total_snaps)
#         train_indices = indices[:train_snaps]
#         test_indices = indices[train_snaps:total_snaps]

        # manual selection 10%
#         train_snaps = int(0.1 * total_snaps)
#         train_indices = indices[:train_snaps]
#         test_indices = indices[train_snaps:total_snaps]
        
#         # manual selection 20%
#         train_snaps = int(0.2 * total_snaps)
#         train_indices = indices[:train_snaps]
#         test_indices = indices[train_snaps:total_snaps]

#        # manual selection 50%
        train_snaps = int(0.5 * total_snaps)
        train_indices = indices[:train_snaps]
        test_indices = indices[train_snaps:total_snaps]
        
# #         random selection 3%
#         train_snaps = int(0.03 * total_snaps)
#         train_indices = indices[:train_snaps]
#         test_indices = indices[train_snaps:total_snaps]
        
#         random selection 60%
#         train_snaps = int(0.6 * total_snaps)
#         indices = np.arange(total_snaps)
#         np.random.shuffle(indices)
#         train_indices = indices[:train_snaps]
#         test_indices = indices[train_snaps:total_snaps]
                               
#  #         manual selection 80%
#         train_snaps = int(0.8 * total_snaps)
#         indices = np.arange(total_snaps)
#         train_indices = indices[:train_snaps]
#         test_indices = indices[train_snaps:total_snaps]
        
            ##all indices for tr data
#         train_indices = np.arange(total_snaps)
#         test_indices = train_indices
        #test_indices = indices[train_snaps:total_snaps]
    elif sys_name == 'VC_SPNN_SVD':
                #random selection 80%
#         train_indices = indices[:train_snaps]
#         test_indices = indices[train_snaps:total_snaps]
        
        # random selection 50%
        train_snaps = int(0.5 * total_snaps)
        indices = np.arange(total_snaps)
        np.random.shuffle(indices)
        train_indices = indices[:train_snaps]
        test_indices = indices[train_snaps:total_snaps]
        
#         ## manual selection
#         indices  = np.arange(total_snaps)
#         train_indices = indices[:train_snaps]
#         #train_indices = indices
#         test_indices = indices[train_snaps:total_snaps]
        
            ##all indices for tr data
#         train_indices = np.arange(total_snaps)
#         test_indices = train_indices
        #test_indices = indices[train_snaps:total_snaps]
    
#     print(indices.shape)
#     print(train_indices.shape)
#     print(test_indices.shape)
    #print(test_indices)
    
    return train_indices, test_indices