"""model.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import functorch
from functorch import vmap, jacrev
from learner.utils import mse, wasserstein, div, grad
import numpy as np
from torch.utils import bottleneck


class SparseAutoEncoder(nn.Module):
    """Sparse Autoencoder"""

    def __init__(self, layer_vec, activation):
        super(SparseAutoEncoder, self).__init__()
        self.layer_vec = layer_vec
        self.dim_latent = layer_vec[-1]
        self.activation = activation
        self.activation_vec = ['linear'] + (len(self.layer_vec) - 3) * [self.activation] + ['linear']
        # self.activation_vec = ['relu'] + (len(self.layer_vec)-3)*[self.activation] + ['relu']
        # self.activation_vec = ['relu'] + (len(self.layer_vec)-3)*[self.activation] + ['linear']
        # self.activation_vec = ['linear'] + (len(self.layer_vec)-3)*[self.activation] + ['relu']

        # Encode
        self.steps = len(self.layer_vec) - 1
        self.fc_encoder = nn.ModuleList()
        for k in range(self.steps):
            self.fc_encoder.append(nn.Linear(self.layer_vec[k], self.layer_vec[k + 1]))

        # Decode
        self.fc_decoder = nn.ModuleList()
        for k in range(self.steps):
            self.fc_decoder.append(nn.Linear(self.layer_vec[self.steps - k], self.layer_vec[self.steps - k - 1]))

    def activation_function(self, x, activation):
        if activation == 'linear':
            x = x
        elif activation == 'sigmoid':
            x = torch.sigmoid(x)
        elif activation == 'relu':
            x = F.relu(x)
        elif activation == 'rrelu':
            x = F.rrelu(x)
        elif activation == 'tanh':
            x = torch.tanh(x)
        elif activation == 'elu':
            x = F.elu(x)
        else:
            raise NotImplementedError
        return x

    # Encoder
    def encode(self, x):
        #print(x.shape)
        idx = 0
        for layer in self.fc_encoder:
            # print(x.shape)
            x = layer(x)
            x = self.activation_function(x, self.activation_vec[idx])
            idx += 1
        return x

    # Decoder
    def decode(self, x):
        idx = 0
        for layer in self.fc_decoder:
            x = layer(x)
            x = self.activation_function(x, self.activation_vec[idx])
            idx += 1
        return x



    def jacobian_norm_trunc(self, z, x, trunc_period):


        # print(z.shape)#[119, 400]

        dim_z = z.shape[1]

        idx_trunc = range(0, dim_z - 1, trunc_period)  # 3 for VC, 10 for BG

        def decode_trunc(xx):
            idx = 0
            for layer in self.fc_decoder:
                xx = layer(xx)
                xx = self.activation_function(xx, self.activation_vec[idx])
                idx += 1

            return xx[idx_trunc]

        J_e_func = vmap(lambda x: jacrev(self.encode, argnums=0)(x)[:, idx_trunc], in_dims=(0))
        # J_e_func = vmap(jacrev(self.encode, argnums=0), in_dims=(0))
        
        # print(J_e.shape)
        J_d_func = vmap(jacrev(decode_trunc, argnums=0), in_dims=(0))
        

        #with torch.no_grad():
        J_e = J_e_func(z)
        J_d = J_d_func(x)
        # print(J_d.shape)
        
        #        ##----------further batch
    
#         chunk_size = int(x.shape[0]/10)
        
#         #print(chunk_size)

#         # Create chunks of x and z
#         x_chunks = torch.chunk(x, chunk_size, dim=0)
#         z_chunks = torch.chunk(z, chunk_size, dim=0)
        
#         #print(x_chunks[0].shape)


#         # Compute J_e using batches
#         J_e_func = vmap(lambda x: jacrev(self.encode, argnums=0)(x)[:, idx_trunc], in_dims=(0))

#         J_e_chunks = []
#         for z_chunk in z_chunks:
#             #with torch.no_grad():
#             J_e_chunk = J_e_func(z_chunk)
#             J_e_chunks.append(J_e_chunk)
#         J_e = torch.cat(J_e_chunks, dim=0)


#         # Compute J_d using batches
#         J_d_func = vmap(jacrev(decode_trunc, argnums=0), in_dims=(0))

#         J_d_chunks = []
#         for x_chunk in x_chunks:
#             #with torch.no_grad():
#             J_d_chunk = J_d_func(x_chunk)
#             J_d_chunks.append(J_d_chunk)
#         J_d = torch.cat(J_d_chunks, dim=0)
               
        
        
        J_ed = J_d @ J_e
        

        loss_jacobian = torch.mean(torch.pow(J_ed.diagonal(dim1=-2, dim2=-1).sub_(1), 2))
        
        return loss_jacobian, J_e, J_d, idx_trunc
        

    def jacobian_norm_trunc_gpu(self, z, x,trunc_period):

        dim_z = z.shape[1]

        idx_trunc = range(0, dim_z - 1, trunc_period)  # 3 for VC, 10 for BG
        #print('Current GPU memory allocated part1: ', torch.cuda.memory_allocated() / 1024 ** 3, 'GB')
        def decode_trunc(xx):
            idx = 0
            for layer in self.fc_decoder:
                xx = layer(xx)
                xx = self.activation_function(xx, self.activation_vec[idx])
                idx += 1

            return xx[idx_trunc]

        
#           #--------no further batch
        J_e_func = vmap(lambda x: jacrev(self.encode, argnums=0)(x)[:, idx_trunc], in_dims=(0))
     
        J_d_func = vmap(jacrev(decode_trunc, argnums=0), in_dims=(0))
        
        #with torch.no_grad():
        J_e = J_e_func(z)
        J_d = J_d_func(x)
        #print('Current GPU memory allocated before part2: ', torch.cuda.memory_allocated() / 1024 ** 3, 'GB')

#        ##----------further batch
    
#         chunk_size = int(x.shape[0]/10)
        
#         #print(chunk_size)

#         # Create chunks of x and z
#         x_chunks = torch.chunk(x, chunk_size, dim=0)
#         z_chunks = torch.chunk(z, chunk_size, dim=0)
        
#         #print(x_chunks[0].shape)


#         # Compute J_e using batches
#         J_e_func = vmap(lambda x: jacrev(self.encode, argnums=0)(x)[:, idx_trunc], in_dims=(0))

#         J_e_chunks = []
#         for z_chunk in z_chunks:
#             #with torch.no_grad():
#             J_e_chunk = J_e_func(z_chunk)
#             J_e_chunks.append(J_e_chunk)
#         J_e = torch.cat(J_e_chunks, dim=0)


#         # Compute J_d using batches
#         J_d_func = vmap(jacrev(decode_trunc, argnums=0), in_dims=(0))

#         J_d_chunks = []
#         for x_chunk in x_chunks:
#             #with torch.no_grad():
#             J_d_chunk = J_d_func(x_chunk)
#             J_d_chunks.append(J_d_chunk)
#         J_d = torch.cat(J_d_chunks, dim=0)
        
       
               
        J_ed = J_d @ J_e
        
        
#         print(J_ed.shape)
#         print(J_ed.diagonal(dim1=-2, dim2=-1).sub_(1).shape)
        
        #print(J_ed.diagonal(dim1=-2, dim2=-1).shape)# = J_ed.diagonal(dim1=-2, dim2=-1).sub_(1)
        
        
        J_ed.diagonal(dim1=-2, dim2=-1).sub_(1)
               

        loss_jacobian = torch.mean(torch.pow(J_ed, 2))
        
        return loss_jacobian, J_e, J_d, idx_trunc
        

    def jacobian_norm_trunc_wo_jac_loss(self, z, x,trunc_period):

        dim_z = z.shape[1]

        idx_trunc = range(0, dim_z - 1, trunc_period)  # 3 for VC, 10 for BG

        def decode_trunc(xx):
            idx = 0
            for layer in self.fc_decoder:
                xx = layer(xx)
                xx = self.activation_function(xx, self.activation_vec[idx])
                idx += 1
            # print(xx.shape)
            return xx[idx_trunc]
        
        #           #--------no further batch
        J_e_func = vmap(lambda x: jacrev(self.encode, argnums=0)(x)[:, idx_trunc], in_dims=(0))
     
        J_d_func = vmap(jacrev(decode_trunc, argnums=0), in_dims=(0))
        
        #with torch.no_grad():
        J_e = J_e_func(z)
        J_d = J_d_func(x)

        
        ## ----------- further batch
#         chunk_size = int(x.shape[0]/10)
        
#         #print(chunk_size)

#         # Create chunks of x and z
#         x_chunks = torch.chunk(x, chunk_size, dim=0)
#         z_chunks = torch.chunk(z, chunk_size, dim=0)
        
#         #print(x_chunks[0].shape)


#         # Compute J_e using batches
#         J_e_func = vmap(lambda x: jacrev(self.encode, argnums=0)(x)[:, idx_trunc], in_dims=(0))

#         J_e_chunks = []
#         for z_chunk in z_chunks:
#             #with torch.no_grad():
#             J_e_chunk = J_e_func(z_chunk)
#             J_e_chunks.append(J_e_chunk)
#         J_e = torch.cat(J_e_chunks, dim=0)


#         # Compute J_d using batches
#         J_d_func = vmap(jacrev(decode_trunc, argnums=0), in_dims=(0))

#         J_d_chunks = []
#         for x_chunk in x_chunks:
#             #with torch.no_grad():
#             J_d_chunk = J_d_func(x_chunk)
#             J_d_chunks.append(J_d_chunk)
#         J_d = torch.cat(J_d_chunks, dim=0)

               
        J_ed = J_d @ J_e
        

        return J_ed, J_e, J_d, idx_trunc

#     def jacobian_norm_wo_jac_loss(self, z, x):

#         J_e_func = vmap(jacrev(self.encode, argnums=0), in_dims=(0))
#         J_e = J_e_func(z)
#         J_d_func = vmap(jacrev(self.encode, argnums=0), in_dims=(0))
#         J_d = J_d_func(x)
        
#         J_ed = J_d @ J_e

#         return J_ed, J_e, J_d

#     def jacobian_norm_trunc_v2(self, z, x,trunc_period):

#         dim_z = z.shape[1]

#         # idx_1 = range(0,dim_z-1,20)
#         # idx_2 = range(0,dim_z-1,20)
#         idx_trunc = range(0, dim_z - 1, trunc_period)

#         z_decode = self.decode(x)
#         z_trunc = z[:, idx_trunc]
#         z_trunc = z_trunc.requires_grad_(True)

#         J_e = grad(self.encode(z), z_trunc)
#         J_d = grad(z_decode[:, idx_trunc], x)

#         J_ed = J_d @ J_e

#         # print(J_e.shape)

#         eye_cat = torch.eye(z.shape[1]).unsqueeze(0).expand(z.shape[0], z.shape[1], z.shape[1])

#         loss_jacobian = torch.mean(torch.pow(J_ed - eye_cat[:, idx_trunc, :][:, :, idx_trunc], 2))

#         return loss_jacobian

    # Forward pass
    def forward(self, z):
        x = self.encode(z)
        z_reconst = self.decode(x)
        return z_reconst, x

    # Normalization
    def normalize(self, z):
        return z

    def denormalize(self, z_norm):
        return z_norm


class StackedSparseAutoEncoder(nn.Module):
    """Sparse Autoencoder"""

    def __init__(self, layer_vec_q, layer_vec_v, layer_vec_sigma, activation, dtype):
        super(StackedSparseAutoEncoder, self).__init__()
        self.layer_vec_q = layer_vec_q
        self.layer_vec_v = layer_vec_v
        self.layer_vec_sigma = layer_vec_sigma
        self.dim_latent_q = layer_vec_q[-1]
        self.dim_latent_v = layer_vec_v[-1]
        self.dim_latent_sigma = layer_vec_sigma[-1]
        self.dim_latent = self.dim_latent_q + self.dim_latent_v + self.dim_latent_sigma

        if dtype == 'double':
            self.SAE_q = SparseAutoEncoder(layer_vec_q, activation).double()
            self.SAE_v = SparseAutoEncoder(layer_vec_v, activation).double()
            self.SAE_sigma = SparseAutoEncoder(layer_vec_sigma, activation).double()
        elif dtype == 'float':
            self.SAE_q = SparseAutoEncoder(layer_vec_q, activation).float()
            self.SAE_v = SparseAutoEncoder(layer_vec_v, activation).float()
            self.SAE_sigma = SparseAutoEncoder(layer_vec_sigma, activation).float()
            
    # Stacked Encoder
    def encode(self, z):
        #print(z.shape) #159 49680 RT
        q, v, sigma = self.split_state(z)
        x_q = self.SAE_q.encode(q)
        x_v = self.SAE_v.encode(v)
        x_sigma = self.SAE_sigma.encode(sigma)
        if z.dim() == 2:
            x = torch.cat((x_q, x_v, x_sigma), 1)
        elif z.dim() == 1:
            x = torch.cat((x_q, x_v, x_sigma), 0)
        return x

    # Stacked Decoder
    def decode(self, x):
        x_q, x_v, x_sigma = self.split_latent(x)
        q = self.SAE_q.decode(x_q)
        v = self.SAE_v.decode(x_v)
        sigma = self.SAE_sigma.decode(x_sigma)
        if x.dim() == 2:
            z = torch.cat((q, v, sigma), 1)
        elif x.dim() == 1:
            z = torch.cat((q, v, sigma), 0)
        return z




    def jacobian_norm_trunc(self, z, x, trunc_period):

        # print(z.shape)#[119, 400]

        dim_z = z.shape[1]
        #print(z.shape) #RT 159 49680


        idx_trunc = range(0, dim_z - 1, trunc_period)  # 3 for VC, 10 for BG

        def decode_trunc(xx):
            idx = 0
            # for layer in self.fc_decoder:
            #     xx = layer(xx)
            #     xx = self.activation_function(xx, self.activation_vec[idx])
            #     idx += 1

            return self.decode(xx)[idx_trunc]

        #print(z.shape)

        J_e_func = vmap(lambda x: jacrev(self.encode, argnums=0)(x)[:, idx_trunc], in_dims=(0))
        # J_e_func = vmap(jacrev(self.encode, argnums=0), in_dims=(0))


        J_d_func = vmap(jacrev(decode_trunc, argnums=0), in_dims=(0))
        
        #with torch.no_grad():
        J_e = J_e_func(z)
        J_d = J_d_func(x)
        # print(J_d.shape)

        
        #        ##----------further batch
    
#         chunk_size = int(x.shape[0]/10)
        
#         #print(chunk_size)

#         # Create chunks of x and z
#         x_chunks = torch.chunk(x, chunk_size, dim=0)
#         z_chunks = torch.chunk(z, chunk_size, dim=0)
        
#         #print(x_chunks[0].shape)


#         # Compute J_e using batches
#         J_e_func = vmap(lambda x: jacrev(self.encode, argnums=0)(x)[:, idx_trunc], in_dims=(0))

#         J_e_chunks = []
#         for z_chunk in z_chunks:
#             #with torch.no_grad():
#             J_e_chunk = J_e_func(z_chunk)
#             J_e_chunks.append(J_e_chunk)
#         J_e = torch.cat(J_e_chunks, dim=0)


#         # Compute J_d using batches
#         J_d_func = vmap(jacrev(decode_trunc, argnums=0), in_dims=(0))

#         J_d_chunks = []
#         for x_chunk in x_chunks:
#             #with torch.no_grad():
#             J_d_chunk = J_d_func(x_chunk)
#             J_d_chunks.append(J_d_chunk)
#         J_d = torch.cat(J_d_chunks, dim=0)

        
        J_ed = J_d @ J_e


        J_ed.diagonal(dim1=-2, dim2=-1).sub_(1)
               

        loss_jacobian = torch.mean(torch.pow(J_ed, 2))

        return loss_jacobian, J_e, J_d, idx_trunc

    def jacobian_norm_trunc_gpu(self, z, x, trunc_period):

        dim_z = z.shape[1]

        idx_trunc = range(0, dim_z - 1, trunc_period)  # 3 for VC, 10 for BG
        #print('Current GPU memory allocated part1: ', torch.cuda.memory_allocated() / 1024 ** 3, 'GB')

        def decode_trunc(xx):
            # idx = 0
            # for layer in self.fc_decoder:
            #     xx = layer(xx)
            #     xx = self.activation_function(xx, self.activation_vec[idx])
            #     idx += 1

            return self.decode(xx)[idx_trunc]

        J_e_func = vmap(jacrev(self.encode, argnums=0), in_dims=(0))
        J_e_func = vmap(lambda x: jacrev(self.encode, argnums=0)(x)[:, idx_trunc], in_dims=(0))

        
        J_d_func = vmap(jacrev(decode_trunc, argnums=0), in_dims=(0))
        #J_d_func = vmap(lambda x: jacrev(self.decode, argnums=0)(x)[idx_trunc, :], in_dims=(0))

#         print(x.detach().requires_grad)
#         print(z.requires_grad)
        
        #with torch.no_grad():
        J_e = J_e_func(z)
        #print('Current GPU memory allocated part1-1: ', torch.cuda.memory_allocated() / 1024 ** 3, 'GB')
        J_d = J_d_func(x)

    
#         chunk_size = int(x.shape[0]/20)
        
#         #print(chunk_size)

#         # Create chunks of x and z
#         x_chunks = torch.chunk(x, chunk_size, dim=0)
#         z_chunks = torch.chunk(z, chunk_size, dim=0)
        
#         #print(x_chunks[0].shape)


#         # Compute J_e using batches
#         J_e_func = vmap(lambda x: jacrev(self.encode, argnums=0)(x)[:, idx_trunc], in_dims=(0))

#         J_e_chunks = []
#         for z_chunk in z_chunks:
#             with torch.no_grad():
#                 J_e_chunk = J_e_func(z_chunk)
#             J_e_chunks.append(J_e_chunk)
#         J_e = torch.cat(J_e_chunks, dim=0)


#         # Compute J_d using batches
#         J_d_func = vmap(jacrev(decode_trunc, argnums=0), in_dims=(0))

#         J_d_chunks = []
#         for x_chunk in x_chunks:
#             with torch.no_grad():
#                 J_d_chunk = J_d_func(x_chunk)
#             J_d_chunks.append(J_d_chunk)
#         J_d = torch.cat(J_d_chunks, dim=0)
            
               
        
        #print('Current GPU memory allocated part2: ', torch.cuda.memory_allocated() / 1024 ** 3, 'GB')
        
        J_ed = J_d @ J_e
        

        J_ed.diagonal(dim1=-2, dim2=-1).sub_(1)
               

        loss_jacobian = torch.mean(torch.pow(J_ed, 2))

#         eye_cat = None
        
        #print('Current GPU memory allocated part4: ', torch.cuda.memory_allocated() / 1024 ** 3, 'GB')

        return loss_jacobian, J_e, J_d, idx_trunc

    def jacobian_norm_trunc_wo_jac_loss(self, z, x, trunc_period):

        dim_z = z.shape[1]

        idx_trunc = range(0, dim_z - 1, trunc_period)  # 3 for VC, 10 for BG

        def decode_trunc(xx):
            # idx = 0
            # for layer in self.fc_decoder:
            #     xx = layer(xx)
            #     xx = self.activation_function(xx, self.activation_vec[idx])
            #     idx += 1
            # # print(xx.shape)
            return self.decode(xx)[idx_trunc]

        J_e_func = vmap(lambda x: jacrev(self.encode, argnums=0)(x)[:, idx_trunc], in_dims=(0))

        J_e = J_e_func(z)
        J_d_func = vmap(jacrev(decode_trunc, argnums=0), in_dims=(0))
        J_d = J_d_func(x)
        
  #       #---------further batch     
            
#         chunk_size = int(x.shape[0]/20)
        
#         #print(chunk_size)

#         # Create chunks of x and z
#         x_chunks = torch.chunk(x, chunk_size, dim=0)
#         z_chunks = torch.chunk(z, chunk_size, dim=0)
        
#         #print(x_chunks[0].shape)


#         # Compute J_e using batches
#         J_e_func = vmap(lambda x: jacrev(self.encode, argnums=0)(x)[:, idx_trunc], in_dims=(0))

#         J_e_chunks = []
#         for z_chunk in z_chunks:
#             with torch.no_grad():
#                 J_e_chunk = J_e_func(z_chunk)
#             J_e_chunks.append(J_e_chunk)
#         J_e = torch.cat(J_e_chunks, dim=0)


#         # Compute J_d using batches
#         J_d_func = vmap(jacrev(decode_trunc, argnums=0), in_dims=(0))

#         J_d_chunks = []
#         for x_chunk in x_chunks:
#             with torch.no_grad():
#                 J_d_chunk = J_d_func(x_chunk)
#             J_d_chunks.append(J_d_chunk)
#         J_d = torch.cat(J_d_chunks, dim=0)
        
        
        
        J_ed = J_d @ J_e

        return J_ed, J_e, J_d, idx_trunc


    def forward(self, z):
        x = self.encode(z)
        z_reconst = self.decode(x)
        return z_reconst, x

    # Database processing functions
    def split_state(self, z):
        if z.dim() >1:
            start, end = 0, self.layer_vec_q[0]
     #       print(z.shape)
            q = z[:, start:end]
            start, end = end, end + self.layer_vec_v[0]
            v = z[:, start:end]
            start, end = end, end + self.layer_vec_sigma[0]
            sigma = z[:, start:end]
        else:
            start, end = 0, self.layer_vec_q[0]
            q = z[start:end]
            start, end = end, end + self.layer_vec_v[0]
            v = z[start:end]
            start, end = end, end + self.layer_vec_sigma[0]
            sigma = z[start:end]

        return q, v, sigma

    def split_latent(self, x):
        if x.dim() > 1:
            start, end = 0, self.layer_vec_q[-1]
            x_q = x[:, start:end]
            start, end = end, end + self.layer_vec_v[-1]
            x_v = x[:, start:end]
            start, end = end, end + self.layer_vec_sigma[-1]
            x_sigma = x[:, start:end]
        else:
            start, end = 0, self.layer_vec_q[-1]
            x_q = x[start:end]
            start, end = end, end + self.layer_vec_v[-1]
            x_v = x[start:end]
            start, end = end, end + self.layer_vec_sigma[-1]
            x_sigma = x[start:end]
        return x_q, x_v, x_sigma

    # Normalization
    def normalize(self, z):
        #print(z.shape[1])
        if z.shape[1] == 12420:
            n_nodes = 1035
        elif z.shape[1] == 24840:
            n_nodes = 2070
        elif z.shape[1] == 49680:
            n_nodes = 4140
        # Position
        q1_norm = z[:, n_nodes * 0:n_nodes * 1] / 1.5
        q2_norm = z[:, n_nodes * 1:n_nodes * 2] / 0.1
        q3_norm = z[:, n_nodes * 2:n_nodes * 3] / 0.3
        q_norm = torch.cat((q1_norm, q2_norm, q3_norm), 1)
        # Velocity
        v1_norm = z[:, n_nodes * 3:n_nodes * 4] / 5
        v2_norm = z[:, n_nodes * 4:n_nodes * 5] / 1
        v3_norm = z[:, n_nodes * 5:n_nodes * 6] / 3
        v_norm = torch.cat((v1_norm, v2_norm, v3_norm), 1)
        # Stress
        s11_norm = z[:, n_nodes * 6:n_nodes * 7] / 0.5
        s22_norm = z[:, n_nodes * 7:n_nodes * 8] / 0.5
        s33_norm = z[:, n_nodes * 8:n_nodes * 9] / 0.5
        s12_norm = z[:, n_nodes * 9:n_nodes * 10] / 0.5
        s13_norm = z[:, n_nodes * 10:n_nodes * 11] / 0.5
        s23_norm = z[:, n_nodes * 11:n_nodes * 12] / 0.5
        sigma_norm = torch.cat((s11_norm, s22_norm, s33_norm, s12_norm, s13_norm, s23_norm), 1)

        z_norm = torch.cat((q_norm, v_norm, sigma_norm), 1)
        #z_norm = z

        return z_norm

    def denormalize(self, z_norm):
        
        if z_norm.shape[1] == 12420:
            n_nodes = 1035
        elif z_norm.shape[1] == 24840:
            n_nodes = 2070
        elif z_norm.shape[1] == 49680:
            n_nodes = 4140
        # Position
        q1 = z_norm[:, n_nodes * 0:n_nodes * 1] * 1.5
        q2 = z_norm[:, n_nodes * 1:n_nodes * 2] * 0.1
        q3 = z_norm[:, n_nodes * 2:n_nodes * 3] * 0.3
        q = torch.cat((q1, q2, q3), 1)
        # Velocity
        v1 = z_norm[:, n_nodes * 3:n_nodes * 4] * 5
        v2 = z_norm[:, n_nodes * 4:n_nodes * 5] * 1
        v3 = z_norm[:, n_nodes * 5:n_nodes * 6] * 3
        v = torch.cat((v1, v2, v3), 1)
        # Stress
        s11 = z_norm[:, n_nodes * 6:n_nodes * 7] * 0.5
        s22 = z_norm[:, n_nodes * 7:n_nodes * 8] * 0.5
        s33 = z_norm[:, n_nodes * 8:n_nodes * 9] * 0.5
        s12 = z_norm[:, n_nodes * 9:n_nodes * 10] * 0.5
        s13 = z_norm[:, n_nodes * 10:n_nodes * 11] * 0.5
        s23 = z_norm[:, n_nodes * 11:n_nodes * 12] * 0.5
        sigma = torch.cat((s11, s22, s33, s12, s13, s23), 1)

        z = torch.cat((q, v, sigma), 1)
        #z = z_norm
    
        return z


class StructurePreservingNN(nn.Module):
    """Structure Preserving Neural Network"""

    def __init__(self, dim_in, dim_out, hidden_vec, activation):
        super(StructurePreservingNN, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.hidden_vec = hidden_vec
        self.activation = activation

        # Net Hidden Layers
        self.layer_vec = [self.dim_in] + self.hidden_vec + [self.dim_out]
        self.activation_vec = (len(self.layer_vec) - 2) * [self.activation] + ['linear']
        # self.activation_vec = (len(self.layer_vec)-2)*[self.activation] + ['tanh']

        # Net Output GENERIC matrices (L and M)
        self.diag = torch.eye(self.dim_in, self.dim_in)
        self.diag = self.diag.reshape((-1, self.dim_in, self.dim_in))

        # Linear layers append from the layer vector
        self.fc_hidden_layers = nn.ModuleList()
        for k in range(len(self.layer_vec) - 1):
            self.fc_hidden_layers.append(nn.Linear(self.layer_vec[k], self.layer_vec[k + 1]))

    def activation_function(self, x, activation):
        if activation == 'linear':
            x = x
        elif activation == 'sigmoid':
            x = torch.sigmoid(x)
        elif activation == 'relu':
            x = F.relu(x)
        elif activation == 'rrelu':
            x = F.rrelu(x)
        elif activation == 'tanh':
            x = torch.tanh(x)
        elif activation == 'sin':
            x = torch.sin(x)
        elif activation == 'elu':
            x = F.elu(x)
        else:
            raise NotImplementedError
        return x

    def SPNN(self, x):
        x = x.view(-1, self.dim_in)
        idx = 0
        # Apply activation function on each layer
        for layer in self.fc_hidden_layers:
            x = layer(x)
            x = self.activation_function(x, self.activation_vec[idx])
            idx += 1

        # Split output in GENERIC matrices
        start, end = 0, self.dim_in
        dEdz_out = x[:, start:end].unsqueeze(2)
        start, end = end, end + self.dim_in
        dSdz_out = x[:, start:end].unsqueeze(2)
        start, end = end, end + int(
            self.dim_in * (self.dim_in + 1) / 2) - self.dim_in  # Lower triangular elements (No diagonal)
        L_out_vec = x[:, start:end]
        start, end = end, end + int(self.dim_in * (self.dim_in + 1) / 2)  # Diagonal + lower triangular elements
        M_out_vec = x[:, start:end]

        # Rearrange L and M matrices
        L_out = torch.zeros(x.size(0), self.dim_in, self.dim_in)
        M_out = torch.zeros(x.size(0), self.dim_in, self.dim_in)
        L_out[:, torch.tril(torch.ones(self.dim_in, self.dim_in), -1) == 1] = L_out_vec
        M_out[:, torch.tril(torch.ones(self.dim_in, self.dim_in)) == 1] = M_out_vec

        # L symmetric
        L_out = (L_out - torch.transpose(L_out, 1, 2))

        # M skew-symmetric and positive semi-definite
        M_out = M_out - M_out * self.diag + abs(M_out) * self.diag  # Lower triangular + Positive diagonal
        M_out = torch.bmm(M_out, torch.transpose(M_out, 1, 2))  # Cholesky factorization

        return L_out, M_out, dEdz_out, dSdz_out

    def forward(self, x, dt):

        L, M, dEdz, dSdz = self.SPNN(x)
        dzdt, deg_E, deg_S = self.integrator(L, M, dEdz, dSdz)
        x1 = x + dt * dzdt

        return x1, deg_E, deg_S

    def integrator(self, L, M, dEdz, dSdz):
        # GENERIC time integration and degeneration
        dzdt = torch.bmm(L, dEdz) + torch.bmm(M, dSdz)

        deg_E = torch.bmm(M, dEdz)
        deg_S = torch.bmm(L, dSdz)

        return dzdt.view(-1, L.size(1)), deg_E.view(-1, L.size(1)), deg_S.view(-1, L.size(1))

    def get_thermodynamics(self, x):

        L, M, dEdz, dSdz = self.SPNN(x)

        # Energy and Entropy time derivatives
        LdEdz = torch.bmm(L, dEdz)
        MdSdz = torch.bmm(M, dSdz)

        dEdt = torch.bmm(torch.transpose(dEdz, 1, 2), LdEdz).squeeze(2) + torch.bmm(torch.transpose(dEdz, 1, 2),
                                                                                    MdSdz).squeeze(2)
        dSdt = torch.bmm(torch.transpose(dSdz, 1, 2), LdEdz).squeeze(2) + torch.bmm(torch.transpose(dSdz, 1, 2),
                                                                                    MdSdz).squeeze(2)

        return dEdt, dSdt

    def weight_init(self, net_initialization):
        for layer in self.fc_hidden_layers:
            if net_initialization == 'zeros':
                init.constant_(layer.bias, 0)
                init.constant_(layer.weight, 0)
            elif net_initialization == 'xavier_normal':
                init.constant_(layer.bias, 0)
                init.xavier_normal_(layer.weight)
            elif net_initialization == 'xavier_uniform':
                init.constant_(layer.bias, 0)
                init.xavier_uniform_(layer.weight)
            elif net_initialization == 'kaiming_uniform':
                init.constant_(layer.bias, 0)
                init.kaiming_uniform_(layer.weight)
            elif net_initialization == 'sparse':
                init.constant_(layer.bias, 0)
                init.sparse_(layer.weight, sparsity=0.5)
            else:
                raise NotImplementedError


if __name__ == '__main__':
    pass
