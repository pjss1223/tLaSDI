"""model.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import functorch
from functorch import vmap, jacrev, jacfwd
from learner.utils import mse, wasserstein, div, grad
import numpy as np
from torch.utils import bottleneck
import time


class AutoEncoder(nn.Module):
    """Autoencoder"""

    def __init__(self, layer_vec, activation):
        super(AutoEncoder, self).__init__()
        self.layer_vec = layer_vec
        self.dim_latent = layer_vec[-1]
        self.activation = activation

        self.activation_vec = (len(self.layer_vec) - 2) * [self.activation] + ['linear']


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
        elif activation == 'sin':
            x = torch.sin(x)
        elif activation == 'elu':
            x = F.elu(x)
        elif activation == 'gelu':
            x = F.gelu(x)
        elif activation == 'silu':
            x = F.silu(x)
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


    def JVP(self, z, x, dz, dx, trunc_period):

        dim_z = z.shape[1]

        idx_trunc = range(0, dim_z, trunc_period)

        def decode_trunc(xx):

            xx = self.decode(xx)
            return xx[:,idx_trunc]

        
        def jvp_de(xa, dxa):

            J_f_x = torch.autograd.functional.jvp(decode_trunc, xa, dxa,create_graph=True)
            J_f_x_v = J_f_x[1]
            J_f_x = None
            return J_f_x_v
        
        def jvp_en(za, dza):

            J_f_x = torch.autograd.functional.jvp(self.encode, za, dza,create_graph=True)
            J_f_x_v = J_f_x[1]
            J_f_x = None
            return J_f_x_v
 

        J_dV = jvp_de(x,  dx)
        J_eV = jvp_en(z,  dz)
        J_edV = jvp_de(x, J_eV)


        return J_edV, J_eV, J_dV, idx_trunc
    
    def Jac(self, z, x, trunc_period):

        dim_z = z.shape[1]

        idx_trunc = range(0, dim_z - 1, trunc_period)  # 
        
        def decode_trunc(xx):
            idx = 0
            for layer in self.fc_decoder:
                xx = layer(xx)
                xx = self.activation_function(xx, self.activation_vec[idx])
                idx += 1

            return xx[idx_trunc]

        J_e_func = vmap(lambda x: jacrev(self.encode, argnums=0)(x)[:, idx_trunc], in_dims=(0))
     
        J_d_func = vmap(jacfwd(decode_trunc, argnums=0), in_dims=(0))
        
        J_e = J_e_func(z)
        J_d = J_d_func(x)
               
        J_ed = J_d @ J_e
        
        
#         J_ed.diagonal(dim1=-2, dim2=-1).sub_(1)
               

#         loss_jacobian = torch.mean(torch.pow(J_ed, 2))
        
        eye_cat = torch.eye(z.shape[1]).unsqueeze(0).expand(z.shape[0], z.shape[1], z.shape[1]).to(torch.device('cuda'))
        

        loss_jacobian = torch.mean(torch.pow(J_ed - eye_cat[:, idx_trunc, :][:, :, idx_trunc], 2))
        
        return loss_jacobian, idx_trunc
        

    
    

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
    
    



if __name__ == '__main__':
    pass
