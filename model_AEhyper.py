"""model_AEhyper.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from learner.utils import mse, wasserstein, div, grad
import numpy as np
import time

class FC_layer(nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__()

        self.num_in = num_in
        self.num_out = num_out
        self.weight_size = torch.tensor([num_out, num_in])
        self.bias_size = torch.tensor([num_out])



    def forward(self, x, param):
        B = x.shape[0]
        w = param[:, :torch.prod(self.weight_size)]
        b = param[:, torch.prod(self.weight_size):]

        w = w.reshape(B, self.weight_size[0], self.weight_size[1])
        b = b.reshape(B, self.bias_size)

        return torch.einsum('ab,acb->ac', x, w) + b

    def get_weight_size(self):
        return self.weight_size

    def get_bias_size(self):
        return self.bias_size

    def get_param_size(self):
        return torch.prod(self.weight_size) + self.bias_size



class AutoEncoder(nn.Module):
    """Autoencoder"""

    def __init__(self, layer_vec, activation, depth_hyper, width_hyper, act_hyper, num_sensor):
        super(AutoEncoder, self).__init__()
        self.layer_vec = layer_vec
        self.dim_latent = layer_vec[-1]
        self.activation = activation
        self.activation_vec = ['linear'] + (len(self.layer_vec) - 3) * [self.activation] + ['linear']


        #encoder
        self.steps = len(self.layer_vec) - 1
        self.fc_encoder = []
        self.en_param_sizes=[]

        for k in range(self.steps):
            self.fc_encoder.append(FC_layer(self.layer_vec[k],self.layer_vec[k+1]))
            self.en_param_sizes.append(FC_layer(self.layer_vec[k],self.layer_vec[k+1]).get_param_size())
        self.en_param_size=int(sum(self.en_param_sizes))

        # decoder
        # self.steps = len(self.layer_vec) - 1
        self.fc_decoder = []
        self.de_param_sizes = []
        for k in range(self.steps):
            self.fc_decoder.append(FC_layer(self.layer_vec[self.steps - k], self.layer_vec[self.steps - k - 1]))
            self.de_param_sizes.append(FC_layer(self.layer_vec[self.steps - k], self.layer_vec[self.steps - k - 1]).get_param_size())
        self.de_param_size = int(sum(self.de_param_sizes))


        ##hyper net
        self.depth_hyper=depth_hyper
        if act_hyper=='tanh':
            self.activation_hyper=nn.Tanh()
        elif act_hyper=='prelu':
            self.activation_hyper=nn.PReLU()
        elif act_hyper=='relu':
            self.activation_hyper=nn.ReLU()
        else:
            print('error!!')

        self.en_hyper_list = []
        self.en_hyper_list.append(nn.Linear(num_sensor,width_hyper))
        self.en_hyper_list.append(self.activation_hyper)
        for i in range(depth_hyper-1):
            self.en_hyper_list.append(nn.Linear(width_hyper, width_hyper))
            self.en_hyper_list.append(self.activation_hyper)
        self.en_hyper_list.append(nn.Linear(width_hyper,self.en_param_size))
        self.en_hyper_list = nn.Sequential(*self.en_hyper_list)

        self.de_hyper_list = []
        self.de_hyper_list.append(nn.Linear(num_sensor,width_hyper))
        self.de_hyper_list.append(self.activation_hyper)
        for i in range(depth_hyper-1):
            self.de_hyper_list.append(nn.Linear(width_hyper, width_hyper))
            self.de_hyper_list.append(self.activation_hyper)
        self.de_hyper_list.append(nn.Linear(width_hyper,self.de_param_size))
        self.de_hyper_list = nn.Sequential(*self.de_hyper_list)


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
    def encode(self, data_grid, data_sensor):
        #print(x.shape)
        cut = 0
        idx = 0

        if len(data_grid.size()) == 1:
            data_grid = data_grid.unsqueeze(0)
        if len(data_sensor.size()) == 1:
            data_sensor = data_sensor.unsqueeze(0)
        
            
        weight = self.get_param_en(data_sensor)
        #print(weight.shape) # ntrain*dim_t, parasizes (11210)

        for layer in self.fc_encoder:
            #print(self.en_param_sizes[idx])
            data_grid = layer(data_grid, weight[..., cut:cut + self.en_param_sizes[idx]])
            data_grid = self.activation_function(data_grid, self.activation_vec[idx])
            cut += self.en_param_sizes[idx]
            idx += 1
        return data_grid

    # Decoder
    def decode(self, data_grid, data_sensor):
        # print(x.shape)
        # print(data_grid.shape)
        # print(data_sensor.shape)
        cut = 0
        idx = 0
        if len(data_grid.size()) == 1:
            data_grid = data_grid.unsqueeze(0)
        if len(data_sensor.size()) == 1:
            data_sensor = data_sensor.unsqueeze(0)
        
        weight = self.get_param_de(data_sensor)

        for layer in self.fc_decoder:
            # print(x.shape)
            data_grid = layer(data_grid, weight[..., cut:cut + self.de_param_sizes[idx]])
            data_grid = self.activation_function(data_grid, self.activation_vec[idx])
            cut += self.de_param_sizes[idx]
            idx += 1
        return data_grid

    def get_param_en(self, data):
        return self.en_hyper_list(data)

    def get_param_de(self, data):
        return self.de_hyper_list(data)

    
    def JVP(self, z, x, dz, dx, mu, trunc_period):

        dim_z = z.shape[1]

        idx_trunc = range(0, dim_z - 1, trunc_period)  # 3 for VC, 10 for BG

        def decode_trunc(xx,mu):

            xx = self.decode(xx,mu)
            return xx[:,idx_trunc]

        
        def jvp_de(xa, mua, dxa):
            decode_wrapper = lambda xx: decode_trunc(xx, mua)
            J_f_x = torch.autograd.functional.jvp(decode_wrapper, xa, dxa,create_graph=True)
            J_f_x_v = J_f_x[1]
            J_f_x = None
            return J_f_x_v
        
        def jvp_en(za, mua, dza):
            encode_wrapper = lambda zz: self.encode(zz, mua)

            J_f_x = torch.autograd.functional.jvp(encode_wrapper, za, dza,create_graph=True)
            J_f_x_v = J_f_x[1]
            J_f_x = None
            return J_f_x_v
        
        

        J_dV = jvp_de(x, mu, dx)
        J_eV = jvp_en(z, mu, dz)
        J_edV = jvp_de(x, mu, J_eV)

        


        return J_edV, J_eV, J_dV, idx_trunc
    
    
        # Forward pass
    def forward(self, z, mu):

        x = self.encode(z, mu)
        z_reconst = self.decode(x, mu)
        return z_reconst, x

    # Normalization
    def normalize(self, z):
        return z

    def denormalize(self, z_norm):
            return z_norm

    


if __name__ == '__main__':
    pass
