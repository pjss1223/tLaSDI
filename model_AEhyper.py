"""model_AEhyper.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import functorch
from functorch import vmap, jacrev
from learner.utils import mse, wasserstein, div, grad
import numpy as np

class FC_layer(nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__()

        self.num_in = num_in
        self.num_out = num_out
        self.weight_size = torch.tensor([num_out, num_in])
        self.bias_size = torch.tensor([num_out])



    def forward(self, x, param):
        #print(x.shape)
        B = x.shape[0]
        w = param[:, :torch.prod(self.weight_size)]
        b = param[:, torch.prod(self.weight_size):]

        # print(x.shape)
        # print(w.shape)
        # print(b.shape)
        w = w.reshape(B, self.weight_size[0], self.weight_size[1])
        b = b.reshape(B, self.bias_size)

        # print(x.shape)
        # print(w.shape)
        # print(b.shape)
        return torch.einsum('ab,acb->ac', x, w) + b

    def get_weight_size(self):
        return self.weight_size

    def get_bias_size(self):
        return self.bias_size

    def get_param_size(self):
        return torch.prod(self.weight_size) + self.bias_size



class SparseAutoEncoder(nn.Module):
    """Sparse Autoencoder"""

    def __init__(self, layer_vec, activation, depth_hyper, width_hyper, act_hyper, num_sensor):
        super(SparseAutoEncoder, self).__init__()
        self.layer_vec = layer_vec
        self.dim_latent = layer_vec[-1]
        self.activation = activation
        self.activation_vec = ['linear'] + (len(self.layer_vec) - 3) * [self.activation] + ['linear']


        #encoder
        self.steps = len(self.layer_vec) - 1
        self.fc_encoder = []
        self.en_param_sizes=[]
        # print(self.layer_vec)
        # print(self.layer_vec[2]) #10
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

        # Encode
        # self.steps = len(self.layer_vec) - 1
        # self.fc_encoder = nn.ModuleList()
        # for k in range(self.steps):
        #     self.fc_encoder.append(nn.Linear(self.layer_vec[k], self.layer_vec[k + 1]))
        #
        # # Decode
        # self.fc_decoder = nn.ModuleList()
        # for k in range(self.steps):
        #     self.fc_decoder.append(nn.Linear(self.layer_vec[self.steps - k], self.layer_vec[self.steps - k - 1]))

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

    def jacobian_norm(self, z, x, mu):

        J_e_func =vmap(jacrev(self.encode, argnums=0), in_dims=(0))
        J_e = J_e_func(z, mu)
        J_d_func = vmap(jacrev(self.decode, argnums=0), in_dims=(0))
        J_d = J_d_func(x, mu)

        eye_cat = torch.eye(z.shape[1]).unsqueeze(0).expand(z.shape[0], z.shape[1], z.shape[1])

        loss_jacobian = torch.mean(torch.pow((J_d @ J_e) - eye_cat, 2))

        return loss_jacobian

    def jacobian_norm_gpu(self, z, x, mu):

        J_e_func = vmap(jacrev(self.encode, argnums=0), in_dims=(0))
        J_e = J_e_func(z, mu)
        J_d_func = vmap(jacrev(self.decode, argnums=0), in_dims=(0))
        J_d = J_d_func(x, mu)

        eye_cat = torch.eye(z.shape[1], device='cuda').unsqueeze(0).expand(z.shape[0], z.shape[1], z.shape[1])

        loss_jacobian = torch.mean(torch.pow((J_d @ J_e) - eye_cat, 2))

        return loss_jacobian

    def jacobian_norm_trunc(self, z, x,mu, trunc_period):


        # print(z.shape)#[119, 400]

        dim_z = z.shape[1]

        idx_trunc = range(0, dim_z - 1, trunc_period)  # 3 for VC, 10 for BG, 1 for para BG

        def decode_trunc(xx, mu):
            # idx = 0
            # for layer in self.fc_decoder:
            #     xx = layer(xx)
            #     xx = self.activation_function(xx, self.activation_vec[idx])
            #     idx += 1
            xx = self.decode(xx, mu)
            return xx[idx_trunc]

        #J_e_func = vmap(lambda x: jacrev(self.encode, argnums=0)(x), in_dims=(0))
        #J_e_func = vmap(lambda x: jacrev(lambda y: self.encode, argnums=0)(x), in_dims=(0))

        # print(z.shape) # 400 101
        # print(mu.shape) # 400 2
        #J_e_func = vmap(lambda x: jacrev(self.encode(x[:,:-2],x[:,-2:]), argnums=0)(x), in_dims=(0))



        #J_e = jacobian(lambda x: self.encode(x[:,:-2],x[:,-2:]), torch.cat((z, mu), dim=1))
        #J_e = vmap(lambda x: jacobian(lambda x: self.encode(x[:, :-2], x[:, -2:]), x[:,:-2]))(torch.cat((z, mu), dim=1))

        #print(torch.cat((z, mu), dim=1).shape)
        #J_e = J_e_func(torch.cat((z, mu), dim=1))
        z = z.requires_grad_(True)
        En = self.encode(z,mu)
        J_e = grad(En, z)

        x = x.requires_grad_(True)
        De = self.decode(x, mu)
        J_d = grad(De, x)


        # # print(J_e.shape)
        #J_d_func = vmap(jacrev(decode_trunc, argnums=0), in_dims=(0))

        #J_d = jacobian(decode_trunc, torch.cat((x, mu), dim=1))


        #J_d = J_d_func(x,mu)
        # print(J_d.shape)


        J_ed = J_d @ J_e


        eye_cat = torch.eye(z.shape[1]).unsqueeze(0).expand(z.shape[0], z.shape[1], z.shape[1])
       # print(J_ed[:, idx_trunc, :][:, :, idx_trunc].shape)

        loss_jacobian = torch.mean(torch.pow(J_ed[:, idx_trunc, :][:, :, idx_trunc] - eye_cat[:, idx_trunc, :][:, :, idx_trunc], 2))

        return loss_jacobian, J_e[:, :, :][:, :, idx_trunc], J_d[:, idx_trunc, :][:, :, :], idx_trunc

    def jacobian_norm_trunc_gpu(self, z, x, mu, trunc_period):

        dim_z = z.shape[1]

        idx_trunc = range(0, dim_z - 1, trunc_period)  # 3 for VC, 10 for BG

        def decode_trunc(xx, mu):
            # idx = 0
            # for layer in self.fc_decoder:
            #     xx = layer(xx)
            #     xx = self.activation_function(xx, self.activation_vec[idx])
            #     idx += 1
            xx = self.decode(xx,mu)
            return xx[idx_trunc]


#         J_e_func = vmap(lambda z, mu: jacrev(self.encode, argnums=0)(z, mu)[:, idx_trunc], in_dims=(0,0), out_dims=0)
#         J_e = J_e_func(z, mu)

#         J_d_func = vmap(lambda x, mu: jacrev(decode_trunc, argnums=0)(x, mu), in_dims=(0,0), out_dims=0)
#         J_d = J_d_func(x, mu)
        
#         J_ed = J_d @ J_e

#         eye_cat = torch.eye(z.shape[1], device='cuda').unsqueeze(0).expand(z.shape[0], z.shape[1], z.shape[1])

#         # loss_jacobian = torch.mean(torch.pow(J_ed[:, :, idx_trunc] - eye_cat[:, idx_trunc, :][:, :, idx_trunc], 2))
#         loss_jacobian = torch.mean(torch.pow(J_ed[:, :, :] - eye_cat[:, idx_trunc, :][:, :, idx_trunc], 2))

        z = z.requires_grad_(True)
        En = self.encode(z,mu)
        J_e = grad(En, z)

        x = x.requires_grad_(True)
        De = self.decode(x, mu)
        J_d = grad(De, x)


        J_ed = J_d @ J_e


        eye_cat = torch.eye(z.shape[1], device='cuda').unsqueeze(0).expand(z.shape[0], z.shape[1], z.shape[1])
       # print(J_ed[:, idx_trunc, :][:, :, idx_trunc].shape)

        loss_jacobian = torch.mean(torch.pow(J_ed[:, idx_trunc, :][:, :, idx_trunc] - eye_cat[:, idx_trunc, :][:, :, idx_trunc], 2))

        return loss_jacobian, J_e[:, :, :][:, :, idx_trunc], J_d[:, idx_trunc, :][:, :, :], idx_trunc

    def jacobian_norm_trunc_wo_jac_loss(self, z, x, mu, trunc_period):

        dim_z = z.shape[1]

        idx_trunc = range(0, dim_z - 1, trunc_period)  # 3 for VC, 10 for BG

        def decode_trunc(xx,mu):
            # idx = 0
            # for layer in self.fc_decoder:
            #     xx = layer(xx)
            #     xx = self.activation_function(xx, self.activation_vec[idx])
            #     idx += 1
            # print(xx.shape)
            xx = self.decode(xx,mu)
            return xx[idx_trunc]


        J_e_func = vmap(lambda x: jacrev(self.encode, argnums=0)(x)[:, idx_trunc], in_dims=(0))

        J_e = J_e_func(z,mu)
        J_d_func = vmap(jacrev(decode_trunc, argnums=0), in_dims=(0))
        J_d = J_d_func(x,mu)

        return J_e, J_d, idx_trunc

    def jacobian_norm_wo_jac_loss(self, z, x, mu):

        J_e_func = vmap(jacrev(self.encode, argnums=0), in_dims=(0))
        J_e = J_e_func(z,mu)
        J_d_func = vmap(jacrev(self.encode, argnums=0), in_dims=(0))
        J_d = J_d_func(x,mu)

        return J_e, J_d



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


class StackedSparseAutoEncoder(nn.Module):
    """Sparse Autoencoder"""

    def __init__(self, layer_vec_q, layer_vec_v, layer_vec_sigma, activation):
        super(StackedSparseAutoEncoder, self).__init__()
        self.layer_vec_q = layer_vec_q
        self.layer_vec_v = layer_vec_v
        self.layer_vec_sigma = layer_vec_sigma
        self.dim_latent_q = layer_vec_q[-1]
        self.dim_latent_v = layer_vec_v[-1]
        self.dim_latent_sigma = layer_vec_sigma[-1]
        self.dim_latent = self.dim_latent_q + self.dim_latent_v + self.dim_latent_sigma


        self.SAE_q = SparseAutoEncoder(layer_vec_q, activation).double()
        self.SAE_v = SparseAutoEncoder(layer_vec_v, activation).double()
        self.SAE_sigma = SparseAutoEncoder(layer_vec_sigma, activation).double()

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

    # def jacobian_norm(self, z, x):
    #     x_q, x_v, x_sigma = self.split_latent(x)
    #     q, v, sigma = self.split_state(z)
    #
    #     J_e_q_func = vmap(jacrev(self.SAE_q.encode, argnums=0), in_dims=(0))
    #     J_e_q = J_e_q_func(q)
    #     J_d_q_func = vmap(jacrev(self.SAE_q.decode, argnums=0), in_dims=(0))
    #     J_d_q = J_d_q_func(x_q)
    #
    #     # print(J_e_q.shape)
    #
    #     J_e_v_func = vmap(jacrev(self.SAE_v.encode, argnums=0), in_dims=(0))
    #     J_e_v = J_e_v_func(v)
    #     J_d_v_func = vmap(jacrev(self.SAE_v.decode, argnums=0), in_dims=(0))
    #     J_d_v = J_d_v_func(x_v)
    #
    #     J_e_sigma_func = vmap(jacrev(self.SAE_sigma.encode, argnums=0), in_dims=(0))
    #     J_e_sigma = J_e_sigma_func(sigma)
    #     J_d_sigma_func = vmap(jacrev(self.SAE_sigma.decode, argnums=0), in_dims=(0))
    #     J_d_sigma = J_d_sigma_func(x_sigma)
    #
    #     eye_cat_q = torch.eye(q.shape[1]).unsqueeze(0).expand(q.shape[0], q.shape[1], q.shape[1])
    #     eye_cat_v = torch.eye(v.shape[1]).unsqueeze(0).expand(v.shape[0], v.shape[1], v.shape[1])
    #     eye_cat_sigma = torch.eye(sigma.shape[1]).unsqueeze(0).expand(sigma.shape[0], sigma.shape[1], sigma.shape[1])
    #     # print((J_d @ J_e).shape)
    #     loss_jacobian_q = torch.mean(torch.pow((J_d_q @ J_e_q) - eye_cat_q, 2))
    #     loss_jacobian_v = torch.mean(torch.pow((J_d_v @ J_e_v) - eye_cat_v, 2))
    #     loss_jacobian_sigma = torch.mean(torch.pow((J_d_sigma @ J_e_sigma) - eye_cat_sigma, 2))
    #     loss_jacobian = loss_jacobian_q + loss_jacobian_v + loss_jacobian_sigma
    #
    #     return loss_jacobian
    #
    #
    #
    # def jacobian_norm_gpu(self, z, x):
    #     x_q, x_v, x_sigma = self.split_latent(x)
    #     q, v, sigma = self.split_state(z)
    #
    #     J_e_q_func = vmap(jacrev(self.SAE_q.encode, argnums=0), in_dims=(0))
    #     J_e_q = J_e_q_func(q)
    #     J_d_q_func = vmap(jacrev(self.SAE_q.decode, argnums=0), in_dims=(0))
    #     J_d_q = J_d_q_func(x_q)
    #
    #     # print(J_e_q.shape)
    #
    #     J_e_v_func = vmap(jacrev(self.SAE_v.encode, argnums=0), in_dims=(0))
    #     J_e_v = J_e_v_func(v)
    #     J_d_v_func = vmap(jacrev(self.SAE_v.decode, argnums=0), in_dims=(0))
    #     J_d_v = J_d_v_func(x_v)
    #
    #     J_e_sigma_func = vmap(jacrev(self.SAE_sigma.encode, argnums=0), in_dims=(0))
    #     J_e_sigma = J_e_sigma_func(sigma)
    #     J_d_sigma_func = vmap(jacrev(self.SAE_sigma.decode, argnums=0), in_dims=(0))
    #     J_d_sigma = J_d_sigma_func(x_sigma)
    #
    #     eye_cat_q = torch.eye(q.shape[1], device='cuda').unsqueeze(0).expand(q.shape[0], q.shape[1], q.shape[1])
    #     eye_cat_v = torch.eye(v.shape[1], device='cuda').unsqueeze(0).expand(v.shape[0], v.shape[1], v.shape[1])
    #     eye_cat_sigma = torch.eye(sigma.shape[1], device='cuda').unsqueeze(0).expand(sigma.shape[0], sigma.shape[1],
    #                                                                                  sigma.shape[1])
    #     # print((J_d @ J_e).shape)
    #     loss_jacobian_q = torch.mean(torch.pow((J_d_q @ J_e_q) - eye_cat_q, 2))
    #     loss_jacobian_v = torch.mean(torch.pow((J_d_v @ J_e_v) - eye_cat_v, 2))
    #     loss_jacobian_sigma = torch.mean(torch.pow((J_d_sigma @ J_e_sigma) - eye_cat_sigma, 2))
    #     loss_jacobian = loss_jacobian_q + loss_jacobian_v + loss_jacobian_sigma
    #
    #     return loss_jacobian
    #
    def jacobian_norm(self, z, x):

        J_e_func = vmap(jacrev(self.encode, argnums=0), in_dims=(0))
        J_e = J_e_func(z)
        J_d_func = vmap(jacrev(self.decode, argnums=0), in_dims=(0))
        J_d = J_d_func(x)

        eye_cat = torch.eye(z.shape[1]).unsqueeze(0).expand(z.shape[0], z.shape[1], z.shape[1])

        loss_jacobian = torch.mean(torch.pow((J_d @ J_e) - eye_cat, 2))

        return loss_jacobian

    def jacobian_norm_gpu(self, z, x):

        J_e_func = vmap(jacrev(self.encode, argnums=0), in_dims=(0))
        J_e = J_e_func(z)
        J_d_func = vmap(jacrev(self.decode, argnums=0), in_dims=(0))
        J_d = J_d_func(x)

        eye_cat = torch.eye(z.shape[1], device='cuda').unsqueeze(0).expand(z.shape[0], z.shape[1], z.shape[1])

        loss_jacobian = torch.mean(torch.pow((J_d @ J_e) - eye_cat, 2))

        return loss_jacobian

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

        #print(idx_trunc)
        J_e = J_e_func(z)
        #print(z.shape)
        # print(J_e.shape)

        J_d_func = vmap(jacrev(decode_trunc, argnums=0), in_dims=(0))
        J_d = J_d_func(x)
        # print(J_d.shape)



        J_ed = J_d @ J_e

        # print(J_e.shape)#[119, 10, 400]
        # print(J_d.shape)#[119, 40, 10]

        # print(J_ed.shape)#[119, 40, 400]

        eye_cat = torch.eye(z.shape[1]).unsqueeze(0).expand(z.shape[0], z.shape[1], z.shape[1])


        loss_jacobian = torch.mean(torch.pow(J_ed[:, :, :] - eye_cat[:, idx_trunc, :][:, :, idx_trunc], 2))

        return loss_jacobian, J_e, J_d, idx_trunc

    def jacobian_norm_trunc_gpu(self, z, x, trunc_period):

        dim_z = z.shape[1]

        idx_trunc = range(0, dim_z - 1, trunc_period)  # 3 for VC, 10 for BG

        def decode_trunc(xx):
            # idx = 0
            # for layer in self.fc_decoder:
            #     xx = layer(xx)
            #     xx = self.activation_function(xx, self.activation_vec[idx])
            #     idx += 1

            return self.decode(xx)[idx_trunc]

        # J_e_func = vmap(jacrev(self.encode, argnums=0), in_dims=(0))
        J_e_func = vmap(lambda x: jacrev(self.encode, argnums=0)(x)[:, idx_trunc], in_dims=(0))

        J_e = J_e_func(z)
        J_d_func = vmap(jacrev(decode_trunc, argnums=0), in_dims=(0))
        J_d = J_d_func(x)

        J_ed = J_d @ J_e

        eye_cat = torch.eye(z.shape[1], device='cuda').unsqueeze(0).expand(z.shape[0], z.shape[1], z.shape[1])

        # loss_jacobian = torch.mean(torch.pow(J_ed[:, :, idx_trunc] - eye_cat[:, idx_trunc, :][:, :, idx_trunc], 2))
        loss_jacobian = torch.mean(torch.pow(J_ed[:, :, :] - eye_cat[:, idx_trunc, :][:, :, idx_trunc], 2))

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

        return J_e, J_d, idx_trunc

    def jacobian_norm_wo_jac_loss(self, z, x):

        J_e_func = vmap(jacrev(self.encode, argnums=0), in_dims=(0))
        J_e = J_e_func(z)
        J_d_func = vmap(jacrev(self.decode, argnums=0), in_dims=(0))
        J_d = J_d_func(x)

        return J_e, J_d
    # Forward pass
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

        return z_norm

    def denormalize(self, z_norm):
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
