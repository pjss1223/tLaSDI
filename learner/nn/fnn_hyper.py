"""
@author: jpzxshi
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from .module_hyper import StructureNN_hyper


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

        # print(w.shape)
        # print(b.shape)
        # print(x.shape)

        w = w.reshape(B, self.weight_size[0], self.weight_size[1])
        b = b.reshape(B, self.bias_size)
        # print(w.shape)#192000, 10, 10
        # print(b.shape)#192000, 10
        # print(x.shape)#192000

        return torch.einsum('ab,acb->ac', x, w) + b

    def get_weight_size(self):
        return self.weight_size

    def get_bias_size(self):
        return self.bias_size

    def get_param_size(self):
        return torch.prod(self.weight_size) + self.bias_size



class FNN_hyper(StructureNN_hyper):
    '''Fully connected neural networks.
    '''

    def __init__(self, num_sensor,ind, outd, depth_trunk=2, width_trunk=50, depth_hyper=2, width_hyper=50, act_trunk='relu', act_hyper='relu', initializer='default', softmax=False):
        # def __init__(self, ind, outd, layers=2, width=50, activation='relu', initializer='kaiming_uniform', softmax=False):

        super(FNN_hyper, self).__init__()
        self.ind = ind #input trunk
        self.outd = outd #output trunk
        self.depth_trunk = depth_trunk
        self.width_trunk = width_trunk
        self.activation_trunk = act_trunk

        self.depth_hyper = depth_hyper
        self.width_hyper = width_hyper
        self.activation_hyper = act_hyper
        self.initializer = initializer
        self.softmax = softmax

        # self.modus = self.__init_modules()
        # self.__initialize()
        self.activation_vec_trunk = (self.depth_trunk) * [self.activation_trunk] + ['linear']
        # self.activation_vec = (self.layers - 2) * [self.activation] + ['tanh']

        #trunk
        self.trunk_list = []
        self.param_sizes=[]
        self.trunk_list.append(FC_layer(ind,width_trunk))
        self.param_sizes.append(FC_layer(ind,width_trunk).get_param_size())
        for i in range(depth_trunk-1):
            self.trunk_list.append(FC_layer(width_trunk, width_trunk))
            self.param_sizes.append(FC_layer(width_trunk, width_trunk).get_param_size())
        # self.trunk_list.append(FC_layer(width_trunk,num_basis))
        # self.param_sizes.append(FC_layer(width_trunk,num_basis).get_param_size())
        self.trunk_list.append(FC_layer(width_trunk,outd))
        self.param_sizes.append(FC_layer(width_trunk,outd).get_param_size())
        self.param_size=int(sum(self.param_sizes))

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
        self.hyper_list = []
        self.hyper_list.append(nn.Linear(num_sensor,width_hyper))
        self.hyper_list.append(self.activation_hyper)
        for i in range(depth_hyper-1):
            self.hyper_list.append(nn.Linear(width_hyper, width_hyper))
            self.hyper_list.append(self.activation_hyper)
        self.hyper_list.append(nn.Linear(width_hyper,self.param_size))
        self.hyper_list = nn.Sequential(*self.hyper_list)


    def activation_function(self, x, activation):
        if activation == 'linear': x = x
        elif activation == 'sigmoid': x = torch.sigmoid(x)
        elif activation == 'relu': x = F.relu(x)
        elif activation == 'rrelu': x = F.rrelu(x)
        elif activation == 'tanh': x = torch.tanh(x)
        elif activation == 'sin': x = torch.sin(x)
        elif activation == 'elu': x = F.elu(x)
        elif activation == 'gelu':x = F.gelu(x)
        elif activation == 'selu':x = F.selu(x)
        else:
            raise NotImplementedError
        return x

    # def forward(self, x):
    #     idx = 0
    #     for i in range(1, self.depth_trunk):
    #         LinM = self.modus['LinM{}'.format(i)]
    #         x = self.activation_function(LinM(x), self.activation_vec[idx])
    #         # print(self.activation_vec[idx])
    #         idx += 1
    #         # x = self.act(LinM(x))
    #     x = self.modus['LinMout'](x)
    #     if self.softmax:
    #         x = nn.functional.softmax(x, dim=-1)
    #     return x

    def forward(self, data_grid, data_sensor):
        cut = 0
        idx = 0
        weight = self.get_param(data_sensor)
        #print(data_grid.shape)#192000 wrong
        for i in range(self.depth_trunk):
            data_grid = self.trunk_list[i](data_grid, weight[..., cut:cut + self.param_sizes[i]])
            data_grid = self.activation_function(data_grid, self.activation_vec_trunk[idx])
            cut += self.param_sizes[i]
            idx += 1

        output = self.trunk_list[self.depth_trunk](data_grid,
                                                       weight[..., cut:cut + self.param_sizes[self.depth_trunk]])
        return output

    def get_param(self, data):
        return self.hyper_list(data)

    # def __init_modules(self):
    #     modules = nn.ModuleDict()
    #     if self.layers > 1:
    #         modules['LinM1'] = nn.Linear(self.ind, self.width)
    #         for i in range(2, self.layers):
    #             modules['LinM{}'.format(i)] = nn.Linear(self.width, self.width)
    #         modules['LinMout'] = nn.Linear(self.width, self.outd)
    #     else:
    #         modules['LinMout'] = nn.Linear(self.ind, self.outd)
    #
    #     return modules
    #
    # def __initialize(self):
    #     for i in range(1, self.layers):
    #         self.weight_init_(self.modus['LinM{}'.format(i)].weight)
    #         nn.init.constant_(self.modus['LinM{}'.format(i)].bias, 0)
    #     self.weight_init_(self.modus['LinMout'].weight)
    #     nn.init.constant_(self.modus['LinMout'].bias, 0)