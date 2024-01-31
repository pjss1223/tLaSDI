"""utils.py"""

import torch
import numpy as np

def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_variables(z, sys_name):

    if (sys_name == 'viscoelastic') or (sys_name == 'GC') or (sys_name == 'GC_SVD_concat'):
        n_nodes = 100
        #n_nodes = z.shape[1]/4

        # MSE Error
        q = z[:,n_nodes*0:n_nodes*1]
        v = z[:,n_nodes*1:n_nodes*2]
        e = z[:,n_nodes*2:n_nodes*3]
        tau = z[:,n_nodes*3:n_nodes*4]

        return q, v, e, tau
    
    elif (sys_name == '2DBurgers'):

        n_nodes = int(z.shape[1]/2)

        # MSE Error
        u = z[:,n_nodes*0:n_nodes*1]
        v = z[:,n_nodes*1:n_nodes*2]

        return u,v

    elif (sys_name == '1DBurgers'):


        n_nodes = z.shape[1]
        #print(z.shape)

        # MSE Error
        u = z[:,n_nodes*0:n_nodes*1]

        return u
    
    elif (sys_name == '1DHeat'):


        n_nodes = z.shape[1]

        # MSE Error
        u = z[:,n_nodes*0:n_nodes*1]

        return u
    
    elif (sys_name == 'GC_SVD') or (sys_name == 'VC_SPNN_SVD'):

        n_nodes = z.shape[1]
        #print(z.shape)

        # MSE Error
        u = z[:,n_nodes*0:n_nodes*1]

        return u


    elif (sys_name == 'rolling_tire'):
        #n_nodes = 4140
        #print(z.shape[1])
        n_nodes = int(z.shape[1]/12)
        #print(n_nodes)

        # Initialize vectors
        q = torch.zeros([3, z.size(0), n_nodes])
        v = torch.zeros([3, z.size(0), n_nodes])
        sigma = torch.zeros([6, z.size(0), n_nodes])

        # Position
        q[0] = z[:,n_nodes*0:n_nodes*1]
        q[1] = z[:,n_nodes*1:n_nodes*2]
        q[2] = z[:,n_nodes*2:n_nodes*3]
        # Velocity
        v[0] = z[:,n_nodes*3:n_nodes*4]
        v[1] = z[:,n_nodes*4:n_nodes*5]
        v[2] = z[:,n_nodes*5:n_nodes*6]
        # Stress
        sigma[0] = z[:,n_nodes*6:n_nodes*7]
        sigma[1] = z[:,n_nodes*7:n_nodes*8]
        sigma[2] = z[:,n_nodes*8:n_nodes*9]
        sigma[3] = z[:,n_nodes*9:n_nodes*10]
        sigma[4] = z[:,n_nodes*10:n_nodes*11]
        sigma[5] = z[:,n_nodes*11:n_nodes*12]

        return q, v, sigma


def print_mse(z_net, z_gt, sys_name):

    if (sys_name == 'viscoelastic'):
        # Get variables
        q_net, v_net, e_net, tau_net = get_variables(z_net, sys_name)
        q_gt, v_gt, e_gt, tau_gt = get_variables(z_gt, sys_name)


        # v_mse = torch.mean(torch.mean((v_net - v_gt)**2,0))
        # e_mse = torch.mean(torch.mean((e_net - e_gt)**2,0))
        # tau_mse = torch.mean(torch.mean((tau_net - tau_gt)**2,0))
        
        #1: avg over snapshot errors, 0: avg over trajectory errors
        q_l2 = torch.mean(torch.sqrt(torch.sum((q_gt - q_net) ** 2, 1) / torch.sum(q_gt ** 2, 1)))
        v_l2 = torch.mean(torch.sqrt(torch.sum((v_gt - v_net) ** 2, 1) / torch.sum(v_gt ** 2, 1)))
        e_l2 = torch.mean(torch.sqrt(torch.sum((e_gt - e_net) ** 2, 1) / torch.sum(e_gt ** 2, 1)))
        tau_l2 = torch.mean(torch.sqrt(torch.sum((tau_gt - tau_net) ** 2, 1) / torch.sum(tau_gt ** 2, 1)))
#         print(torch.sqrt(torch.sum((tau_gt - tau_net) ** 2, 0) / torch.sum(tau_gt ** 2, 0)).shape) #100
#         q_mse = torch.mean(torch.mean((q_net - q_gt)**2,0)/torch.mean(q_gt**2,0))
#         v_mse = torch.mean(torch.mean((v_net - v_gt)**2,0)/torch.mean(v_gt**2,0))


        # Print MSE
        print('Position relative l2 error = {:1.2e}'.format(q_l2))
        print('Velocity relative l2 error = {:1.2e}'.format(v_l2))
        print('Energy relative l2 error = {:1.2e}'.format(e_l2))
        print('Conformation Tensor relative l2 error = {:1.2e}'.format(tau_l2))
        
    elif (sys_name == 'GC') or (sys_name == 'GC_SVD_concat'):
        # Get variables
        q_net, p_net, s1_net, s2_net = get_variables(z_net, sys_name)
        q_gt, p_gt, s1_gt, s2_gt = get_variables(z_gt, sys_name)


        # v_mse = torch.mean(torch.mean((v_net - v_gt)**2,0))
        # e_mse = torch.mean(torch.mean((e_net - e_gt)**2,0))
        # tau_mse = torch.mean(torch.mean((tau_net - tau_gt)**2,0))
        
        #1: avg over snapshots changing t, parameters, 0: avg over trajectory errors
        q_l2 = torch.mean(torch.sqrt(torch.sum((q_gt - q_net) ** 2, 1) / torch.sum(q_gt ** 2, 1)))
        p_l2 = torch.mean(torch.sqrt(torch.sum((p_gt - p_net) ** 2, 1) / torch.sum(p_gt ** 2, 1)))
        s1_l2 = torch.mean(torch.sqrt(torch.sum((s1_gt - s1_net) ** 2, 1) / torch.sum(s1_gt ** 2, 1)))
        s2_l2 = torch.mean(torch.sqrt(torch.sum((s2_gt - s2_net) ** 2, 1) / torch.sum(s2_gt ** 2, 1)))
#         q_mse = torch.mean(torch.mean((q_net - q_gt)**2,0)/torch.mean(q_gt**2,0))
#         v_mse = torch.mean(torch.mean((v_net - v_gt)**2,0)/torch.mean(v_gt**2,0))

        # Print MSE
        print('Position relative l2 error = {:1.2e}'.format(q_l2))
        print('Momentum relative l2 error = {:1.2e}'.format(p_l2))
        print('Entropy1 relative l2 error = {:1.2e}'.format(s1_l2))
        print('Entropy2 relative l2 error = {:1.2e}'.format(s2_l2))
        

        
    elif (sys_name == '1DBurgers'):
        u_net = get_variables(z_net, sys_name)
        u_gt = get_variables(z_gt, sys_name)

        u_mse = torch.mean(torch.sqrt(torch.sum((u_gt - u_net) ** 2, 1) / torch.sum(u_gt ** 2, 1)))

        print('U relative l2 error = {:1.2e}\n'.format(u_mse))
        
    elif (sys_name == '1DHeat'):
        u_net = get_variables(z_net, sys_name)
        u_gt = get_variables(z_gt, sys_name)

        u_mse = torch.mean(torch.sqrt(torch.sum((u_gt - u_net) ** 2, 1) / torch.sum(u_gt ** 2, 1)))

        print('U relative l2 error = {:1.2e}\n'.format(u_mse))
        
        
    elif (sys_name == 'GC_SVD') or (sys_name == 'VC_SPNN_SVD'):
        u_net = get_variables(z_net, sys_name)
        u_gt = get_variables(z_gt, sys_name)

        # u_mse = torch.mean(torch.mean((u_net - u_gt) ** 2, 0))
        #u_mse = torch.mean(torch.mean((u_net - u_gt) ** 2, 0)/torch.mean((u_gt) ** 2, 0))
        u_mse = torch.mean(torch.sqrt(torch.sum((u_gt - u_net) ** 2, 1) / torch.sum(u_gt ** 2, 1)))

        print('X relative l2 error = {:1.2e}\n'.format(u_mse))




def truncate_latent(x):
    # Sort latent vector by L2 norm
    #print(x.shape) [150, 10]
    latent = np.sum(x.detach().cpu().numpy()**2, axis = 0)**0.5
    #print(latent.shape)
    latent_val = np.sort(latent) #[1,10]
    latent_idx = np.argsort(latent)

    # Select the most energetic modes
    rel_importance = latent_val/np.max(latent_val)
    latent_dim_trunc = sum(1 for i in rel_importance if i > 0.1)

    # Get the relevant latent variables (truncation)
    _, full_shape = x.shape
    latent_idx_trunc = latent_idx[full_shape-latent_dim_trunc:full_shape]

    return x[:,latent_idx_trunc], latent_idx_trunc

def all_latent(x):
    # Sort latent vector by L2 norm
    #print(x.shape)
    latent = np.sum(x.detach().cpu().numpy()**2, axis = 0)**0.5
    latent_val = np.sort(latent)
    latent_idx = np.argsort(latent)


    # Select the most energetic modes
    rel_importance = latent_val/np.max(latent_val)
    latent_dim_trunc = sum(1 for i in rel_importance if i > 0.0)

    # Get the relevant latent variables (truncation)
    _, full_shape = x.shape
    latent_idx_trunc = latent_idx[full_shape-latent_dim_trunc:full_shape]
    #print(x[:,latent_idx_trunc].shape)
    return x[:,latent_idx_trunc], latent_idx_trunc

