"""utils.py"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from utilities.utils import get_variables
import os


    


def plot_latent_visco(x, dt, plot_name, output_dir):
    plt.clf()
    N = x.shape[0]
    t_vec = np.linspace(dt,N*dt,N)

    fig, axes = plt.subplots(1,1, figsize=(5, 5))
    fig.suptitle(plot_name)

    axes.plot(t_vec, x.detach().cpu())
    axes.set_ylabel('$x$ [-]')
    axes.set_xlabel('$t$ [s]')
    axes.grid()

    save_dir = os.path.join(output_dir, plot_name)
    plt.savefig(save_dir)
    plt.clf()


def plot_latent_tire(x_q, x_v, x_sigma, dt, plot_name, output_dir):
    plt.clf()
    N = x_q.shape[0]
    t_vec = np.linspace(dt,N*dt,N)

    fig, axes = plt.subplots(1,3, figsize=(15, 5))
    ax1, ax2, ax3 = axes.flatten()
    fig.suptitle(plot_name)

    ax1.plot(t_vec, x_q.detach().cpu())
    ax1.set_ylabel('$x_q$ [-]')
    ax1.set_xlabel('$t$ [s]')
    ax1.grid()

    ax2.plot(t_vec, x_v.detach().cpu())
    ax2.set_ylabel('$x_v$ [-]')
    ax2.set_xlabel('$t$ [s]')
    ax2.grid()

    ax3.plot(t_vec, x_sigma.detach().cpu())
    ax3.set_ylabel('$x_\sigma$ [-]')
    ax3.set_xlabel('$t$ [s]')
    ax3.grid()

    save_dir = os.path.join(output_dir, plot_name)
    plt.savefig(save_dir)
    plt.clf()


#def plot_latent(x_spnn, x_trunc, dEdt, dSdt, dt, plot_name, output_dir, sys_name):
def plot_latent(dEdt, dSdt, dt, plot_name, output_dir, sys_name):

    plt.clf()
    N = dEdt.shape[0]
    t_vec = np.linspace(dt,N*dt,N)

    if (sys_name == 'viscoelastic'): plot_name = '[Viscoelastic] ' + plot_name
    elif (sys_name == 'rolling_tire'): plot_name = '[Rolling Tire] ' + plot_name
    elif (sys_name == '1DBurgers'): plot_name = '[1DBurgers] ' + plot_name
    elif (sys_name == 'GC_SVD'): plot_name = '[GC_SVD] ' + plot_name
    elif (sys_name == 'GC_SVD_concat'): plot_name = '[GC_SVD_concat] ' + plot_name
    elif (sys_name == 'GC'): plot_name = '[GC] ' + plot_name
    elif (sys_name == '2DBurgers'): plot_name = '[2DBurgers] ' + plot_name

    fig, ax = plt.subplots(1,1, figsize=(10, 5))


    ax.plot(t_vec, dEdt.detach().cpu(),'r')
    ax.plot(t_vec, dSdt.detach().cpu(),'b')
    l1, = ax.plot([],[],'r')
    l2, = ax.plot([],[],'b')
    ax.legend((l1, l2), ('dEdt','dSdt'))
    ax.set_ylabel('$dEdt, dSdt$ [-]')
    ax.set_xlabel('$t$ [s]')
    ax.grid()

    save_dir = os.path.join(output_dir, plot_name)
    plt.savefig(save_dir)
    plt.clf()
    
    

def plot_pred_errors(z_net, z_gt, dt, name, output_dir, N ,n_pred, sys_name):
    
    plt.clf()
    
    
    tstart = (N+2)*dt - n_pred*dt
    
    t_vec = np.linspace(tstart,(N+1)*dt,n_pred)
    
    if (sys_name == 'viscoelastic'):

        # Get Variables
        q_net, v_net, e_net, tau_net = get_variables(z_net, sys_name)
        q_gt, v_gt, e_gt, tau_gt = get_variables(z_gt, sys_name)
        
        
        q_l2_t = torch.sqrt(torch.sum((q_gt - q_net) ** 2, 1) / torch.sum(q_gt ** 2, 1))
        v_l2_t = torch.sqrt(torch.sum((v_gt - v_net) ** 2, 1) / torch.sum(v_gt ** 2, 1))
        e_l2_t = torch.sqrt(torch.sum((e_gt - e_net) ** 2, 1) / torch.sum(e_gt ** 2, 1))
        tau_l2_t = torch.sqrt(torch.sum((tau_gt - tau_net) ** 2, 1) / torch.sum(tau_gt ** 2, 1))

        
     
        fig, axes = plt.subplots(1,4, figsize=(20, 5))
        ax1, ax2, ax3, ax4 = axes.flatten()
        plot_name = '[VC] ' + name
        fig.suptitle(plot_name)

      
        ax1.plot(t_vec, q_l2_t.detach().cpu(),'b')
        ax1.set_ylabel('rel. $l_2$ for $q$')
        ax1.set_xlabel('$t$')
        ax1.grid()
  
        ax2.plot(t_vec, v_l2_t.detach().cpu(),'b')
        ax2.set_ylabel('rel. $l_2$ for $v$')
        ax2.set_xlabel('$t$')
        ax2.grid()
        
        ax3.plot(t_vec, e_l2_t.detach().cpu(),'b')
        ax3.set_ylabel('rel. $l_2$ for $e$')
        ax3.set_xlabel('$t$')
        ax3.grid()
       
        ax4.plot(t_vec, tau_l2_t.detach().cpu(),'b')
        ax4.set_ylabel('rel. $l_2$ for $\tau$')
        ax4.set_xlabel('$t$')
        ax4.grid()

        save_dir = os.path.join(output_dir, plot_name)
        
    elif (sys_name == 'GC'):

        # Get Variables
        q_net, p_net, s1_net, s2_net = get_variables(z_net, sys_name)
        q_gt, p_gt, s1_gt, s2_gt = get_variables(z_gt, sys_name)
        
        
        q_l2_t = torch.sqrt(torch.sum((q_gt - q_net) ** 2, 1) / torch.sum(q_gt ** 2, 1))
        p_l2_t = torch.sqrt(torch.sum((p_gt - p_net) ** 2, 1) / torch.sum(p_gt ** 2, 1))
        s1_l2_t = torch.sqrt(torch.sum((s1_gt - s1_net) ** 2, 1) / torch.sum(s1_gt ** 2, 1))
        s2_l2_t = torch.sqrt(torch.sum((s2_gt - s2_net) ** 2, 1) / torch.sum(s2_gt ** 2, 1))

     
        fig, axes = plt.subplots(1,4, figsize=(20, 5))
        ax1, ax2, ax3, ax4 = axes.flatten()
        plot_name = '[GC] ' + name
        fig.suptitle(plot_name)

      
        ax1.plot(t_vec, q_l2_t.detach().cpu(),'b')
        ax1.set_ylabel('rel. $l_2$ for $q$')
        ax1.set_xlabel('$t$')
        ax1.grid()
  
        ax2.plot(t_vec, p_l2_t.detach().cpu(),'b')
        ax2.set_ylabel('rel. $l_2$ for $p$')
        ax2.set_xlabel('$t$')
        ax2.grid()
        
        ax3.plot(t_vec, s1_l2_t.detach().cpu(),'b')
        ax3.set_ylabel('rel. $l_2$ for $s_1$')
        ax3.set_xlabel('$t$')
        ax3.grid()
       
        ax4.plot(t_vec, s2_l2_t.detach().cpu(),'b')
        ax4.set_ylabel('rel. $l_2$ for $s_2$')
        ax4.set_xlabel('$t$')
        ax4.grid()

        save_dir = os.path.join(output_dir, plot_name)
        
    plt.savefig(save_dir)
    plt.clf()
    
    

    
def plot_results_last_tr_init(z_net, z_gt, dt, name, output_dir, N ,n_pred, sys_name):
    plt.clf()
    
    
    tstart = (N+2)*dt - n_pred*dt 
    
    t_vec = np.linspace(tstart,(N+1)*dt,n_pred)
        

    if (sys_name == 'viscoelastic'):

        # Get Variables
        q_net, v_net, e_net, tau_net = get_variables(z_net, sys_name)
        q_gt, v_gt, e_gt, tau_gt = get_variables(z_gt, sys_name)
        nodes = [20-1, 40-1, 60-1, 80-1]
     
        fig, axes = plt.subplots(1,4, figsize=(20, 5))
        ax1, ax2, ax3, ax4 = axes.flatten()
        plot_name = '[VC] ' + name
        fig.suptitle(plot_name)

      
        ax1.plot(t_vec, q_net[:,nodes].detach().cpu(),'b')
        ax1.plot(t_vec, q_gt[:,nodes].detach().cpu(),'k--')
        l1, = ax1.plot([],[],'k--')
        l2, = ax1.plot([],[],'b')
        ax1.legend((l1, l2), ('GT','Net'))
        ax1.set_ylabel('$q$ [-]')
        ax1.set_xlabel('$t$ [s]')
        ax1.grid()
  
        ax2.plot(t_vec, v_net[:,nodes].detach().cpu(),'b')
        ax2.plot(t_vec, v_gt[:,nodes].detach().cpu(),'k--')
        l1, = ax2.plot([],[],'k--')
        l2, = ax2.plot([],[],'b')
        ax2.legend((l1, l2), ('GT','Net'))
        ax2.set_ylabel('$v$ [-]')
        ax2.set_xlabel('$t$ [s]')
        ax2.grid()
        
        ax3.plot(t_vec, e_net[:,nodes].detach().cpu(),'b')
        ax3.plot(t_vec, e_gt[:,nodes].detach().cpu(),'k--')
        l1, = ax3.plot([],[],'k--')
        l2, = ax3.plot([],[],'b')
        ax3.legend((l1, l2), ('GT','Net'))
        ax3.set_ylabel('$e$ [-]')
        ax3.set_xlabel('$t$ [s]')
        ax3.grid()
       
        ax4.plot(t_vec, tau_net[:,nodes].detach().cpu(),'b')
        ax4.plot(t_vec, tau_gt[:,nodes].detach().cpu(),'k--')
        l1, = ax4.plot([],[],'k--')
        l2, = ax4.plot([],[],'b')
        ax4.legend((l1, l2), ('GT','Net'))
        ax4.set_ylabel('$\tau$ [-]')
        ax4.set_xlabel('$t$ [s]')
        ax4.grid()

        save_dir = os.path.join(output_dir, plot_name)
        
    elif (sys_name == 'GC'):

        # Get Variables
        q_net, p_net, s1_net, s2_net = get_variables(z_net, sys_name)
        q_gt, p_gt, s1_gt, s2_gt = get_variables(z_gt, sys_name)
#         nodes = [20-1, 40-1, 60-1, 80-1]
#         nodes = [15-1, 30-1, 70-1, 85-1]#okay
        nodes = [15-1, 30-1, 60-1, 85-1]#good
     
        fig, axes = plt.subplots(1,4, figsize=(20, 5))
        ax1, ax2, ax3, ax4 = axes.flatten()
        plot_name = '[GC] ' + name
        fig.suptitle(plot_name)

      
        ax1.plot(t_vec, q_net[:,nodes].detach().cpu(),'b')
        ax1.plot(t_vec, q_gt[:,nodes].detach().cpu(),'k--')
        l1, = ax1.plot([],[],'k--')
        l2, = ax1.plot([],[],'b')
        ax1.legend((l1, l2), ('GT','Net'))
        ax1.set_ylabel('$q$ [-]')
        ax1.set_xlabel('$t$ [s]')
        ax1.grid()
  
        ax2.plot(t_vec, p_net[:,nodes].detach().cpu(),'b')
        ax2.plot(t_vec, p_gt[:,nodes].detach().cpu(),'k--')
        l1, = ax2.plot([],[],'k--')
        l2, = ax2.plot([],[],'b')
        ax2.legend((l1, l2), ('GT','Net'))
        ax2.set_ylabel('$p$ [-]')
        ax2.set_xlabel('$t$ [s]')
        ax2.grid()
        
        ax3.plot(t_vec, s1_net[:,nodes].detach().cpu(),'b')
        ax3.plot(t_vec, s1_gt[:,nodes].detach().cpu(),'k--')
        l1, = ax3.plot([],[],'k--')
        l2, = ax3.plot([],[],'b')
        ax3.legend((l1, l2), ('GT','Net'))
        ax3.set_ylabel('$S_1$ [-]')
        ax3.set_xlabel('$t$ [s]')
        ax3.grid()
       
        ax4.plot(t_vec, s2_net[:,nodes].detach().cpu(),'b')
        ax4.plot(t_vec, s2_gt[:,nodes].detach().cpu(),'k--')
        l1, = ax4.plot([],[],'k--')
        l2, = ax4.plot([],[],'b')
        ax4.legend((l1, l2), ('GT','Net'))
        ax4.set_ylabel('$S_2$ [-]')
        ax4.set_xlabel('$t$ [s]')
        ax4.grid()

        save_dir = os.path.join(output_dir, plot_name)
        
    plt.savefig(save_dir)
    plt.clf()

    
# def plot_results_test(z_net, z_gt, dt, name, output_dir,train_indices,test_indices, sys_name):
#     plt.clf()
#     N = z_gt.shape[0]
#     t_vec = np.linspace(dt,N*dt,N)
    
#     test_indices = sorted(test_indices)
    
# #     print(t_vec)
# #     print(test_indices)
#     t_vec = t_vec[test_indices]
# #     print(t_vec)
    
#     if (sys_name == 'GC'):

#         # Get Variables
#         q_net, p_net, s1_net, s2_net = get_variables(z_net, sys_name)
#         q_gt, p_gt, s1_gt, s2_gt = get_variables(z_gt, sys_name)
        
        
#         q_net = q_net[test_indices,:]
#         p_net = p_net[test_indices,:]
#         s1_net = s1_net[test_indices,:]
#         s2_net = s2_net[test_indices,:]
        
#         q_gt = q_gt[test_indices,:]
#         p_gt = p_gt[test_indices,:]
#         s1_gt = s1_gt[test_indices,:]
#         s2_gt = s2_gt[test_indices,:]


# #         nodes = [20-1, 40-1, 60-1, 80-1]
#         nodes = [15-1, 30-1, 60-1, 85-1

#         fig, axes = plt.subplots(1,4, figsize=(20, 5))
#         ax1, ax2, ax3, ax4 = axes.flatten()
#         plot_name = '[GC] ' + name
#         fig.suptitle(plot_name)


#         ax1.plot(t_vec, q_net[:,nodes].detach().cpu(),'b')
#         ax1.plot(t_vec, q_gt[:,nodes].detach().cpu(),'k--')
#         l1, = ax1.plot([],[],'k--')
#         l2, = ax1.plot([],[],'b')
#         ax1.legend((l1, l2), ('GT','Net'))
#         ax1.set_ylabel('$q$ [-]')
#         ax1.set_xlabel('$t$ [s]')
#         ax1.grid()

#         ax2.plot(t_vec, p_net[:,nodes].detach().cpu(),'b')
#         ax2.plot(t_vec, p_gt[:,nodes].detach().cpu(),'k--')
#         l1, = ax2.plot([],[],'k--')
#         l2, = ax2.plot([],[],'b')
#         ax2.legend((l1, l2), ('GT','Net'))
#         ax2.set_ylabel('$p$ [-]')
#         ax2.set_xlabel('$t$ [s]')
#         ax2.grid()

#         ax3.plot(t_vec, s1_net[:,nodes].detach().cpu(),'b')
#         ax3.plot(t_vec, s1_gt[:,nodes].detach().cpu(),'k--')
#         l1, = ax3.plot([],[],'k--')
#         l2, = ax3.plot([],[],'b')
#         ax3.legend((l1, l2), ('GT','Net'))
#         ax3.set_ylabel('$S_1$ [-]')
#         ax3.set_xlabel('$t$ [s]')
#         ax3.grid()

#         ax4.plot(t_vec, s2_net[:,nodes].detach().cpu(),'b')
#         ax4.plot(t_vec, s2_gt[:,nodes].detach().cpu(),'k--')
#         l1, = ax4.plot([],[],'k--')
#         l2, = ax4.plot([],[],'b')
#         ax4.legend((l1, l2), ('GT','Net'))
#         ax4.set_ylabel('$S_2$ [-]')
#         ax4.set_xlabel('$t$ [s]')
#         ax4.grid()

#         save_dir = os.path.join(output_dir, plot_name)
#     plt.savefig(save_dir)
#     plt.clf()

    
def plot_results(z_net, z_gt, dt, name, output_dir, sys_name):
    plt.clf()
    N = z_gt.shape[0]
    t_vec = np.linspace(dt,N*dt,N)

    if (sys_name == 'viscoelastic'):

        # Get Variables
        q_net, v_net, e_net, tau_net = get_variables(z_net, sys_name)
        q_gt, v_gt, e_gt, tau_gt = get_variables(z_gt, sys_name)
        nodes = [20-1, 40-1, 60-1, 80-1]
     
        fig, axes = plt.subplots(1,4, figsize=(20, 5))
        ax1, ax2, ax3, ax4 = axes.flatten()
        plot_name = '[VC] ' + name
        fig.suptitle(plot_name)

      
        ax1.plot(t_vec, q_net[:,nodes].detach().cpu(),'b')
        ax1.plot(t_vec, q_gt[:,nodes].detach().cpu(),'k--')
        l1, = ax1.plot([],[],'k--')
        l2, = ax1.plot([],[],'b')
        ax1.legend((l1, l2), ('GT','Net'))
        ax1.set_ylabel('$q$ [-]')
        ax1.set_xlabel('$t$ [s]')
        ax1.grid()
  
        ax2.plot(t_vec, v_net[:,nodes].detach().cpu(),'b')
        ax2.plot(t_vec, v_gt[:,nodes].detach().cpu(),'k--')
        l1, = ax2.plot([],[],'k--')
        l2, = ax2.plot([],[],'b')
        ax2.legend((l1, l2), ('GT','Net'))
        ax2.set_ylabel('$v$ [-]')
        ax2.set_xlabel('$t$ [s]')
        ax2.grid()
        
        ax3.plot(t_vec, e_net[:,nodes].detach().cpu(),'b')
        ax3.plot(t_vec, e_gt[:,nodes].detach().cpu(),'k--')
        l1, = ax3.plot([],[],'k--')
        l2, = ax3.plot([],[],'b')
        ax3.legend((l1, l2), ('GT','Net'))
        ax3.set_ylabel('$e$ [-]')
        ax3.set_xlabel('$t$ [s]')
        ax3.grid()
       
        ax4.plot(t_vec, tau_net[:,nodes].detach().cpu(),'b')
        ax4.plot(t_vec, tau_gt[:,nodes].detach().cpu(),'k--')
        l1, = ax4.plot([],[],'k--')
        l2, = ax4.plot([],[],'b')
        ax4.legend((l1, l2), ('GT','Net'))
        ax4.set_ylabel('$\tau$ [-]')
        ax4.set_xlabel('$t$ [s]')
        ax4.grid()

        save_dir = os.path.join(output_dir, plot_name)
    if (sys_name == '2DBurgers'):

        # Get Variables
        u_net, v_net = get_variables(z_net, sys_name)
        u_gt, v_gt = get_variables(z_gt, sys_name)
        
        if z_gt.shape[1] == 3200:
            nodes = [400-1, 800-1, 1200-1, 1600-1]
        elif z_gt.shape[1] == 1800:
            nodes = [225-1, 450-1, 775-1, 900-1]
            
            
        fig, axes = plt.subplots(1,2, figsize=(20, 5))
        ax1, ax2 = axes.flatten()
        plot_name = '[2DBG] ' + name
        fig.suptitle(plot_name)

      
        ax1.plot(t_vec, u_net[:,nodes].detach().cpu(),'b')
        ax1.plot(t_vec, u_gt[:,nodes].detach().cpu(),'k--')
        l1, = ax1.plot([],[],'k--')
        l2, = ax1.plot([],[],'b')
        ax1.legend((l1, l2), ('GT','Net'))
        ax1.set_ylabel('$u$ [-]')
        ax1.set_xlabel('$t$ [s]')
        ax1.grid()
  
        ax2.plot(t_vec, v_net[:,nodes].detach().cpu(),'b')
        ax2.plot(t_vec, v_gt[:,nodes].detach().cpu(),'k--')
        l1, = ax2.plot([],[],'k--')
        l2, = ax2.plot([],[],'b')
        ax2.legend((l1, l2), ('GT','Net'))
        ax2.set_ylabel('$v$ [-]')
        ax2.set_xlabel('$t$ [s]')
        ax2.grid()

        save_dir = os.path.join(output_dir, plot_name)
        
    elif (sys_name == 'GC'):

        # Get Variables
        q_net, p_net, s1_net, s2_net = get_variables(z_net, sys_name)
        q_gt, p_gt, s1_gt, s2_gt = get_variables(z_gt, sys_name)
        #nodes = [20-1, 40-1, 60-1, 80-1]
        nodes = [15-1, 30-1, 60-1, 85-1]
     
        fig, axes = plt.subplots(1,4, figsize=(20, 5))
        ax1, ax2, ax3, ax4 = axes.flatten()
        plot_name = '[GC] ' + name
        fig.suptitle(plot_name)

      
        ax1.plot(t_vec, q_net[:,nodes].detach().cpu(),'b')
        ax1.plot(t_vec, q_gt[:,nodes].detach().cpu(),'k--')
        l1, = ax1.plot([],[],'k--')
        l2, = ax1.plot([],[],'b')
        ax1.legend((l1, l2), ('GT','Net'))
        ax1.set_ylabel('$q$ [-]')
        ax1.set_xlabel('$t$ [s]')
        ax1.grid()
  
        ax2.plot(t_vec, p_net[:,nodes].detach().cpu(),'b')
        ax2.plot(t_vec, p_gt[:,nodes].detach().cpu(),'k--')
        l1, = ax2.plot([],[],'k--')
        l2, = ax2.plot([],[],'b')
        ax2.legend((l1, l2), ('GT','Net'))
        ax2.set_ylabel('$p$ [-]')
        ax2.set_xlabel('$t$ [s]')
        ax2.grid()
        
        ax3.plot(t_vec, s1_net[:,nodes].detach().cpu(),'b')
        ax3.plot(t_vec, s1_gt[:,nodes].detach().cpu(),'k--')
        l1, = ax3.plot([],[],'k--')
        l2, = ax3.plot([],[],'b')
        ax3.legend((l1, l2), ('GT','Net'))
        ax3.set_ylabel('$S_1$ [-]')
        ax3.set_xlabel('$t$ [s]')
        ax3.grid()
       
        ax4.plot(t_vec, s2_net[:,nodes].detach().cpu(),'b')
        ax4.plot(t_vec, s2_gt[:,nodes].detach().cpu(),'k--')
        l1, = ax4.plot([],[],'k--')
        l2, = ax4.plot([],[],'b')
        ax4.legend((l1, l2), ('GT','Net'))
        ax4.set_ylabel('$S_2$ [-]')
        ax4.set_xlabel('$t$ [s]')
        ax4.grid()

        save_dir = os.path.join(output_dir, plot_name)
        
   

    elif (sys_name == '1DBurgers'):
        # Get Variables
        u_net = get_variables(z_net, sys_name)
        u_gt = get_variables(z_gt, sys_name)

        if z_gt.shape[1] == 1001:
            nodes = [250-1, 500-1, 750-1, 1000-1]
        elif z_gt.shape[1] == 601:
            nodes = [150-1, 300-1, 450-1, 600-1]
        elif z_gt.shape[1] == 501:
            nodes = [125 - 1, 250 - 1, 375 - 1, 500 - 1]
        elif z_gt.shape[1] == 201:
            nodes = [50 - 1, 100 - 1, 150 - 1, 200 - 1]
        elif z_gt.shape[1] == 301:
            nodes = [70 - 1, 140 - 1, 220 - 1, 300 - 1]
        elif z_gt.shape[1] == 101:
            nodes = [25 - 1, 50 - 1, 75 - 1, 100 - 1]

        fig, ax1 = plt.subplots(1,1, figsize=(10, 10))
        #ax1 = axes.flatten()
        plot_name = '[BG] ' + name
        fig.suptitle(plot_name)

        ax1.plot(t_vec, u_net[:,nodes].detach().cpu(),'b')
        ax1.plot(t_vec, u_gt[:,nodes].detach().cpu(),'k--')
        l1, = ax1.plot([],[],'k--')
        l2, = ax1.plot([],[],'b')
        ax1.legend((l1, l2), ('GT','Net'))
        ax1.set_ylabel('$u$ [-]')
        ax1.set_xlabel('$t$ [s]')
        ax1.grid()

        save_dir = os.path.join(output_dir, plot_name)
   
  

    elif (sys_name == 'rolling_tire'):
        
        #print(z_gt.shape[1])
        # Only 4 Nodes to plot
        if z_gt.shape[1] == 49680:
            nodes = [1000-1, 2000-1, 3000-1, 4000-1]
        elif z_gt.shape[1] == 24840:
            nodes = [500-1, 1000-1, 1500-1, 2000-1]
        elif z_gt.shape[1] == 12420:
            nodes = [250-1, 500-1, 750-1, 1000-1]
        # Get Variables
        q_net, v_net, sigma_net = get_variables(z_net, sys_name)
        q_gt, v_gt, sigma_gt = get_variables(z_gt, sys_name)

        # Position and Velocity Figure
        fig, axes = plt.subplots(2,3, figsize=(20, 10))
        ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
        plot_name = '[Rolling Tire] ' + name + ' (Position and Velocity)'
        fig.suptitle(plot_name)
     
        ax1.plot(t_vec, q_net[0,:,nodes].detach().cpu(),'b')
        ax1.plot(t_vec, q_gt[0,:,nodes].detach().cpu(),'k--')
        l1, = ax1.plot([],[],'k--')
        l2, = ax1.plot([],[],'b')
        ax1.legend((l1, l2), ('GT','Net'))
        ax1.set_ylabel('$q_1$ [m]')
        ax1.set_xlabel('$t$ [s]')
        ax1.grid()

        ax2.plot(t_vec, q_net[1,:,nodes].detach().cpu(),'b')
        ax2.plot(t_vec, q_gt[1,:,nodes].detach().cpu(),'k--')
        l1, = ax2.plot([],[],'k--')
        l2, = ax2.plot([],[],'b')
        ax2.legend((l1, l2), ('GT','Net'))
        ax2.set_ylabel('$q_2$ [m]')
        ax2.set_xlabel('$t$ [s]')
        ax2.grid()

        ax3.plot(t_vec, q_net[2,:,nodes].detach().cpu(),'b')
        ax3.plot(t_vec, q_gt[2,:,nodes].detach().cpu(),'k--')
        l1, = ax3.plot([],[],'k--')
        l2, = ax3.plot([],[],'b')
        ax3.legend((l1, l2), ('GT','Net'))
        ax3.set_ylabel('$q_3$ [m]')
        ax3.set_xlabel('$t$ [s]')
        ax3.grid()

        ax4.plot(t_vec, v_net[0,:,nodes].detach().cpu(),'b')
        ax4.plot(t_vec, v_gt[0,:,nodes].detach().cpu(),'k--')
        l1, = ax4.plot([],[],'k--')
        l2, = ax4.plot([],[],'b')
        ax4.legend((l1, l2), ('GT','Net'))
        ax4.set_ylabel('$v_1$ [m/s]')
        ax4.set_xlabel('$t$ [s]')
        ax4.grid()

        ax5.plot(t_vec, v_net[1,:,nodes].detach().cpu(),'b')
        ax5.plot(t_vec, v_gt[1,:,nodes].detach().cpu(),'k--')
        l1, = ax5.plot([],[],'k--')
        l2, = ax5.plot([],[],'b')
        ax5.legend((l1, l2), ('GT','Net'))
        ax5.set_ylabel('$v_2$ [m/s]')
        ax5.set_xlabel('$t$ [s]')
        ax5.grid()

        ax6.plot(t_vec, v_net[2,:,nodes].detach().cpu(),'b')
        ax6.plot(t_vec, v_gt[2,:,nodes].detach().cpu(),'k--')
        l1, = ax6.plot([],[],'k--')
        l2, = ax6.plot([],[],'b')
        ax6.legend((l1, l2), ('GT','Net'))
        ax6.set_ylabel('$v_3$ [m/s]')
        ax6.set_xlabel('$t$ [s]')
        ax6.grid()

        save_dir = os.path.join(output_dir, plot_name)
        plt.savefig(save_dir)
        plt.clf()

        # Stress Tensor Figure
        fig, axes = plt.subplots(2,3, figsize=(20, 10))
        ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
        plot_name = '[Rolling Tire] ' + name + ' (Stress Tensor)'
        fig.suptitle(plot_name)
        
        ax1.plot(t_vec, sigma_net[0,:,nodes].detach().cpu(),'b')
        ax1.plot(t_vec, sigma_gt[0,:,nodes].detach().cpu(),'k--')
        l1, = ax1.plot([],[],'k--')
        l2, = ax1.plot([],[],'b')
        ax1.legend((l1, l2), ('GT','Net'))
        ax1.set_ylabel('$\\sigma_{11}$ [MPa]')
        ax1.set_xlabel('$t$ [s]')
        ax1.grid()
        
        ax2.plot(t_vec, sigma_net[1,:,nodes].detach().cpu(),'b')
        ax2.plot(t_vec, sigma_gt[1,:,nodes].detach().cpu(),'k--')
        l1, = ax2.plot([],[],'k--')
        l2, = ax2.plot([],[],'b')
        ax2.legend((l1, l2), ('GT','Net'))
        ax2.set_ylabel('$\\sigma_{22}$ [MPa]')
        ax2.set_xlabel('$t$ [s]')
        ax2.grid()
       
        ax3.plot(t_vec, sigma_net[2,:,nodes].detach().cpu(),'b')
        ax3.plot(t_vec, sigma_gt[2,:,nodes].detach().cpu(),'k--')
        l1, = ax3.plot([],[],'k--')
        l2, = ax3.plot([],[],'b')
        ax3.legend((l1, l2), ('GT','Net'))
        ax3.set_ylabel('$\\sigma_{33}$ [MPa]')
        ax3.set_xlabel('$t$ [s]')
        ax3.grid()
        
        ax4.plot(t_vec, sigma_net[3,:,nodes].detach().cpu(),'b')
        ax4.plot(t_vec, sigma_gt[3,:,nodes].detach().cpu(),'k--')
        l1, = ax4.plot([],[],'k--')
        l2, = ax4.plot([],[],'b')
        ax4.legend((l1, l2), ('GT','Net'))
        ax4.set_ylabel('$\\sigma_{12}$ [MPa]')
        ax4.set_xlabel('$t$ [s]')
        ax4.grid()
        
        ax5.plot(t_vec, sigma_net[4,:,nodes].detach().cpu(),'b')
        ax5.plot(t_vec, sigma_gt[4,:,nodes].detach().cpu(),'k--')
        l1, = ax5.plot([],[],'k--')
        l2, = ax5.plot([],[],'b')
        ax5.legend((l1, l2), ('GT','Net'))
        ax5.set_ylabel('$\\sigma_{13}$ [MPa]')
        ax5.set_xlabel('$t$ [s]')
        ax5.grid()
        
        ax6.plot(t_vec, sigma_net[5,:,nodes].detach().cpu(),'b')
        ax6.plot(t_vec, sigma_gt[5,:,nodes].detach().cpu(),'k--')
        l1, = ax6.plot([],[],'k--')
        l2, = ax6.plot([],[],'b')
        ax6.legend((l1, l2), ('GT','Net'))
        ax6.set_ylabel('$\\sigma_{23}$ [MPa]')
        ax6.set_xlabel('$t$ [s]')
        ax6.grid()

        save_dir = os.path.join(output_dir, plot_name)

    plt.savefig(save_dir)
    plt.clf()
    
    
