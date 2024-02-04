"""utils.py"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from utilities.utils import get_variables
import os
import matplotlib



    


def plot_latent_dynamics(x, dt, plot_name, output_dir):
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



def plot_latent(dEdt, dSdt, dt, plot_name, output_dir, sys_name):

    plt.clf()
    N = dEdt.shape[0]
    t_vec = np.linspace(dt,N*dt,N)

    if (sys_name == 'viscoelastic'): plot_name = '[Viscoelastic] ' + plot_name
    elif (sys_name == '1DBurgers'): plot_name = '[1DBurgers] ' + plot_name
    elif (sys_name == '1DHeat'): plot_name = '[1DHeat] ' + plot_name
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
    
    


    
def plot_results_last_tr_init(z_net, z_gt, dt, name, output_dir, N ,n_pred, sys_name):
    plt.clf()
    
    
    tstart = (N+2)*dt - n_pred*dt 
    
    t_vec = np.linspace(tstart,(N+1)*dt,n_pred)
    
    
       # N= test_final, n_pred = self.dim_t_tt

    if (sys_name == 'viscoelastic'):

        
        tstart = (N+3)*dt - n_pred*dt 
    
        t_vec = np.linspace(tstart,(N+2)*dt,n_pred)

        # Get Variables
        q_net, v_net, e_net, tau_net = get_variables(z_net, sys_name)
        q_gt, v_gt, e_gt, tau_gt = get_variables(z_gt, sys_name)
        nodes = [20-1, 40-1, 60-1, 80-1]
     
        fig, axes = plt.subplots(1,4, figsize=(20, 5))
        ax1, ax2, ax3, ax4 = axes.flatten()
        
        plt.subplots_adjust(wspace=.35)
        
        plot_name = '[VC] ' + name

        fig.suptitle(plot_name)
        

        matplotlib.rcParams['text.usetex'] = True
        plt.rc('text', usetex=True)

        ax1.plot(t_vec, q_net[:,nodes].detach().cpu(),'b')
        ax1.plot(t_vec, q_gt[:,nodes].detach().cpu(),'r--')
        l1, = ax1.plot([],[],'r--')
        l2, = ax1.plot([],[],'b')
        ax1.legend((l1, l2), ('GT','NN'), fontsize='14')
        ax1.set_ylabel('$q$ ', fontsize='20')
        ax1.tick_params(axis='y', labelsize=12)
        ax1.tick_params(axis='x', labelsize=12)
        ax1.set_xlabel('$t$ [s]', fontsize='19')
        ax1.grid()
  
        ax2.plot(t_vec, v_net[:,nodes].detach().cpu(),'b')
        ax2.plot(t_vec, v_gt[:,nodes].detach().cpu(),'r--')
        l1, = ax2.plot([],[],'r--')
        l2, = ax2.plot([],[],'b')
        ax2.legend((l1, l2), ('GT','NN'), fontsize='14')
        ax2.set_ylabel('$\mathit{v}$', fontsize='20')
        ax2.set_xlabel('$t$ [s]', fontsize='19')
        ax2.tick_params(axis='y', labelsize=12)
        ax2.tick_params(axis='x', labelsize=12)
        ax2.grid()
        
        ax3.plot(t_vec, e_net[:,nodes].detach().cpu(),'b')
        ax3.plot(t_vec, e_gt[:,nodes].detach().cpu(),'r--')
        l1, = ax3.plot([],[],'r--')
        l2, = ax3.plot([],[],'b')
        ax3.legend((l1, l2), ('GT','NN'), fontsize='14')
        ax3.set_ylabel('$e$ ', fontsize='20')
        ax3.set_xlabel('$t$ [s]', fontsize='19')
        ax3.tick_params(axis='y', labelsize=12)
        ax3.tick_params(axis='x', labelsize=12)
        ax3.grid()
        
       
        ax4.plot(t_vec, tau_net[:,nodes].detach().cpu(),'b')
        ax4.plot(t_vec, tau_gt[:,nodes].detach().cpu(),'r--')
        l1, = ax4.plot([],[],'r--')
        l2, = ax4.plot([],[],'b')
        ax4.legend((l1, l2), ('GT','NN'), fontsize='14')
        ax4.set_ylabel('$\\tau$ ', fontsize='20')
        ax4.set_xlabel('$t$ [s]', fontsize='19')
        ax4.tick_params(axis='y', labelsize=12)
        ax4.tick_params(axis='x', labelsize=12)
        ax4.grid()
        

        save_dir = os.path.join(output_dir, plot_name)
        
    elif (sys_name == 'GC'):

        # Get Variables
        q_net, p_net, s1_net, s2_net = get_variables(z_net, sys_name)
        q_gt, p_gt, s1_gt, s2_gt = get_variables(z_gt, sys_name)
        nodes = [15-1, 30-1, 60-1, 85-1]#good
    
     
        fig, axes = plt.subplots(1,4, figsize=(20, 5))
        ax1, ax2, ax3, ax4 = axes.flatten()
        plot_name = '[GC] ' + name
        
        
        plt.subplots_adjust(wspace=.35)

            
        fig.suptitle(plot_name)

      
        ax1.plot(t_vec, q_net[:,nodes].detach().cpu(),'b')
        ax1.plot(t_vec, q_gt[:,nodes].detach().cpu(),'r--')
        l1, = ax1.plot([],[],'r--')
        l2, = ax1.plot([],[],'b')
        ax1.legend((l1, l2), ('GT','Net'), fontsize='13')
        ax1.set_ylabel('$q$ ', fontsize='20')
        ax1.tick_params(axis='y', labelsize=12)
        ax1.tick_params(axis='x', labelsize=12)
        ax1.set_xlabel('$t$ [s]', fontsize='19')
        ax1.grid()
  
        ax2.plot(t_vec, p_net[:,nodes].detach().cpu(),'b')
        ax2.plot(t_vec, p_gt[:,nodes].detach().cpu(),'r--')
        l1, = ax2.plot([],[],'r--')
        l2, = ax2.plot([],[],'b')
        ax2.legend((l1, l2), ('GT','Net'), fontsize='13')
        ax2.set_ylabel('$p$ ', fontsize='20')
        ax2.set_xlabel('$t$ [s]', fontsize='19')
        ax2.tick_params(axis='y', labelsize=12)
        ax2.tick_params(axis='x', labelsize=12)
        ax2.grid()
        
        ax3.plot(t_vec, s1_net[:,nodes].detach().cpu(),'b')
        ax3.plot(t_vec, s1_gt[:,nodes].detach().cpu(),'r--')
        l1, = ax3.plot([],[],'r--')
        l2, = ax3.plot([],[],'b')
        ax3.legend((l1, l2), ('GT','Net'), fontsize='13')
        ax3.set_ylabel('$S_1$ ', fontsize='20')
        ax3.set_xlabel('$t$ [s]', fontsize='19')
        ax3.tick_params(axis='y', labelsize=12)
        ax3.tick_params(axis='x', labelsize=12)
        ax3.grid()
       
        ax4.plot(t_vec, s2_net[:,nodes].detach().cpu(),'b')
        ax4.plot(t_vec, s2_gt[:,nodes].detach().cpu(),'r--')
        l1, = ax4.plot([],[],'r--')
        l2, = ax4.plot([],[],'b')
        ax4.legend((l1, l2), ('GT','Net'), fontsize='13')
        ax4.set_ylabel('$S_2$', fontsize='20')
        ax4.set_xlabel('$t$ [s]', fontsize='19')
        ax4.tick_params(axis='y', labelsize=12)
        ax4.tick_params(axis='x', labelsize=12)
        ax4.grid()

        save_dir = os.path.join(output_dir, plot_name)
        
    plt.savefig(save_dir)
    plt.clf()

    


