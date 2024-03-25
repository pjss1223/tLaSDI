"""utils.py"""

import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import torch
from utilities.utils import get_variables
import os
import matplotlib

plt.style.use('science')
    


def plot_latent_dynamics(x, dt, plot_name, output_dir):
    plt.clf()
    N = x.shape[0]
    t_vec = np.linspace(dt,N*dt,N)

    fig, axes = plt.subplots(1,1, figsize=(5, 5))
    fig.suptitle(plot_name)

    axes.plot(t_vec, x.detach().cpu())
    axes.set_ylabel('$x$ [-]')
    axes.set_xlabel('$t$')
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
    ax.set_xlabel('$t$')
    ax.grid()

    save_dir = os.path.join(output_dir, plot_name)
    plt.savefig(save_dir)
    plt.clf()
    
    


    
def plot_test_results(z_net, z_gt, dt, name, output_dir, N ,n_pred, sys_name,ROM_model):
    plt.clf()
    
    
    tstart = (N+2)*dt - n_pred*dt 
    
    t_vec = np.linspace(tstart,(N+1)*dt,n_pred)
    
    if ROM_model == 'tLaSDI':
        color = 'b'
    elif ROM_model == 'TA-ROM':
        color = 'r'
    elif ROM_model == 'Vanilla-FNN': 
        color = 'g'
        
    
    
       # N= test_final, n_pred = self.dim_t_tt

    if (sys_name == 'viscoelastic'):
        
        if ROM_model == 'tLaSDI':
            save_dir_tmp = 'tLaSDI_VC_sols'
        elif ROM_model == 'TA-ROM':
            save_dir_tmp = 'TA_ROM_VC_sols'
        elif ROM_model == 'Vanilla-FNN': 
            save_dir_tmp = 'Vanilla_FNN_VC_sols'
        
        

        
        tstart = (N+3)*dt - n_pred*dt 
    
        t_vec = np.linspace(tstart,(N+2)*dt,n_pred)

        # Get Variables
        q_net, v_net, e_net, tau_net = get_variables(z_net, sys_name)
        q_gt, v_gt, e_gt, tau_gt = get_variables(z_gt, sys_name)
        nodes = [20-1, 40-1, 60-1, 80-1]
        #nodes = [i for i in range(10, 20)]
     
        fig, axes = plt.subplots(1,4, figsize=(20, 5))
        ax1, ax2, ax3, ax4 = axes.flatten()
        
        plt.subplots_adjust(wspace=.05)
        
        plot_name = '[VC] ' + name

        #fig.suptitle(plot_name)
        

        matplotlib.rcParams['text.usetex'] = True
        plt.rc('text', usetex=True)

        ax1.plot(t_vec, q_net[:,nodes].detach().cpu(),color,linewidth=2.5)
        ax1.plot(t_vec, q_gt[:,nodes].detach().cpu(),'k--',linewidth=2.5)
        l1, = ax1.plot([],[],'k--',linewidth=2.5)
        l2, = ax1.plot([],[],color,linewidth=2.5)
#         ax1.legend((l1, l2), ('GT',ROM_model), fontsize='16')
        ax1.set_ylabel('$q$ ', fontsize='24')
        ax1.set_xlim(0.9,1)
        ax1.tick_params(axis='y', labelsize=16)
        ax1.tick_params(axis='x', labelsize=16)
        ax1.set_xlabel('$t$', fontsize='24')
        ax1.grid()
  
        ax2.plot(t_vec, v_net[:,nodes].detach().cpu(),color,linewidth=2.5)
        ax2.plot(t_vec, v_gt[:,nodes].detach().cpu(),'k--',linewidth=2.5)
        l1, = ax2.plot([],[],'k--',linewidth=2.5)
        l2, = ax2.plot([],[],color,linewidth=2.5)
        #ax2.legend((l1, l2), ('GT',ROM_model), fontsize='16')
        ax2.set_ylabel('$\mathit{v}$', fontsize='24')
        ax2.set_xlim(0.9,1)
        ax2.set_xlabel('$t$', fontsize='24')
        ax2.tick_params(axis='y', labelsize=16)
        ax2.tick_params(axis='x', labelsize=16)
        ax2.grid()
        
        ax3.plot(t_vec, e_net[:,nodes].detach().cpu(),color,linewidth=2.5)
        ax3.plot(t_vec, e_gt[:,nodes].detach().cpu(),'k--',linewidth=2.5)
        l1, = ax3.plot([],[],'k--',linewidth=2.5)
        l2, = ax3.plot([],[],color,linewidth=2.5)
        #ax3.legend((l1, l2), ('GT',ROM_model), fontsize='16')
        ax3.set_ylabel('$e$ ', fontsize='24')
        ax3.set_xlabel('$t$', fontsize='24')
        ax3.set_xlim(0.9,1)
        ax3.tick_params(axis='y', labelsize=16)
        ax3.tick_params(axis='x', labelsize=16)
        ax3.grid()
        
       
        ax4.plot(t_vec, tau_net[:,nodes].detach().cpu(),color,linewidth=2.5)
        ax4.plot(t_vec, tau_gt[:,nodes].detach().cpu(),'k--',linewidth=2.5)
        l1, = ax4.plot([],[],'k--',linewidth=2.5)
        l2, = ax4.plot([],[],color,linewidth=2.5)
        ax4.legend((l1, l2), ('GT',ROM_model), fontsize='18',loc='upper right')
        ax4.set_ylabel('$\\tau$ ', fontsize='24')
        ax4.set_xlabel('$t$', fontsize='24')
        ax4.set_xlim(0.9,1)
        ax4.tick_params(axis='y', labelsize=16)
        ax4.tick_params(axis='x', labelsize=16)
        ax4.set_yticks(np.arange(-0.62,-0.58,0.01))
        ax4.grid()
        

        
    elif (sys_name == 'GC'):
        
        if ROM_model == 'tLaSDI':
            save_dir_tmp = 'tLaSDI_GC_sols'
        elif ROM_model == 'TA-ROM':
            save_dir_tmp = 'TA_ROM_GC_sols'
        elif ROM_model == 'Vanilla-FNN': 
            save_dir_tmp = 'Vanilla_FNN_GC_sols'

        # Get Variables
        q_net, p_net, s1_net, s2_net = get_variables(z_net, sys_name)
        q_gt, p_gt, s1_gt, s2_gt = get_variables(z_gt, sys_name)
        nodes = [15-1, 30-1, 60-1, 85-1]#good
    
     
        fig, axes = plt.subplots(1,4, figsize=(20, 5))
        ax1, ax2, ax3, ax4 = axes.flatten()
        plot_name = '[GC] ' + name
        
        
        plt.subplots_adjust(wspace=.35)

            
        #fig.suptitle(plot_name)

      
        ax1.plot(t_vec, q_net[:,nodes].detach().cpu(),color,linewidth=2.5)
        ax1.plot(t_vec, q_gt[:,nodes].detach().cpu(),'k--',linewidth=2.5)
        l1, = ax1.plot([],[],'k--',linewidth=2.5)
        l2, = ax1.plot([],[],color,linewidth=2.5)
        ax1.legend((l1, l2), ('GT',ROM_model), fontsize='16',loc='upper left')
        ax1.set_ylabel('$q$ ', fontsize='24',rotation=90)
#         ax1.yaxis.set_label_coords(-0.15, 0.5)
        ax1.tick_params(axis='y', labelsize=16)
        ax1.tick_params(axis='x', labelsize=16)
        ax1.set_xlabel('$t$', fontsize='24')
        ax1.set_xlim(7.84,8)
        ax1.grid()
  
        ax2.plot(t_vec, p_net[:,nodes].detach().cpu(),color,linewidth=2.5)
        ax2.plot(t_vec, p_gt[:,nodes].detach().cpu(),'k--',linewidth=2.5)
        l1, = ax2.plot([],[],'k--',linewidth=2.5)
        l2, = ax2.plot([],[],color,linewidth=2.5)
        #ax2.legend((l1, l2), ('GT',ROM_model), fontsize='14',loc=(0.05,0.6))
        ax2.set_ylabel('$p$ ', fontsize='24',rotation=90)
        ax2.set_xlabel('$t$', fontsize='24')
#         ax2.yaxis.set_label_coords(-0.15, 0.5)
        ax2.tick_params(axis='y', labelsize=16)
        ax2.tick_params(axis='x', labelsize=16)
        ax2.set_xlim(7.84,8)
        ax2.grid()
        
        ax3.plot(t_vec, s1_net[:,nodes].detach().cpu(),color,linewidth=2.5)
        ax3.plot(t_vec, s1_gt[:,nodes].detach().cpu(),'k--',linewidth=2.5)
        l1, = ax3.plot([],[],'k--',linewidth=2.5)
        l2, = ax3.plot([],[],color,linewidth=2.5)
        #ax3.legend((l1, l2), ('GT',ROM_model), fontsize='14')
        ax3.set_ylabel('$S_1$ ', fontsize='24',rotation=90)
        ax3.set_xlabel('$t$', fontsize='24')
#         ax3.yaxis.set_label_coords(-0.15, 0.5)
        ax3.tick_params(axis='y', labelsize=16)
        ax3.tick_params(axis='x', labelsize=16)
        ax3.set_xlim(7.84,8)
        ax3.grid()
       
        ax4.plot(t_vec, s2_net[:,nodes].detach().cpu(),color,linewidth=2.5)
        ax4.plot(t_vec, s2_gt[:,nodes].detach().cpu(),'k--',linewidth=2.5)
        l1, = ax4.plot([],[],'k--',linewidth=2.5)
        l2, = ax4.plot([],[],color,linewidth=2.5)
        #ax4.legend((l1, l2), ('GT',ROM_model), fontsize='14')
        ax4.set_ylabel('$S_2$', fontsize='24',rotation=90)
        ax4.set_xlabel('$t$', fontsize='24')
#         ax4.yaxis.set_label_coords(-0.15, 0.5)
        ax4.tick_params(axis='y', labelsize=16)
        ax4.tick_params(axis='x', labelsize=16)
        ax4.set_xlim(7.84,8)
        ax4.grid()
        
        
#     plot_name = 'Vanilla-FNN_smallest_loss_GC_scaled'

    save_dir = os.path.join(output_dir, plot_name)
#     save_dir = os.path.join(output_dir, save_dir_tmp)
    plt.tight_layout()
    
    plt.savefig(save_dir)
    plt.clf()
    
    
def plot_test_results_noise(z_net, z_gt, zGT, dt, name, output_dir, N ,n_pred, sys_name, ROM_model):
    plt.clf()
    
    
    tstart = (N+2)*dt - n_pred*dt 
    
    t_vec = np.linspace(tstart,(N+1)*dt,n_pred)
    
    if ROM_model == 'tLaSDI':
        color = 'b'
    elif ROM_model == 'TA-ROM':
        color = 'r'
    elif ROM_model == 'Vanilla-FNN': 
        color = 'g'
        
    
    
       # N= test_final, n_pred = self.dim_t_tt

    if (sys_name == 'viscoelastic'):
        
        if ROM_model == 'tLaSDI':
            save_dir_tmp = 'tLaSDI_VC_sols'
        elif ROM_model == 'TA-ROM':
            save_dir_tmp = 'TA_ROM_VC_sols'
        elif ROM_model == 'Vanilla-FNN': 
            save_dir_tmp = 'Vanilla_FNN_VC_sols'
        
        

        
        tstart = (N+3)*dt - n_pred*dt 
    
        t_vec = np.linspace(tstart,(N+2)*dt,n_pred)

        # Get Variables
        q_net, v_net, e_net, tau_net = get_variables(z_net, sys_name)
        q_gt, v_gt, e_gt, tau_gt = get_variables(z_gt, sys_name)
        q_GT, v_GT, e_GT, tau_GT = get_variables(zGT, sys_name)
        
        nodes = [20-1, 40-1, 60-1, 80-1]
        #nodes = [i for i in range(10, 20)]
     
        fig, axes = plt.subplots(1,4, figsize=(20, 5))
        ax1, ax2, ax3, ax4 = axes.flatten()
        
        plt.subplots_adjust(wspace=.35)
        
        plot_name = '[VC] ' + name

        #fig.suptitle(plot_name)
        

        matplotlib.rcParams['text.usetex'] = True
        plt.rc('text', usetex=True)

        ax1.plot(t_vec, q_net[:,nodes].detach().cpu(),color,linewidth=2.5)
        ax1.plot(t_vec, q_gt[:,nodes].detach().cpu(),'k--',linewidth=2.5)
        ax1.plot(t_vec, q_GT[:,nodes].detach().cpu(),'.',linewidth=2.5)
        l1, = ax1.plot([],[],'k--',linewidth=2.5)
        l2, = ax1.plot([],[],color,linewidth=2.5)
#         ax1.legend((l1, l2), ('GT',ROM_model), fontsize='16')
        ax1.set_ylabel('$q$ ', fontsize='24')
        ax1.tick_params(axis='y', labelsize=16)
        ax1.tick_params(axis='x', labelsize=16)
        ax1.set_xlabel('$t$', fontsize='24')
        ax1.grid()
  
        ax2.plot(t_vec, v_net[:,nodes].detach().cpu(),color,linewidth=2.5)
        ax2.plot(t_vec, v_gt[:,nodes].detach().cpu(),'k--',linewidth=2.5)
        ax2.plot(t_vec, v_GT[:,nodes].detach().cpu(),'.',linewidth=2.5)
        l1, = ax2.plot([],[],'k--',linewidth=2.5)
        l2, = ax2.plot([],[],color,linewidth=2.5)
        #ax2.legend((l1, l2), ('GT',ROM_model), fontsize='16')
        ax2.set_ylabel('$\mathit{v}$', fontsize='24')
        ax2.set_xlabel('$t$', fontsize='24')
        ax2.tick_params(axis='y', labelsize=16)
        ax2.tick_params(axis='x', labelsize=16)
        ax2.grid()
        
        ax3.plot(t_vec, e_net[:,nodes].detach().cpu(),color,linewidth=2.5)
        ax3.plot(t_vec, e_gt[:,nodes].detach().cpu(),'k--',linewidth=2.5)
        ax3.plot(t_vec, e_GT[:,nodes].detach().cpu(),'.',linewidth=2.5)
        l1, = ax3.plot([],[],'k--',linewidth=2.5)
        l2, = ax3.plot([],[],color,linewidth=2.5)
        #ax3.legend((l1, l2), ('GT',ROM_model), fontsize='16')
        ax3.set_ylabel('$e$ ', fontsize='24')
        ax3.set_xlabel('$t$', fontsize='24')
        ax3.tick_params(axis='y', labelsize=16)
        ax3.tick_params(axis='x', labelsize=16)
        ax3.grid()
        
       
        ax4.plot(t_vec, tau_net[:,nodes].detach().cpu(),color,linewidth=2.5)
        ax4.plot(t_vec, tau_gt[:,nodes].detach().cpu(),'k--',linewidth=2.5)
        ax4.plot(t_vec, tau_GT[:,nodes].detach().cpu(),'.',linewidth=2.5)
        l1, = ax4.plot([],[],'k--',linewidth=2.5)
        l2, = ax4.plot([],[],color,linewidth=2.5)
        l3, = ax4.plot([],[],'.',linewidth=2.5)
        ax4.legend((l1, l2, l3), ('GT',ROM_model,'data'), fontsize='18')
        ax4.set_ylabel('$\\tau$ ', fontsize='24')
        ax4.set_xlabel('$t$', fontsize='24')
        ax4.tick_params(axis='y', labelsize=16)
        ax4.tick_params(axis='x', labelsize=16)
        ax4.grid()
        

        
    elif (sys_name == 'GC'):
        
        if ROM_model == 'tLaSDI':
            save_dir_tmp = 'tLaSDI_GC_sols'
        elif ROM_model == 'TA-ROM':
            save_dir_tmp = 'TA_ROM_GC_sols'
        elif ROM_model == 'Vanilla-FNN': 
            save_dir_tmp = 'Vanilla_FNN_GC_sols'

        # Get Variables
        q_net, p_net, s1_net, s2_net = get_variables(z_net, sys_name)
        q_gt, p_gt, s1_gt, s2_gt = get_variables(z_gt, sys_name)
        q_GT, p_GT, s1_GT, s2_GT = get_variables(zGT, sys_name)
        nodes = [15-1, 30-1, 60-1, 85-1]#good
    
     
        fig, axes = plt.subplots(1,4, figsize=(20, 5))
        ax1, ax2, ax3, ax4 = axes.flatten()
        plot_name = '[GC] ' + name
        
        
        plt.subplots_adjust(wspace=.35)

            
        #fig.suptitle(plot_name)

      
        ax1.plot(t_vec, q_net[:,nodes].detach().cpu(),color,linewidth=2.5)
        ax1.plot(t_vec, q_gt[:,nodes].detach().cpu(),'k--',linewidth=2.5)
        ax1.plot(t_vec, q_GT[:,nodes].detach().cpu(),'.',linewidth=2.5)
        l1, = ax1.plot([],[],'k--',linewidth=2.5)
        l2, = ax1.plot([],[],color,linewidth=2.5)
        l3, = ax1.plot([],[],'.',linewidth=2.5)
        ax1.legend((l1, l2, l3), ('GT',ROM_model,'data'), fontsize='16',loc='upper left')
        ax1.set_ylabel('$q$ ', fontsize='24',rotation=90)
#         ax1.yaxis.set_label_coords(-0.15, 0.5)
        ax1.tick_params(axis='y', labelsize=16)
        ax1.tick_params(axis='x', labelsize=16)
        ax1.set_xlabel('$t$', fontsize='24')
        ax1.grid()
  
        ax2.plot(t_vec, p_net[:,nodes].detach().cpu(),color,linewidth=2.5)
        ax2.plot(t_vec, p_gt[:,nodes].detach().cpu(),'k--',linewidth=2.5)
        ax2.plot(t_vec, p_GT[:,nodes].detach().cpu(),'.',linewidth=2.5)
        l1, = ax2.plot([],[],'k--',linewidth=2.5)
        l2, = ax2.plot([],[],color,linewidth=2.5)
        #ax2.legend((l1, l2), ('GT',ROM_model), fontsize='14',loc=(0.05,0.6))
        ax2.set_ylabel('$p$ ', fontsize='24',rotation=90)
        ax2.set_xlabel('$t$', fontsize='24')
#         ax2.yaxis.set_label_coords(-0.15, 0.5)
        ax2.tick_params(axis='y', labelsize=16)
        ax2.tick_params(axis='x', labelsize=16)
        ax2.grid()
        
        ax3.plot(t_vec, s1_net[:,nodes].detach().cpu(),color,linewidth=2.5)
        ax3.plot(t_vec, s1_gt[:,nodes].detach().cpu(),'k--',linewidth=2.5)
        ax3.plot(t_vec, s1_GT[:,nodes].detach().cpu(),'.',linewidth=2.5)
        l1, = ax3.plot([],[],'k--',linewidth=2.5)
        l2, = ax3.plot([],[],color,linewidth=2.5)
        #ax3.legend((l1, l2), ('GT',ROM_model), fontsize='14')
        ax3.set_ylabel('$S_1$ ', fontsize='24',rotation=90)
        ax3.set_xlabel('$t$', fontsize='24')
#         ax3.yaxis.set_label_coords(-0.15, 0.5)
        ax3.tick_params(axis='y', labelsize=16)
        ax3.tick_params(axis='x', labelsize=16)
        ax3.grid()
       
        ax4.plot(t_vec, s2_net[:,nodes].detach().cpu(),color,linewidth=2.5)
        ax4.plot(t_vec, s2_gt[:,nodes].detach().cpu(),'k--',linewidth=2.5)
        ax4.plot(t_vec, se_GT[:,nodes].detach().cpu(),'.',linewidth=2.5)
        l1, = ax4.plot([],[],'k--',linewidth=2.5)
        l2, = ax4.plot([],[],color,linewidth=2.5)
        #ax4.legend((l1, l2), ('GT',ROM_model), fontsize='14')
        ax4.set_ylabel('$S_2$', fontsize='24',rotation=90)
        ax4.set_xlabel('$t$', fontsize='24')
#         ax4.yaxis.set_label_coords(-0.15, 0.5)
        ax4.tick_params(axis='y', labelsize=16)
        ax4.tick_params(axis='x', labelsize=16)
        ax4.grid()

    save_dir = os.path.join(output_dir, plot_name)
#     save_dir = os.path.join(output_dir, save_dir_tmp)
    plt.tight_layout()
    
    plt.savefig(save_dir)
    plt.clf()

    
def plot_test_results_all(z_net1,z_net2,z_net3, z_gt, dt, output_dir, N ,n_pred, sys_name):
    plt.clf()
    
    
    tstart = (N+2)*dt - n_pred*dt 
    
    t_vec = np.linspace(tstart,(N+1)*dt,n_pred)
    
        
    
    if (sys_name == 'viscoelastic'):
        

        save_dir_tmp = 'VC_sols_all'
        
        color1 = 'b'
        color2 = 'r'
        color3 = 'g'
        
        tstart = (N+3)*dt - n_pred*dt 
    
        t_vec = np.linspace(tstart,(N+2)*dt,n_pred)

        # Get Variables
        q_net1, v_net1, e_net1, tau_net1 = get_variables(z_net1, sys_name)
        q_net2, v_net2, e_net2, tau_net2 = get_variables(z_net2, sys_name)
        q_net3, v_net3, e_net3, tau_net3 = get_variables(z_net3, sys_name)
        q_gt, v_gt, e_gt, tau_gt = get_variables(z_gt, sys_name)
        
        nodes = [20-1, 40-1, 60-1, 80-1]
#         nodes = [20-1, 60-1]
        #nodes = [i for i in range(10, 20)]
     
        fig, axes = plt.subplots(1,4, figsize=(20, 5))
        ax1, ax2, ax3, ax4 = axes.flatten()
        
        plt.subplots_adjust(wspace=.35)
        
        matplotlib.rcParams['text.usetex'] = True
        plt.rc('text', usetex=True)

        ax1.plot(t_vec, q_net1[:,nodes].detach().cpu(),color1,linewidth=2,markerfacecolor='none')
        ax1.plot(t_vec, q_net2[:,nodes].detach().cpu(),color2,linewidth=2,markerfacecolor='none')
        ax1.plot(t_vec, q_net3[:,nodes].detach().cpu(),color3,linewidth=2,markerfacecolor='none')
        ax1.plot(t_vec, q_gt[:,nodes].detach().cpu(),'k--',linewidth=2)
        l1, = ax1.plot([],[],'k--',linewidth=2)
        l2, = ax1.plot([],[],color1,linewidth=2,markerfacecolor='none')
        l3, = ax1.plot([],[],color2,linewidth=2,markerfacecolor='none')
        l4, = ax1.plot([],[],color3,linewidth=2,markerfacecolor='none')
#         ax1.legend((l1, l2, l3, l4), ('GT','tLaSDI','TA-ROM','Vanilla-FNN'), fontsize='16',loc=(0,0.2))
        ax1.set_ylabel('$q$ ', fontsize='24')
        ax1.tick_params(axis='y', labelsize=14)
        ax1.tick_params(axis='x', labelsize=14)
        ax1.set_xlabel('$t$', fontsize='24')
        ax1.grid()
  
        ax2.plot(t_vec, v_net1[:,nodes].detach().cpu(),color1,linewidth=2,markerfacecolor='none')
        ax2.plot(t_vec, v_net2[:,nodes].detach().cpu(),color2,linewidth=2,markerfacecolor='none')
        ax2.plot(t_vec, v_net3[:,nodes].detach().cpu(),color3,linewidth=2,markerfacecolor='none')
        ax2.plot(t_vec, v_gt[:,nodes].detach().cpu(),'k--',linewidth=2)
        l1, = ax1.plot([],[],'k--',linewidth=2)
        l2, = ax1.plot([],[],color1,linewidth=2,markerfacecolor='none')
        l3, = ax1.plot([],[],color2,linewidth=2,markerfacecolor='none')
        l4, = ax1.plot([],[],color3,linewidth=2,markerfacecolor='none')
#         ax2.legend((l1, l2, l3, l4), ('GT','tLaSDI','TA-ROM','Vanilla-FNN'), fontsize='16',loc=(0,0.2))
        ax2.set_ylabel('$\mathit{v}$', fontsize='24')
        ax2.set_xlabel('$t$', fontsize='24')
        ax2.tick_params(axis='y', labelsize=14)
        ax2.tick_params(axis='x', labelsize=14)
        ax2.grid()
        
        ax3.plot(t_vec, e_net1[:,nodes].detach().cpu(),color1,linewidth=2,markerfacecolor='none')
        ax3.plot(t_vec, e_net2[:,nodes].detach().cpu(),color2,linewidth=2,markerfacecolor='none')
        ax3.plot(t_vec, e_net3[:,nodes].detach().cpu(),color3,linewidth=2,markerfacecolor='none')
        ax3.plot(t_vec, e_gt[:,nodes].detach().cpu(),'k--',linewidth=2)
        l1, = ax1.plot([],[],'k--',linewidth=2)
        l2, = ax1.plot([],[],color1,linewidth=2,markerfacecolor='none')
        l3, = ax1.plot([],[],color2,linewidth=2,markerfacecolor='none')
        l4, = ax1.plot([],[],color3,linewidth=2,markerfacecolor='none')
#         ax3.legend((l1, l2, l3, l4), ('GT','tLaSDI','TA-ROM','Vanilla-FNN'), fontsize='16',loc=(0,0.2))
        ax3.set_ylabel('$e$ ', fontsize='24')
        ax3.set_xlabel('$t$', fontsize='24')
        ax3.tick_params(axis='y', labelsize=14)
        ax3.tick_params(axis='x', labelsize=14)
        ax3.grid()
        
       
        ax4.plot(t_vec, tau_net1[:,nodes].detach().cpu(),color1,linewidth=2,markerfacecolor='none')
        ax4.plot(t_vec, tau_net2[:,nodes].detach().cpu(),color2,linewidth=2,markerfacecolor='none')
        ax4.plot(t_vec, tau_net3[:,nodes].detach().cpu(),color3,linewidth=2,markerfacecolor='none')
        ax4.plot(t_vec, tau_gt[:,nodes].detach().cpu(),'k--',linewidth=2)
        l1, = ax1.plot([],[],'k--',linewidth=2)
        l2, = ax1.plot([],[],color1,linewidth=2,markerfacecolor='none')
        l3, = ax1.plot([],[],color2,linewidth=2,markerfacecolor='none')
        l4, = ax1.plot([],[],color3,linewidth=2,markerfacecolor='none')
        ax4.legend((l1, l2, l3, l4), ('GT','tLaSDI','TA-ROM','Vanilla-FNN'), fontsize='16',loc=(0,0.05))
        ax4.set_ylabel('$\\tau$ ', fontsize='24')
        ax4.set_xlabel('$t$', fontsize='24')
        ax4.tick_params(axis='y', labelsize=14)
        ax4.tick_params(axis='x', labelsize=14)
        ax4.grid()
        
    elif (sys_name == 'GC'):
        
        save_dir_tmp = 'GC_sols_all'
        
        color1 = 'b'
        color2 = 'r'
        color3 = 'g'
        
        
        if ROM_model == 'tLaSDI':
            save_dir_tmp = 'tLaSDI_GC_sols'
        elif ROM_model == 'TA-ROM':
            save_dir_tmp = 'TA_ROM_GC_sols'
        elif ROM_model == 'Vanilla-FNN': 
            save_dir_tmp = 'Vanilla_FNN_GC_sols'

        # Get Variables
        q_net1, p_net1, s1_net1, s2_net1 = get_variables(z_net1, sys_name)
        q_net2, p_net2, s1_net2, s2_net2 = get_variables(z_net2, sys_name)
        q_net3, p_net3, s1_net3, s2_net3 = get_variables(z_net3, sys_name)
        q_gt, p_gt, s1_gt, s2_gt = get_variables(z_gt, sys_name)
        nodes = [15-1, 30-1, 60-1, 85-1]#good
    
     
        fig, axes = plt.subplots(1,4, figsize=(20, 5))
        ax1, ax2, ax3, ax4 = axes.flatten()
        plot_name = '[GC] ' + name
        
        
        plt.subplots_adjust(wspace=.35)
        
        
        matplotlib.rcParams['text.usetex'] = True
        plt.rc('text', usetex=True)

        ax1.plot(t_vec, q_net1[:,nodes].detach().cpu(),color1,linewidth=2,markerfacecolor='none')
        ax1.plot(t_vec, q_net2[:,nodes].detach().cpu(),color2,linewidth=2,markerfacecolor='none')
        ax1.plot(t_vec, q_net3[:,nodes].detach().cpu(),color3,linewidth=2,markerfacecolor='none')
        ax1.plot(t_vec, q_gt[:,nodes].detach().cpu(),'k--',linewidth=2)
        l1, = ax1.plot([],[],'k--',linewidth=2)
        l2, = ax1.plot([],[],color1,linewidth=2,markerfacecolor='none')
        l3, = ax1.plot([],[],color2,linewidth=2,markerfacecolor='none')
        l4, = ax1.plot([],[],color3,linewidth=2,markerfacecolor='none')
        ax1.legend((l1, l2, l3, l4), ('GT','tLaSDI','TA-ROM','Vanilla-FNN'), fontsize='16')
        ax1.set_ylabel('$q$ ', fontsize='24')
        ax1.tick_params(axis='y', labelsize=14)
        ax1.tick_params(axis='x', labelsize=14)
        ax1.set_xlabel('$t$', fontsize='24')
        ax1.grid()
  
        ax2.plot(t_vec, p_net1[:,nodes].detach().cpu(),color1,linewidth=2,markerfacecolor='none')
        ax2.plot(t_vec, p_net2[:,nodes].detach().cpu(),color2,linewidth=2,markerfacecolor='none')
        ax2.plot(t_vec, p_net3[:,nodes].detach().cpu(),color3,linewidth=2,markerfacecolor='none')
        ax2.plot(t_vec, p_gt[:,nodes].detach().cpu(),'k--',linewidth=2)
        l1, = ax1.plot([],[],'k--',linewidth=2)
        l2, = ax1.plot([],[],color1,linewidth=2,markerfacecolor='none')
        l3, = ax1.plot([],[],color2,linewidth=2,markerfacecolor='none')
        l4, = ax1.plot([],[],color3,linewidth=2,markerfacecolor='none')
        ax2.legend((l1, l2, l3, l4), ('GT','tLaSDI','TA-ROM','Vanilla-FNN'), fontsize='16')
        ax2.set_ylabel('$p$', fontsize='24')
        ax2.set_xlabel('$t$', fontsize='24')
        ax2.tick_params(axis='y', labelsize=14)
        ax2.tick_params(axis='x', labelsize=14)
        ax2.grid()
        
        ax3.plot(t_vec, s1_net1[:,nodes].detach().cpu(),color1,linewidth=2,markerfacecolor='none')
        ax3.plot(t_vec, s1_net2[:,nodes].detach().cpu(),color2,linewidth=2,markerfacecolor='none')
        ax3.plot(t_vec, s1_net3[:,nodes].detach().cpu(),color3,linewidth=2,markerfacecolor='none')
        ax3.plot(t_vec, s1_gt[:,nodes].detach().cpu(),'k--',linewidth=2)
        l1, = ax1.plot([],[],'k--',linewidth=2)
        l2, = ax1.plot([],[],color1,linewidth=2,markerfacecolor='none')
        l3, = ax1.plot([],[],color2,linewidth=2,markerfacecolor='none')
        l4, = ax1.plot([],[],color3,linewidth=2,markerfacecolor='none')
        ax3.legend((l1, l2, l3, l4), ('GT','tLaSDI','TA-ROM','Vanilla-FNN'), fontsize='16')
        ax3.set_ylabel('$S_1$ ', fontsize='24')
        ax3.set_xlabel('$t$', fontsize='24')
        ax3.tick_params(axis='y', labelsize=14)
        ax3.tick_params(axis='x', labelsize=14)
        ax3.grid()
        
       
        ax4.plot(t_vec, s2_net1[:,nodes].detach().cpu(),color1,linewidth=2,markerfacecolor='none')
        ax4.plot(t_vec, s2_net2[:,nodes].detach().cpu(),color2,linewidth=2,markerfacecolor='none')
        ax4.plot(t_vec, s2_net3[:,nodes].detach().cpu(),color3,linewidth=2,markerfacecolor='none')
        ax4.plot(t_vec, s2_gt[:,nodes].detach().cpu(),'k--',linewidth=2)
        l1, = ax1.plot([],[],'k--',linewidth=2)
        l2, = ax1.plot([],[],color1,linewidth=2,markerfacecolor='none')
        l3, = ax1.plot([],[],color2,linewidth=2,markerfacecolor='none')
        l4, = ax1.plot([],[],color3,linewidth=2,markerfacecolor='none')
        ax4.legend((l1, l2, l3, l4), ('GT','tLaSDI','TA-ROM','Vanilla-FNN'), fontsize='16')
        ax4.set_ylabel('$S_2$ ', fontsize='24')
        ax4.set_xlabel('$t$', fontsize='24')
        ax4.tick_params(axis='y', labelsize=14)
        ax4.tick_params(axis='x', labelsize=14)
        ax4.grid()
        
    save_dir = os.path.join(output_dir, save_dir_tmp)
    plt.tight_layout()
    
    plt.savefig(save_dir)
    plt.clf() 
    

    
def plot_full_integration(z_net, z_gt, dt, name, output_dir, sys_name):
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
        
        plt.subplots_adjust(wspace=.35)
        
        matplotlib.rcParams['text.usetex'] = True
        plt.rc('text', usetex=True)
        
        plot_name = '[VC] ' + name

      
        ax1.plot(t_vec, q_net[:,nodes].detach().cpu(),'b',linewidth=2.5)
        ax1.plot(t_vec, q_gt[:,nodes].detach().cpu(),'k--',linewidth=2.5)
        l1, = ax1.plot([],[],'k--',linewidth=2.5)
        l2, = ax1.plot([],[],'b',linewidth=2.5)
        ax1.legend((l1, l2), ('GT','tLaSDI'), fontsize='16')
        ax1.set_ylabel('$q$', fontsize='24')
        ax1.set_xlabel('$t$', fontsize='24')
        ax1.tick_params(axis='y', labelsize=14)
        ax1.tick_params(axis='x', labelsize=14)
        ax1.grid()
  
        ax2.plot(t_vec, v_net[:,nodes].detach().cpu(),'b',linewidth=2.5)
        ax2.plot(t_vec, v_gt[:,nodes].detach().cpu(),'k--',linewidth=2.5)
        l1, = ax2.plot([],[],'k--',linewidth=2.5)
        l2, = ax2.plot([],[],'b',linewidth=2.5)
        ax2.legend((l1, l2), ('GT','tLaSDI'), fontsize='16')
        ax2.set_ylabel('$v$', fontsize='24')
        ax2.set_xlabel('$t$', fontsize='24')
        ax2.tick_params(axis='y', labelsize=14)
        ax2.tick_params(axis='x', labelsize=14)
        ax2.grid()
        
        ax3.plot(t_vec, e_net[:,nodes].detach().cpu(),'b',linewidth=2.5)
        ax3.plot(t_vec, e_gt[:,nodes].detach().cpu(),'k--',linewidth=2.5)
        l1, = ax3.plot([],[],'k--',linewidth=2.5)
        l2, = ax3.plot([],[],'b',linewidth=2.5)
        ax3.legend((l1, l2), ('GT','tLaSDI'), fontsize='16')
        ax3.set_ylabel('$e$', fontsize='24')
        ax3.set_xlabel('$t$', fontsize='24')
        ax3.tick_params(axis='y', labelsize=14)
        ax3.tick_params(axis='x', labelsize=14)
        ax3.grid()
       
        ax4.plot(t_vec, tau_net[:,nodes].detach().cpu(),'b',linewidth=2.5)
        ax4.plot(t_vec, tau_gt[:,nodes].detach().cpu(),'k--',linewidth=2.5)
        l1, = ax4.plot([],[],'k--',linewidth=2.5)
        l2, = ax4.plot([],[],'b',linewidth=2.5)
        ax4.legend((l1, l2), ('GT','tLaSDI'), fontsize='16')
        ax4.set_ylabel('$\\tau$', fontsize='24')
        ax4.set_xlabel('$t$', fontsize='24')
        ax4.tick_params(axis='y', labelsize=14)
        ax4.tick_params(axis='x', labelsize=14)
        ax4.grid()


        
    elif (sys_name == 'GC'):

        # Get Variables
        q_net, p_net, s1_net, s2_net = get_variables(z_net, sys_name)
        q_gt, p_gt, s1_gt, s2_gt = get_variables(z_gt, sys_name)
        #nodes = [20-1, 40-1, 60-1, 80-1]
        nodes = [15-1, 30-1, 60-1, 85-1]
     
        fig, axes = plt.subplots(1,4, figsize=(20, 5))
        ax1, ax2, ax3, ax4 = axes.flatten()
        
        plt.subplots_adjust(wspace=.35)
        
        matplotlib.rcParams['text.usetex'] = True
        plt.rc('text', usetex=True)
        plot_name = '[GC] ' + name

      
        ax1.plot(t_vec, q_net[:,nodes].detach().cpu(),'b',linewidth=2.5)
        ax1.plot(t_vec, q_gt[:,nodes].detach().cpu(),'k--',linewidth=2.5)
        l1, = ax1.plot([],[],'k--',linewidth=2.5)
        l2, = ax1.plot([],[],'b',linewidth=2.5)
        ax1.legend((l1, l2), ('GT','tLaSDI'), fontsize='16')
        ax1.set_ylabel('$q$', fontsize='24')
        ax1.set_xlabel('$t$', fontsize='24')
        ax1.tick_params(axis='y', labelsize=14)
        ax1.tick_params(axis='x', labelsize=14)
        ax1.grid()
  
        ax2.plot(t_vec, p_net[:,nodes].detach().cpu(),'b',linewidth=2.5)
        ax2.plot(t_vec, p_gt[:,nodes].detach().cpu(),'k--',linewidth=2.5)
        l1, = ax2.plot([],[],'k--',linewidth=2)
        l2, = ax2.plot([],[],'b',linewidth=2)
        ax2.legend((l1, l2), ('GT','tLaSDI'), fontsize='16')
        ax2.set_ylabel('$p$', fontsize='24')
        ax2.set_xlabel('$t$', fontsize='24')
        ax2.tick_params(axis='y', labelsize=14)
        ax2.tick_params(axis='x', labelsize=14)
        ax2.grid()
        
        ax3.plot(t_vec, s1_net[:,nodes].detach().cpu(),'b',linewidth=2.5)
        ax3.plot(t_vec, s1_gt[:,nodes].detach().cpu(),'k--',linewidth=2.5)
        l1, = ax3.plot([],[],'k--',linewidth=2.5)
        l2, = ax3.plot([],[],'b',linewidth=2.5)
        ax3.legend((l1, l2), ('GT','tLaSDI'), fontsize='16')
        ax3.set_ylabel('$S_1$', fontsize='24')
        ax3.set_xlabel('$t$', fontsize='24')
        ax3.tick_params(axis='y', labelsize=14)
        ax3.tick_params(axis='x', labelsize=14)
        ax3.grid()
       
        ax4.plot(t_vec, s2_net[:,nodes].detach().cpu(),'b',linewidth=2.5)
        ax4.plot(t_vec, s2_gt[:,nodes].detach().cpu(),'k--',linewidth=2.5)
        l1, = ax4.plot([],[],'k--',linewidth=2.5)
        l2, = ax4.plot([],[],'b',linewidth=2.5)
        ax4.legend((l1, l2), ('GT','tLaSDI'), fontsize='16')
        ax4.set_ylabel('$S_2$', fontsize='24')
        ax4.set_xlabel('$t$', fontsize='24')
        ax4.tick_params(axis='y', labelsize=14)
        ax4.tick_params(axis='x', labelsize=14)
        ax4.grid()

        
   

  
    save_dir = os.path.join(output_dir, plot_name)
    plt.tight_layout()
    
    plt.savefig(save_dir)
    plt.clf()
    
    

