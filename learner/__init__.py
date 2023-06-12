"""
@author: jpzxshi
"""
from . import nn
from . import integrator
from .brain import Brain
from .GFINNs import GFINNs_solver
from .brain_test import Brain_test
from .brain_test_sim import Brain_test_sim
from .brain_hyper_sim import Brain_hyper_sim
from .brain_hyper_sim_gradx import Brain_hyper_sim_gradx
from .brain_hyper_sim_gradx_jac import Brain_hyper_sim_gradx_jac
from .brain_hyper_sim_grad import Brain_hyper_sim_grad
from .brain_hyper_sim_greedy import Brain_hyper_sim_greedy
from .brain_hyper_sim_jac_greedy import Brain_hyper_sim_jac_greedy
from .brain_hyper_sim_gradx_greedy import Brain_hyper_sim_gradx_greedy
from .brain_hyper_sim_grad_greedy import Brain_hyper_sim_grad_greedy
from .brain_hyper_sim_gradx_jac_greedy import Brain_hyper_sim_gradx_jac_greedy
from .brain_hyper_sim_grad_jac_greedy import Brain_hyper_sim_grad_jac_greedy
from .brain_tLaSDI_greedy import Brain_tLaSDI_greedy
from .brain_tLaSDI import Brain_tLaSDI

# from .brain_test_parain_sim import Brain_test_parain_sim

from .brain_test_jac_sim import Brain_test_jac_sim
from .brain_test_jac_sim_grad import Brain_test_jac_sim_grad
from .brain_test_sim_grad import Brain_test_sim_grad
from .brain_test_sim_gradx import Brain_test_sim_gradx
from .brain_test_sim_grad_norecon import Brain_test_sim_grad_norecon
from .brain_test_sim_grad_noint import Brain_test_sim_grad_noint


from .data import Data
from .nn import Module

__all__ = [
    'nn',
    'integrator',
    'Brain',
    'GFINNs_solver',
    'Brain_test',
    'Brain_hyper_sim',
    'Brain_hyper_sim_grad',
    'Brain_hyper_sim_gradx',
    'Brain_hyper_sim_gradx_jac',
    'Brain_hyper_sim_greedy',
    'Brain_hyper_sim_jac_greedy',
    'Brain_tLaSDI_greedy',
    'Brain_tLaSDI',
    'Data',
    'Module',
]
