"""
@author: jpzxshi
"""
from . import nn
from . import integrator
from .brain import Brain
from .GFINNs import GFINNs_solver

from .brain_tLaSDI_greedy import Brain_tLaSDI_greedy
from .brain_tLaSDI import Brain_tLaSDI

# from .brain_test_parain_sim import Brain_test_parain_sim


from .data import Data
from .nn import Module

__all__ = [
    'nn',
    'integrator',
    'GFINNs_solver',
    'Brain_hyper_sim_jac_greedy',
    'Brain_tLaSDI_greedy',
    'Brain_tLaSDI',
    'Data',
    'Module',
]
