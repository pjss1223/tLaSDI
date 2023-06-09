"""
@author: jpzxshi
"""
from . import nn
from . import integrator

from .brain_tLaSDI_greedy import Brain_tLaSDI_greedy
from .brain_tLaSDI_GAEhyper import Brain_tLaSDI_GAEhyper
from .brain_tLaSDI import Brain_tLaSDI
from .brain_tLaSDI_sep import Brain_tLaSDI_sep
from .brain_FNN import Brain_FNN


from .data import Data
from .nn import Module

__all__ = [
    'nn',
    'integrator',
    'GFINNs_solver',
    'Brain_hyper_sim_jac_greedy',
    'Brain_hyper_sim_jac_GAEhyper',
    'Brain_tLaSDI_greedy',
    'Brain_tLaSDI',
    'Brain_FNN',
    'Data',
    'Module',
]
