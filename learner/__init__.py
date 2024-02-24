"""
@author: jpzxshi
"""
from . import nn
from . import integrator


from .brain_tLaSDI_GAEhyper import Brain_tLaSDI_GAEhyper

from .brain_tLaSDI import Brain_tLaSDI
from .brain_tLaSDI_rel_loss import Brain_tLaSDI_rel_loss
from .brain_tLaSDI_weighted_loss import Brain_tLaSDI_weighted_loss
from .brain_FNN import Brain_FNN
from .brain_tLaSDI_SAE_sep import Brain_tLaSDI_SAE_sep

from .data import Data
from .nn import Module

__all__ = [
    'nn',
    'integrator',
    'Brain_tLaSDI',
    'Data',
    'Module',
]
