"""
@author: jpzxshi
"""
from . import nn
from . import integrator


from .brain_tLaSDI_param import Brain_tLaSDI_param
from .brain_tLaSDI import Brain_tLaSDI


from .data import Data
from .nn import Module

__all__ = [
    'nn',
    'integrator',
    'Brain_tLaSDI',
    'Data',
    'Module',
]
