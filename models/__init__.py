from __future__ import absolute_import

from .bert import *

__factory = {
    'transformer_pvp': Transformer_PVP,
    'transformer': Transformer_final,
    'transformer_pre_fc_tanh_init': Transformer_Pre_fc_tanh_init,
    'transformer_pre_init': Transformer_Pre_fc_tanh_init,

}

def get_names():
    return __factory.keys()

def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)
