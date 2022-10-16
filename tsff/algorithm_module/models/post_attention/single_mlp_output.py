import torch
from torch import nn
import torch.functional as F
from tsff.algorithm_module.models.builder import POST_ATTENTION
from tsff.algorithm_module.models.layers import *

@POST_ATTENTION.register_module()
class single_mlp_output(nn.Module):

    # set 
    def __init__(self,
            hidden_size,
            dropout,
            output_size,
            ):

        super(single_mlp_output,self).__init__()
        self.output_layer = nn.Linear(hidden_size,output_size)
    
    def forward(self, x):
        _x = x['attn_data']
        _x = self.output_layer(_x)
        x['output_data'] = _x
        return x 

