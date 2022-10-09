import torch
from torch import nn
import torch.functional as F
from tsff.algorithm_module.models.builder import POST_ATTENTION
from tsff.algorithm_module.models.layers import *

@POST_ATTENTION.register_module()
class dec_post_attention(nn.Module):

    # set 
    def __init__(self,
            hidden_size,
            dropout,
            output_size
            ):

        super(dec_post_attention,self).__init__()
        self.output_layer = nn.Linear(hidden_size,output_size)
    
    def forward(self, x):
        _x = x['attn_data']
        _x = self.output_layer(_x)
        x['output_data'] = _x
        return x 

