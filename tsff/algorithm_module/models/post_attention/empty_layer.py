import torch
from torch import nn
import torch.functional as F
from tsff.algorithm_module.models.builder import POST_ATTENTION
from tsff.algorithm_module.models.layers import *

@POST_ATTENTION.register_module()
class empty_layer(nn.Module):

    # set 
    def __init__(self):
        super(empty_layer,self).__init__()
    
    def forward(self,x):
        x['output_data'] = x['attn_data']
        return x 

