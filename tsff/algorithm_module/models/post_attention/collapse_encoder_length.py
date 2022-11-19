import torch
from torch import nn
import torch.functional as F
from tsff.algorithm_module.models.builder import POST_ATTENTION

@POST_ATTENTION.register_module()
class collapse_mlp_output(nn.Module):

    # set 
    def __init__(self,
            hidden_size,
            encoder_length,
            time_interval,
            dropout,
            output_size,
            ):

        super(collapse_mlp_output,self).__init__()
        self.output_layer = nn.Linear(encoder_length*time_interval*hidden_size,output_size)
    
    def forward(self, x):
        _x = x['attn_data']
        _x = _x.view(_x.size(0),-1)
        _x = self.output_layer(_x)
        x['output_data'] = _x
        return x 

