import torch
from torch import nn
import torch.functional as F
from tsff.algorithm_module.models.builder import POST_ATTENTION

@POST_ATTENTION.register_module()
class single_softmax_output(nn.Module):

    # set 
    def __init__(self,
            hidden_size,
            dropout,
            output_size
            ):

        super(single_softmax_output,self).__init__()
        self.output_layer = nn.Linear(hidden_size,output_size)
        self.cls_layer = nn.Softmax()
    
    def forward(self, x):
        _x = x['attn_data']
        _x = self.output_layer(_x)
        _x = self.cls_layer(_x)
        x['output_data'] = _x
        return x 

