import enum
import torch
from torch import nn
import torch.functional as F
from tsff.algorithm_module.models.builder import POST_ATTENTION
from tsff.algorithm_module.models.layers import *

@POST_ATTENTION.register_module()
class single_pred(nn.Module):
    def __init__(self,
                d_model:int,
                hidden_size: int,
                num_classes: int,
                time_steps: int,
                num_layers: int,
                ):
        super(single_pred,self).__init__()

        mlp_layers = []

        mlp_layers.append(nn.Conv1d(d_model * d_model, hidden_size,1))
        if num_layers > 2:
            for i in range(num_layers-2):
                mlp_layers.append(nn.Conv1d(hidden_size,hidden_size,1))
        mlp_layers.append(nn.Conv1d(hidden_size,num_classes,1))

        self.mlp_layers = nn.ModuleList(mlp_layers)

    def forward(self,x):
        _x = x['attn_data']

        B,T,C = _x.size()
        _x = _x.view(B,T*C)
        for i,layer in enumerate(self.mlp_layers):
            _x = layer(_x)
        x['output_data'] = x
        return x 

