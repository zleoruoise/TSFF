import torch
from torch import embedding, nn
from tsff.algorithm_module.models.builder import STATIC_ENCODER, VARIABLE_SELECTION
from tsff.algorithm_module.models.layers import *
from tsff.algorithm_module.models.layers.sub_modules import VariableSelectionNetwork

@STATIC_ENCODER.register_module()
class empty_static_encoder(nn.Module):

    # set 
    def __init__(self,
            input_sizes,
            hidden_size,
            dropout, 
            lstm_layers
            ):
        super().__init__()

    
    def forward(self,embeddings,
                timesteps = None,):
        
        return None,None,None,None
    def expand_static_context(self, context, timesteps):
        """
        add time dimension to static context
        """
        return context[:, None].expand(-1, timesteps, -1)