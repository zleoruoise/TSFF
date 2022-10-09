import torch
from torch import nn
from tsff.algorithm_module.models.builder import VARIABLE_SELECTION, build_embedding, build_enrichment, build_static_encoder,build_variable_selection, build_variable_selection_layer

@VARIABLE_SELECTION.register_module()
class empty_layer(nn.Module):

    # set 
    def __init__(self):
        super().__init__()

    def forward(self,x):
        x['selected_data'] = x['embedded_data']
        return x