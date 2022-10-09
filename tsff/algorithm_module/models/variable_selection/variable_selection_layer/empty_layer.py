import torch
from torch import nn
from tsff.algorithm_module.models.builder import VARIABLE_SELECTION
from tsff.algorithm_module.models.layers import *
from tsff.algorithm_module.models.layers.sub_modules import VariableSelectionNetwork

@VARIABLE_SELECTION.register_module()
class empty_variable_selection_layer(nn.Module):

    # set 
    def __init__(self,
            input_sizes,
            hidden_size,
            dropout, 
            context_size = None
            ):

        super().__init__()
    
    def forward(self,embeddings,
                static_context_vector = None,
                selected_variables = None,
                max_encoder_length = None,
                encoder_flag = None):

        if encoder_flag == "encoder":
            embeddings = {
                name: embeddings[name][:, :max_encoder_length] for name in selected_variables}

        elif encoder_flag == "decoder":
            embeddings = {
                name: embeddings[name][:, max_encoder_length:] for name in selected_variables}

        elif encoder_flag == "static":
            embeddings = {
                name: embeddings[name][:, 0] for name in selected_variables}
        else:
            raise AssertionError("not implemented")

        embeddings = torch.stack([embeddings[i] for i in embeddings], dim = -1)
        embeddings = embeddings.sum(-1)

        embeddings, sparse_weight = embeddings, None
        return embeddings, sparse_weight