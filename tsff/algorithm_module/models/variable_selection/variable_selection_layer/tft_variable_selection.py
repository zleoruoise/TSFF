import torch
from torch import nn
from tsff.algorithm_module.models.builder import VARIABLE_SELECTION
from tsff.algorithm_module.models.layers import *
from tsff.algorithm_module.models.layers.sub_modules import VariableSelectionNetwork

@VARIABLE_SELECTION.register_module()
class base_tft_variable_selection(nn.Module):

    # set 
    def __init__(self,
            input_sizes,
            hidden_size,
            dropout, 
            context_size = None
            ):

        super().__init__()
        self.layer = VariableSelectionNetwork(input_sizes = input_sizes,
                                            hidden_size= hidden_size,
                                            dropout = dropout,
                                            context_size= context_size,
        )
    
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

        embeddings, sparse_weight = self.layer(embeddings,static_context_vector)
        return embeddings, sparse_weight