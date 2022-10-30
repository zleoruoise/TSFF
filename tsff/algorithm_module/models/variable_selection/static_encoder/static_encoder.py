import torch
from torch import embedding, nn
from tsff.algorithm_module.models.builder import STATIC_ENCODER, VARIABLE_SELECTION
from tsff.algorithm_module.models.layers import *
from tsff.algorithm_module.models.layers.sub_modules import VariableSelectionNetwork

@STATIC_ENCODER.register_module()
class base_static_encoder(nn.Module):

    # set 
    def __init__(self,
            input_sizes,
            hidden_size,
            dropout, 
            lstm_layers
            ):
        super().__init__()

        self.context_layer = GatedResidualNetwork(input_size = input_sizes,
                                            hidden_size= hidden_size,
                                            output_size = hidden_size,
                                            dropout = dropout,
        )

        self.hidden_layer = GatedResidualNetwork(input_size = input_sizes,
                                            hidden_size= hidden_size,
                                            output_size = hidden_size,
                                            dropout = dropout,
        )

        self.cell_layer = GatedResidualNetwork(input_size = input_sizes,
                                            hidden_size= hidden_size,
                                            output_size = hidden_size,
                                            dropout = dropout,
        )

        self.enrichment_layer = GatedResidualNetwork(input_size = input_sizes,
                                            hidden_size= hidden_size,
                                            output_size = hidden_size,
                                            dropout = dropout,
        )
        self.lstm_layers = lstm_layers
    
    def forward(self,embeddings,
                timesteps = None,):
        
        context_variable = self.expand_static_context(self.context_layer(embeddings),timesteps)
        input_hidden = self.hidden_layer(embeddings).expand(self.lstm_layers,-1,-1)
        input_cell = self.cell_layer(embeddings).expand(self.lstm_layers,-1,-1)
        enrichment = self.expand_static_context(self.enrichment_layer(embeddings),timesteps)

        return context_variable,input_hidden.contiguous()  ,input_cell.contiguous(),enrichment
    def expand_static_context(self, context, timesteps):
        """
        add time dimension to static context
        """
        return context[:, None].expand(-1, timesteps, -1)