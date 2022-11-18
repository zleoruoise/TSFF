
from tsff.algorithm_module.models.layers.sub_modules import AddNorm, GatedLinearUnit
from pytorch_forecasting import LSTM
import torch
from torch import nn
from tsff.algorithm_module.models.builder import LOCAL_ENCODER_LAYER, build_enrichment, build_local_encoder_layer 

@LOCAL_ENCODER_LAYER.register_module()
class lstm_encoder(nn.Module):

    # set 
    def __init__(self,
            # variable selection
            input_size, 
            hidden_size, 
            num_layers,
            dropout,
            ):

        super().__init__()
        # lstm encoder (history) and decoder (future) for local processing
        self.local_encoder_layer= LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,) 

        self.post_lstm_gate = GatedLinearUnit(input_size,hidden_size,dropout)
        self.post_lstm_add_norm = AddNorm(hidden_size)


    def forward(self,embeddings,initial_states, encoder_length,
                ):

        output, (hidden,cell) = self.local_encoder_layer(embeddings,initial_states, lengths = encoder_length)
        lstm_output = self.post_lstm_gate(output)
        lstm_output = self.post_lstm_add_norm(lstm_output,embeddings)

        return lstm_output, output, (hidden,cell)


        
                                        
