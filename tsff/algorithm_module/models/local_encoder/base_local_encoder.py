
import torch
from torch import nn
from tsff.algorithm_module.models.builder import LOCAL_ENCODER, build_enrichment, build_local_encoder_layer 

@LOCAL_ENCODER.register_module()
class base_local_encoder(nn.Module):

    # set 
    def __init__(self,
            local_encoder_layer,
            local_decoder_layer):

        super().__init__()
        # local encoder layer include gate encoder part  not just LSTM 
        self.local_encoder_layer = build_local_encoder_layer(local_encoder_layer,)

        self.local_decoder_layer = build_local_encoder_layer(local_decoder_layer)


    def forward(self,embeddings_varying_encoder, embeddings_varying_decoder,
                static_input_hidden, static_input_cell, 
                encoder_lengths, decoder_lengths
                ):

        lstm_output_encoder, encoder_output, (hidden,cell) = self.local_encoder_layer(embeddings_varying_encoder, 
                                                                (static_input_hidden,
                                                                static_input_cell), 
                                                                encoder_lengths)
                                                            
        lstm_output_decoder, _, _ = self.local_decoder_layer(embeddings_varying_decoder, 
                                                    (hidden,cell), decoder_lengths)

        #lstm_output = torch.cat([lstm_output_encoder, lstm_output_decoder], dim=1)
        #attn_output = self.static_enrichment(lstm_output, static_context_enrichment)

        return lstm_output_encoder, lstm_output_decoder


        
                                        
