
import torch
from torch import nn
from tsff.algorithm_module.models.builder import LOCAL_ENCODER, build_enrichment, build_local_encoder_layer 

@LOCAL_ENCODER.register_module()
class empty_layer(nn.Module):

    # set 
    def __init__(self):
        super().__init__()

    def forward(self,x):
        x['encoded_data'] = x['selected_data']
        return x


        
                                        
