import torch
from torch import nn
from ....models.builder import ATTENTION_LAYERS, build_attention_layer, build_enrichment, build_local_encoder_layer 

@ATTENTION_LAYERS.register_module()
class empty_layer(nn.Module):
    def __init__(self):
        super().__init__()
        
    # base follows tft- others will reimplement forward    #
    def forward(self,x):
        x['attn_data'] = x['encoded_data']
        return x