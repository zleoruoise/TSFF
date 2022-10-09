import torch
from torch import nn
import torch.functional as F
from ....models.builder import ENRICHMENT 
from ....models.layers import *

@ENRICHMENT.register_module()
class empty_layer(nn.Module):

    # set 
    def __init__(self):

        super(empty_layer,self).__init__()
    
    def forward(self,lstm_output_encoder, lstm_output_decoder,static_context_enrichment):

        return lstm_output_encoder 

