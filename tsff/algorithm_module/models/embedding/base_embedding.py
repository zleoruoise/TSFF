from typing import Dict
from tsff.algorithm_module.models.builder import EMBEDDING, build_embedding, build_embedding_layers,build_variable_selection

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math

@EMBEDDING.register_module()
class base_embedding(nn.Module):
    '''
    This module will apply embedding on both covariate[price, quantity] and time stamp.
    There are 3 different embeddings: value, positional, temporal

    Args:
        value_embedding: function maps [N,T,M,C] -> [N,T,C], in order to extract local features 
        positoinal_embedding: function maps [N,T,M,C] -> [N,T,C], only accounts positions in T, M is ignored
        temporal_embedding: function maps [N,T,M,C] -> [N,T,C], account mean value of timestamp values in 1st dim

    Returns:
        x['embedded_data']: torch.Tensor[N,T,C], this will be treated as regular time-series data by later modules.
    '''
    def __init__(self,
                value_embedding = None,
                positional_embedding = None,
                temporal_embedding = None):
        super().__init__()
        # categorical embedding
        self.value_embedding= build_embedding_layers(value_embedding)
        self.position_embedding = build_embedding_layers(positional_embedding)
        self.temporal_embedding = build_embedding_layers(temporal_embedding)

    
    def forward(self,x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        # TO:DO - value embedding should be pfe 
        if self.position_embedding:
            position = self.position_embedding(x)
        if self.temporal_embedding:
            time = self.temporal_embedding(x)
        if self.value_embedding:
            value = self.value_embedding(x)

        # need to change this 
        x['embedded_data'] = value + position + time
        return x