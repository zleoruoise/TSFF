from typing import Dict
from tsff.algorithm_module.models.builder import EMBEDDING, build_embedding, build_embedding_layers,build_variable_selection

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math

@EMBEDDING.register_module()
class multitype_embedding(nn.Module):

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
                categorical_embedding_layer = None,
                continuous_embedding_layer = None,

                ):
        super().__init__()
        # categorical embedding
        if categorical_embedding_layer is not None:
            self.categorical_embedding_layer = build_embedding_layers(categorical_embedding_layer)
        else:
            self.categorical_embedding = None
        self.continuous_embedding_layer = build_embedding_layers(continuous_embedding_layer)

    
    def forward(self,x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        # TO:DO - value embedding should be pfe 
        cont = self.continuous_embedding_layer(x)

        if hasattr(self,'categorical_embedding_layer'):
            cat = self.categorical_embedding_layer(x)
            embedding_value = cat + cont
        else:
            embedding_layer = cont
        

        # need to change this 
        x['embedded_data'] =  embedding_layer
        return x