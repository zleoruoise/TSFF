from typing import Dict
import torch
from torch import nn
import math
from tsff.algorithm_module.models.builder import EMBEDDING, EMBEDDING_LAYERS, build_embedding,build_variable_selection

from pytorch_forecasting.models.nn import MultiEmbedding
#from tsff.algorithm_module.models.layers.sub_modules import Multi


@EMBEDDING_LAYERS.register_module()
class continuous_embedding_layer(nn.Module):

    # set 
    def __init__(self,
            hidden_continuous_size, 
            num_cov = 1, # cont - prescaler
            max_len = 5000
            ):

        super().__init__()
        # categorical embedding
        self.layers = nn.Linear(num_cov,hidden_continuous_size)
        pe = torch.zeros(max_len, hidden_continuous_size).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, hidden_continuous_size, 2).float() * -(math.log(10000.0) / hidden_continuous_size)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    
    def forward(self,x) -> Dict[str, torch.Tensor]:
        x = x['x_data']

        input_vectors = self.layers(x)
        positional_embedding = self.pe[:,:input_vectors.size(1)]
        input_vectors = input_vectors + positional_embedding
    
        return input_vectors