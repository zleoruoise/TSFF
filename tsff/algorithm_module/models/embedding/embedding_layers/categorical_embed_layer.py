from typing import Dict
import torch
from torch import nn
from tsff.algorithm_module.models.builder import EMBEDDING, EMBEDDING_LAYERS, build_embedding,build_variable_selection

from pytorch_forecasting.models.nn import MultiEmbedding
#from tsff.algorithm_module.models.layers.sub_modules import Multi


@EMBEDDING_LAYERS.register_module()
class categorical_embedding_layer(nn.Module):

    # set 
    def __init__(self,
            embedding_sizes, 
            categorical_groups,
            embedding_paddings,
            x_categoricals,
            max_embedding_size, 
            ):

        super().__init__()
        # categorical embedding
        self.layers =  MultiEmbedding(
            embedding_sizes= embedding_sizes,
            categorical_groups= categorical_groups,
            embedding_paddings= embedding_paddings,
            x_categoricals= x_categoricals,
            max_embedding_size= max_embedding_size,
        )
    
    def forward(self,x_cat) -> Dict[str, torch.Tensor]:

        # input parsing  
        x = x['x_data']

        # embedding 
        input_vectors = self.layers(x_cat)

        return input_vectors