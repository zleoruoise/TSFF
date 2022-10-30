from typing import Dict
from tsff.algorithm_module.models.builder import EMBEDDING_LAYERS
from .fixed_embedding import fixed_embedding

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math

@EMBEDDING_LAYERS.register_module()
class pillar_positional(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(pillar_positional, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        _x = x['coords']
        return self.pe[:, :_x.size(1)]