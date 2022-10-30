from typing import Dict
from tsff.algorithm_module.models.builder import EMBEDDING_LAYERS
from .fixed_embedding import fixed_embedding

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math

@EMBEDDING_LAYERS.register_module()
class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 60
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = fixed_embedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x

@EMBEDDING_LAYERS.register_module()
class pillar_temporal(nn.Module):
    def __init__(self, d_model):
        super(pillar_temporal, self).__init__()


        #Embed = fixed_embedding if embed_type == 'fixed' else nn.Embedding
        self.timestamp = nn.Linear(1, d_model)

    def forward(self, x: torch.Tensor):
        '''
        x['time_stamp]: [B,T,M,1]
        returns = [B,T,d_model]
        '''
        _x = x['time_stamp']
        _x = _x.unsqueeze(-1)

        #_x = _x.mean(-2)
        _x = self.timestamp(_x)

        return _x