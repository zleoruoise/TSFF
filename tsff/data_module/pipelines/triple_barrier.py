import collections
import os

import pandas as pd
import numpy as np
import torch

from ..utils.builder import build_pipeline,PIPELINES

@PIPELINES.register_module()
class triple_barrier:
    def __init__(self,selected_cols,target_pair,barrier_width,aggtrade = False):
        self.selected_cols = selected_cols
        self.target_pair = target_pair
        self.barrier_width = barrier_width
        self.aggtrade = aggtrade
        if not self.aggtrade:
            self.quantity = 'open'
        else:
            self.quantity = 'Price'


    def __call__(self,data):
        x_data = data['target']

        assert len(self.target_pair) == 1
        cur_data = x_data[self.target_pair[0]].loc[:,self.quantity].values
        #base_value = base_data[self.target_pair[0]]['open'] # this is series -> float 
        cur_data = cur_data.reshape(-1,1)
        # to-Do: change for enc-dec model
        horizon_value = cur_data[1:]
        base_value = cur_data[0].reshape(1,1)

        upper = np.where(horizon_value > base_value * (1 + self.barrier_width),1,0)
        lower = np.where(horizon_value < base_value * (1 - self.barrier_width),1,0)

        if len(upper) > 1:
            upper_a1 = np.argmax(upper)
            lower_a1 = np.argmax(lower)

            if upper_a1 > lower_a1:
                data['target'] = torch.tensor([2], dtype = torch.long)
            elif upper_a1 < lower_a1:
                data['target'] = torch.tensor([0], dtype = torch.long)
            else:
                data['target'] = torch.tensor([1], dtype = torch.long)
        else:
            if upper[0][0] == 1:
                data['target'] = torch.tensor([2], dtype = torch.long)
            elif lower[0][0] == 1:
                data['target'] = torch.tensor([0], dtype = torch.long)
            else:
                data['target'] = torch.tensor([1], dtype = torch.long)

        return data 