import collections
import os
import pickle

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

from ..utils.builder import build_pipeline,PIPELINES

# change StandardScaler to load pickle and fit only once
@PIPELINES.register_module()
class scaler:
    def __init__(self,pickle_path,selected_cols):
        self.pickle_path = pickle_path 
        
        self.selected_cols = selected_cols
        with open(self.pickle_path, 'rb') as f:
            self.scaler_dict = pickle.load(f)

    def __call__(self,data):
        x_data = data['x_data']

        result_df = {}
        for pair,datum in x_data.items():
            cur_scaler = self.scaler_dict[pair]
            datum.loc[:,self.selected_cols] = cur_scaler.transform(datum.loc[:,self.selected_cols].to_numpy())
            result_df.update({pair : datum})

        data['x_data'] = result_df
        return data 