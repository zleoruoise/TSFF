import collections
import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

from ..utils.builder import build_pipeline,PIPELINES

# change StandardScaler to load pickle and fit only once
@PIPELINES.register_module()
class scaler:
    def __init__(self,mean, std,selected_cols, value_pickle):
        self.mean = mean
        self.std = std 
        self.selected_cols = selected_cols

        self.scaler = StandardScaler() 
        #self.scaler.mean_ = [self.mean[pair][cur_col] for cur_col in selected_col]
        #self.scaler.scale_ = [self.std[pair][cur_col] for cur_col in selected_col] 

    def __call__(self,data):
        x_data = data['x_data']

        result_df = {}
        for pair,datum in x_data.items():
            scaler = StandardScaler()
            scaler.mean_ = [self.mean[pair][cur_col] for cur_col in self.selected_cols]
            scaler.scale_ = [self.std[pair][cur_col] for cur_col in self.selected_cols] 
            datum.loc[:,self.selected_cols] = scaler.transform(datum.loc[:,self.selected_cols].to_numpy())
            result_df.update({pair : datum})

        data['x_data'] = result_df
        return data 