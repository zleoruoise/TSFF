import collections
import os

import pandas as pd
import numpy as np
import torch

from ..utils.builder import build_pipeline,PIPELINES

@PIPELINES.register_module()
class diff_price:
    def __init__(self,selected_cols):
        self.selected_cols = selected_cols

    def __call__(self,data):
        x_data = data['x_data']

        result_df = {}
        for pair,datum in x_data.items():
            ori_cols = list(datum.columns)
            cur_cols = [i for i in ori_cols if i not in self.selected_cols]
            new_df = datum.loc[:,self.selected_cols].diff(axis= 0,periods =1)
            new_df = pd.concat([new_df.iloc[1:,:],datum.loc[datum.index[1]:,cur_cols]],axis = 1)
            result_df.update({pair : new_df})

        data['x_data'] = result_df
        return data 