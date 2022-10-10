import collections
import os

import pandas as pd
import numpy as np
import torch

from ..utils.builder import build_pipeline,PIPELINES

@PIPELINES.register_module()
class regular_concat:
    def __init__(self,selected_cols):
        self.selected_cols = selected_cols

    def __call__(self,data):
        x_data = data['x_data']

        # make sure self.selected_header - time_stamp always on 0th col - 
        # later used by time_split
        
        df_list = []
        for pair,datum in x_data.items():
            cur_data = datum.loc[:,self.selected_cols]
            df_list.append(cur_data.reset_index(drop = True))
        result_df = pd.concat(df_list, axis = 1, ignore_index= True)

        data['x_data'] = result_df
        return data 