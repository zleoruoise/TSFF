import collections
import os

import pandas as pd
import numpy as np
import torch

from ..utils.builder import build_pipeline,PIPELINES

@PIPELINES.register_module()
class select_columns:
    def __init__(self,selected_headers):
        self.selected_headers = selected_headers 

    def __call__(self,data):
        x_data = data['x_data']

        result_df = {}
        for pair,datum in x_data.items():
            if isinstance(datum,pd.DataFrame):
                cur_data = datum.loc[:,self.selected_headers]
            elif isinstance(datum, list):
                cur_data = []
                for _datum in datum:
                    cur_df = _datum.loc[:,self.selected_headers]
                    cur_data.append(cur_df)
            # overall update - list or df 
            result_df.update({pair : cur_data})
                
        
        data['x_data'] = result_df
        return data 