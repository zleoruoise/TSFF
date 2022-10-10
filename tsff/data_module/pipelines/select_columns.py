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
            cur_data = datum.loc[:,self.selected_headers]
            result_df.update({pair : cur_data})
        
        data['x_data'] = result_df
        return data 