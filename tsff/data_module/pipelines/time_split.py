import collections
import os

import pandas as pd
import numpy as np
from torch import is_inference_mode_enabled

from ..utils.builder import build_pipeline,PIPELINES

@PIPELINES.register_module()
class time_split:
    def __init__(self,selected_cols,isremoval = False):
        self.selected_cols = selected_cols
        self.isremoval = isremoval # remove timestamp from x_data 

    def __call__(self,data):
        x_data = data['x_data']

        result_data = dict()
        result_time = dict()
        for key,datum in x_data.items():
            time_stamp ,new_data = datum.loc[:,'real_time'],datum.loc[:,self.selected_cols] 
            result_data.update({key:new_data})
            result_time.update({key:time_stamp})

        if self.isremoval:
            data['x_data'] = result_data
        data['time_stamp'] = time_stamp
        
        return data 