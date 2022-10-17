import collections
import os

import pandas as pd
import numpy as np

from ..utils.builder import build_pipeline,PIPELINES

@PIPELINES.register_module()
class crop_df_seq:
    def __init__(self,encoder_length, decoder_length,time_interval,pointnet = False):
        self.encoder_length = encoder_length 
        self.decoder_length = decoder_length 
        self.time_interval = time_interval
        self.pointnet = pointnet

    def __call__(self,data):
        x_data = data['x_data']
        base_idx = data['base_idx']
        inside_idx = data['inside_idx']

        result_df = {}
        data_length = self.encoder_length + self.decoder_length # cuz loc is inclusive to endpoints
        for key,value in x_data.items():
            cur_data = value[base_idx]
            cur_df = cur_data.loc[inside_idx:(inside_idx + data_length),:]
            result_df.update({key : cur_df})

        data['x_data'] = result_df
        
        return data 