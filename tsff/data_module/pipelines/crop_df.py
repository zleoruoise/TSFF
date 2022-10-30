import collections
import os

import pandas as pd
import numpy as np

from ..utils.builder import build_pipeline,PIPELINES

@PIPELINES.register_module()
class crop_df:
    def __init__(self,encoder_length, decoder_length,time_interval,pointnet = False):
        self.encoder_length = encoder_length 
        self.decoder_length = decoder_length 
        self.time_interval = time_interval
        self.pointnet = pointnet

    def __call__(self,data):
        x_data = data['x_data']
        start_time = data['start_time']
        end_time = data['end_time']

        result_df = {}
        # data_length is 1 step larger - because we want target to 1 step ahead of decoder input
        # also encoder_length in cfg is 1 larger than target encoder_length, this is because of differencing
        data_length = self.encoder_length + self.decoder_length + 1 
        for key,value in x_data.items():
            cur_df = value.loc[(value['real_time'] >= start_time) &
                                (value['real_time'] < end_time),:]
            result_df.update({key : cur_df})

        if self.pointnet == False:
            # if obs is missing
            for key,value in result_df.items():
                length_diff = value.shape[0] - data_length
                if abs(length_diff) > 2:
                    raise Exception("more than one error")
                elif length_diff < 0:
                    time_diff = value['real_time'].diff(periods = 1)
                    diff_idx = np.where(time_diff > self.time_interval * 1000 )[0]
                    if len(diff_idx) == 0: # if the last row is missing
                        diff_idx = [value.shape[0]-1] 
                    else: # more precise - only consider when found error is one way only
                        diff_idx = diff_idx[:abs(length_diff)]
                    for i in diff_idx:
                        new_value = pd.concat([value.iloc[:i,:],
                                                value.iloc[i:i+1,:],
                                                value.iloc[i:].set_index( 
                                                    value.index[i:]+1)])
                        result_df[key] = new_value
                elif length_diff > 0 :
                    time_diff = value['real_time'].diff(periods = 1)
                    diff_idx = np.where(time_diff < self.time_interval * 1000 )
                    for i in diff_idx:
                        new_value = pd.concat([value.iloc[:diff_idx,:],
                                                value.iloc[diff_idx+1:].set_index( 
                                                    value.index[diff_idx:]-1)])
                        result_df[key] = new_value

        data['x_data'] = result_df
        
        return data 