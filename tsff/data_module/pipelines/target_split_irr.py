import collections
import os
from typing import Any, Callable, Dict, List, Tuple, Union

import pandas as pd
import numpy as np
import torch

from ..utils.builder import build_pipeline,PIPELINES

@PIPELINES.register_module()
class target_split_irr:
    def __init__(self,selected_cols, encoder_length, decoder_length,time_interval):
        self.selected_cols = selected_cols
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.time_interval = time_interval

    def __call__(self,data):
        x_data = data['x_data']
        start_time = data['start_time']
        end_time = data['end_time']

        encoder_length = self.encoder_length
        decoder_length = self.decoder_length

        cov_result_df = {}
        target_result_df = {}
        target_base_price= {}

        def diff_price(df_dict:Dict[str,pd.DataFrame]):
            result_df = {}
            for pair,datum in df_dict.items():
                ori_cols = list(datum.columns)
                # only selected_cols are diffed -> time is not diffed 
                cur_cols = [i for i in ori_cols if i not in self.selected_cols]
                new_df = datum.loc[:,self.selected_cols].diff(axis= 0,periods =1)
                new_df = pd.concat([datum.loc[datum.index[1]:,cur_cols],new_df.iloc[1:,:]],axis = 1)
                result_df.update({pair : new_df})

            return result_df

        for pair,datum in x_data.items():
            new_df = datum.loc[datum['real_time'] < start_time + encoder_length * 1000* self.time_interval,:]
            cov_result_df.update({pair : new_df})

        # diff base price in encoder data
        cov_result_df = diff_price(cov_result_df)

        for pair,datum in x_data.items():
            new_df = datum.loc[datum['real_time'] > start_time + encoder_length * 1000* self.time_interval,:]
            new_df_base = datum.loc[datum['real_time'] > start_time + encoder_length * 1000 * self.time_interval,:]

            target_result_df.update({pair : new_df})
            target_base_price.update({pair : new_df_base})

        target_result_df = diff_price(target_result_df)
        
        data['x_data'] = cov_result_df
        data['y_data'] = target_result_df
        data['target'] =  target_base_price

        return data 