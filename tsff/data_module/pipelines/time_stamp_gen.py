import collections
import os

import pandas as pd
import numpy as np
from torch import is_inference_mode_enabled

from ..utils.builder import build_pipeline,PIPELINES

@PIPELINES.register_module()
class time_stamp_gen:
    def __init__(self,encoder_length,decoder_length,time_interval,):
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length 
        self.time_interval = time_interval

    def __call__(self,data):
        time_interval = self.time_interval * 1000
        start_time = data['start_time']
        end_time = start_time + (self.encoder_length + self.decoder_length) * 60 * time_interval
        time_stamp = np.arange(start_time + time_interval*0.5,end_time,
                                time_interval)
        data['time_stamp'] = time_stamp

        
        return data 