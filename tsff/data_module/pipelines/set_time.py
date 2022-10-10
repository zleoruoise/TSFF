import collections
import os
import time
import datetime as dt
from datetime import datetime

import pandas as pd
import numpy as np
import torch

from ..utils.builder import build_pipeline,PIPELINES

@PIPELINES.register_module()
class set_time:
    def __init__(self,encoder_length,decoder_length,time_interval
                ):
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.time_interval = time_interval 

    def __call__(self,data):
        start_time = data['start_time']

        end_time = start_time + (self.encoder_length + self.decoder_length + 1) * 1000 * self.time_interval
        date_start = datetime.utcfromtimestamp(start_time//1000).strftime('%Y%m')
        date_before = (datetime.utcfromtimestamp(start_time//1000) - dt.timedelta(days=1)).strftime('%Y%m')
        date_end = datetime.utcfromtimestamp(end_time//1000).strftime('%Y%m')


        # make sure self.selected_header - time_stamp always on 0th col - 
        # later used by time_split
        data['date_start'] = date_start
        data['date_before'] = date_before 
        data['date_end'] = date_end 

        data['end_time'] = end_time
        return data 