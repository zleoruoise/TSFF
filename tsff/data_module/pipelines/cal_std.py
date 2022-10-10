import collections
import os

import pandas as pd
import numpy as np
import torch

from ..utils.builder import build_pipeline,PIPELINES

@PIPELINES.register_module()
class cal_std:
    def __init__(self):
        pass

    def __call__(self,data):
        x_data = data['x_data']

        mean = dict()
        std = dict()
        for key,value in x_data.items():
            mean = dict()
            std = dict() 
            for col in value.columns:
                if col == 'real_time':
                    continue
                mean.update({col:value[col].mean()})
                std.update({col:value[col].std()})
            mean.update({key:mean})
            std.update({key:std})
        print(mean)
        print(std)
        # pickle later
