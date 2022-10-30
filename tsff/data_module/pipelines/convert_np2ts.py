import collections
import os

import pandas as pd
import numpy as np
import torch

from ..utils.builder import build_pipeline,PIPELINES

@PIPELINES.register_module()
class convert_np2ts:
    def __init__(self,df_keys=[],numpy_keys=[]):
        self.df_keys = df_keys 
        self.numpy_keys = numpy_keys

    def __call__(self,data):
        for key in self.df_keys:
            new_data = torch.from_numpy(data[key].to_numpy().astype(np.float32))
            data[key] = new_data
        for key in self.numpy_keys:
            new_data = torch.from_numpy(data[key].astype(np.float32))
            data[key] = new_data

        return data
