import collections
import os

import pandas as pd
import numpy as np
import torch

from ..utils.builder import build_pipeline,PIPELINES

@PIPELINES.register_module()
class convert_np2ts:
    def __init__(self,keys):
        self.keys = keys 

    def __call__(self,data):
        for key in self.keys:
            new_data = torch.from_numpy(data[key].to_numpy().astype(np.float32))
            data[key] = new_data

        return data 