import collections
import os
from glob import glob
import datetime as dt
from dateutil.relativedelta import relativedelta
from datetime import datetime

import pandas as pd

from ..utils.builder import build_pipeline,PIPELINES

@PIPELINES.register_module()
class load_memory_seq:
    def __init__(self,pairs,data_path,headers,start_time,end_time):
        self.pairs = pairs
        self.data_path = data_path
        self.headers = headers

    def __call__(self):

        def load_df_m_kl(pair,_data_path,headers):
            cur_dfs = []
            data_path = os.path.join(_data_path,pair,'*.csv')
            path_list = sorted(glob(data_path))
            for cur_path in path_list:
                cur_df = pd.read_csv(cur_path)
                cur_df.columns = headers 
                cur_dfs.append(cur_df)
            return cur_dfs

        df_dict = {}

        for pair in self.pairs:
            df_dict.update({pair : load_df_m_kl(pair,self.data_path,self.headers)})
        
        data = df_dict

        return data 