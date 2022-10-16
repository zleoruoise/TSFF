import collections
import os

import pandas as pd

from ..utils.builder import build_pipeline,PIPELINES

@PIPELINES.register_module()
class load_dfs_time:
    def __init__(self,pairs,data_path,headers):
        self.pairs = pairs
        self.data_path = data_path
        self.headers = headers

    def __call__(self,data):

        def load_df_m_kl(dates,pair,_data_path,headers):
            assert len(dates) > 0
            cur_dfs = []
            for date in dates:
                cur_date = f"{pair}-1m-{date[:4]}-{date[4:6]}.csv"
                data_path = os.path.join(_data_path,pair,'1m',cur_date)
                cur_df = pd.read_csv(data_path)
                cur_df.columns = headers 
                cur_dfs.append(cur_df)
            df = pd.concat(cur_dfs, axis = 0, ignore_index = True)
            return df

        df_dict = {}
        date_start = data['date_start']
        date_end = data['date_end']

        if date_start == date_end:
            dates = [date_start]
        elif date_start != date_end :
            dates = [date_start, date_end]
        else:
            assert 1


        for pair in self.pairs:
            df_dict.update({pair : load_df_m_kl(dates,pair,self.data_path,self.headers)})
        
        data['x_data'] = df_dict

        return data 