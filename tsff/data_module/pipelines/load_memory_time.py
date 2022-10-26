import collections
import os
import datetime as dt
from dateutil.relativedelta import relativedelta
from datetime import datetime

import pandas as pd

from ..utils.builder import build_pipeline,PIPELINES

@PIPELINES.register_module()
class load_memory_time:
    def __init__(self,pairs,data_path,headers,start_time,end_time,data_type = 'klines'):
        self.pairs = pairs
        self.data_path = data_path
        self.headers = headers
        self.start_time = datetime.strptime(start_time , "%Y%m%d")
        self.end_time = datetime.strptime(end_time, "%Y%m%d")
        self.data_type = data_type

    def __call__(self):

        def load_df_m(dates,pair,_data_path,headers):
            assert len(dates) > 0
            cur_dfs = []
            for date in dates:
                if self.data_type == 'klines':
                    cur_date = f"{pair}-1m-{date[:4]}-{date[4:6]}.csv"
                    data_path = os.path.join(_data_path,pair,'1m',cur_date)
                elif self.data_type == 'aggtrades':
                    cur_date = f"{pair}-aggTrades-{date[:4]}-{date[4:6]}.csv"
                    data_path = os.path.join(_data_path,pair,cur_date)
                cur_df = pd.read_csv(data_path)
                cur_df.columns = headers 
                cur_dfs.append(cur_df)
            df = pd.concat(cur_dfs, axis = 0, ignore_index = True)
            return df

        df_dict = {}
        month_diff = relativedelta(self.end_time,self.start_time)
        start_date = self.start_time
        dates = [start_date.strftime('%Y%m')]
        for _ in range(int(month_diff.months + 12 * month_diff.years)):
            date = start_date + relativedelta(months=+1) 
            dates.append(date.strftime('%Y%m'))
            start_date = date

        for pair in self.pairs:
            df_dict.update({pair : load_df_m(dates,pair,self.data_path,self.headers)})
        
        data = df_dict

        return data 