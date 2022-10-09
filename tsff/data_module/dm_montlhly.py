"""
Timeseries datasets.

Timeseries data is special and has to be processed and fed to algorithms in a special way. This module
defines a class that is able to handle a wide variety of timeseries data problems.
"""
# this should be 1 1mins dataset
GLOBAL_TIMEDIFF = 9*60*60

from copy import copy as _copy, deepcopy
from functools import lru_cache, partial
import os 
import inspect
import time
import datetime as dt
from datetime import datetime
from typing import Any, Callable, Dict, List, Tuple, Union
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import RobustScaler, StandardScaler

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

from spconv.utils import Point2VoxelCPU1d
from cumm import tensorview as tv

from tsff.data_module.utils.builder import build_pipeline,DATASETS
from tsff.data_module.utils.compose  import Compose

@DATASETS.register_module()
class monthly_dataset(Dataset):
    """
    PyTorch Dataset for fitting timeseries models.

    The dataset automates common tasks such as


    * scaling and encoding of variables
    * normalizing the target variable
    * efficiently converting timeseries in pandas dataframes to torch tensors
    * holding information about static and time-varying variables known and unknown in the future
    * holiding information about related categories (such as holidays)
    * downsampling for data augmentation
    * generating inference, validation and test datasets
    * etc.
    """

    def __init__(
        self,
        data_path: str,
        pairs: List[str],
        target_pair: str,
        start_date: str, 
        end_date: str, 
        time_interval: float = 1,
        weight: Union[str, None] = None,
        encoder_length: int = 30,
        decoder_length: int = 1,
        val_cutoff: float = 0.8,
        transforms: List[str] = [],
        predict_mode: bool = False,
        data_type: str = 'ohlcv',
        max_obs: int = 1, # pointpillar specific
        batch_size: int = 8,
        mean: List[float] = [0,0,0,0,10],
        std: List[float] = [0,0,0,0,100],

    ):
        """
        test
        Args:
            pairs: including target column and covariates column.
            start_date (str): %Y%m%d
            end_date (str): %Y%m%d
            
        """
        super().__init__()
        self.data_path = data_path
        self.pairs = pairs
        self.target_pair = target_pair

        self.start_date = start_date
        self.end_date = end_date
        self.time_interval = time_interval

        self.encoder_length = encoder_length

        self.decoder_length = decoder_length 

        self.weight = weight
        # set automatic defaults
        self.predict_mode = predict_mode

        self.max_obs = max_obs # this is one for pointpillar dataset - per each voxel
        self.val_cutoff = val_cutoff
        self.batch_size = batch_size

        # 
        self.mean = mean
        self.std = std


        # headers
        if data_type == 'ohlcv':
            self.headers =('real_time', 'open', 'high','low','close','volume',
            'Close_time','Quote_asset_volumne','Number_of_trades','Taker_buy_base_asset_volume',"Taker_buy_quote_asset_volume",'ignore') 
            self.selected_headers =  ("real_time","open","close","high","low","volume")
            self.selected_cols = ("open","close","high","low","volume")
        elif data_type == 'aggTrade':
            # change later 
            self.headers = ['Agg_tradeId','Price','Quantity','First_tradeID','Last_tradeID','Timestamp','maker','bestPrice'] # copy from github
            self.selected_headers= ['Timestamp', 'Price', 'Quantity']
        else:
            AssertionError('data_type not implemented')

        # filter data

        self.index = self._construct_index()

        # ToDo: seperate classes for each transformations - refactoring
        # set transforms 
        #self.transform = Compose(transforms)

        # set 

        # convert to torch tensor for high performance data loading later


    def save(self, fname: str) -> None:
        """
        Save dataset to disk

        Args:
            fname (str): filename to save to
        """
        torch.save(self, fname)

    @classmethod
    def load(cls, fname: str):
        """
        Load dataset from disk

        Args:
            fname (str): filename to load from

        Returns:
            TimeSeriesDataSet
        """
        obj = torch.load(fname)
        assert isinstance(obj, cls), f"Loaded file is not of class {cls}"
        return obj


    def _construct_index(self,) -> List[float]:
        """
        Create index of samples.

        Args:
            data (pd.DataFrame): preprocessed data
            predict_mode (bool): if to create one same per group with prediction length equals ``max_decoder_length``

        Returns:
            pd.DataFrame: index dataframe
        """
        # file
        start_time = datetime.strptime(self.start_date, "%Y%m%d")
        start_time = time.mktime(start_time.timetuple()) - GLOBAL_TIMEDIFF
        end_time = datetime.strptime(self.end_date, "%Y%m%d") + dt.timedelta(days=1)
        end_time = time.mktime(end_time.timetuple()) - GLOBAL_TIMEDIFF

        start_time *= 1000
        end_time *= 1000
        time_interval = self.time_interval * 1000

        index_val = np.arange(start_time,end_time,time_interval)

        return index_val

    def __len__(self) -> int:
        """
        Length of dataset.

        Returns:
            int: length
        """
        lagged_time = self.encoder_length + self.decoder_length 
        return len(self.index) - int(lagged_time)

    def load_dfs(self,dates):
        df_dict = {}
        for pair in self.pairs:
            df_dict.update({pair : self.load_df_m_kl(dates,pair)})

        return df_dict


    def load_df_m_kl(self,dates,pair):
        assert len(dates) > 0
        cur_dfs = []
        for date in dates:
            cur_date = f"{pair}-1m-{date[:4]}-{date[4:6]}.csv"
            data_path = os.path.join(self.data_path,pair,'1m',cur_date)
            cur_df = pd.read_csv(data_path)
            cur_df.columns = self.headers 
            cur_dfs.append(cur_df)
        df = pd.concat(cur_dfs, axis = 0, ignore_index = True)

        
        return df


    def crop_df(self,data:Dict[str,pd.DataFrame],start_time,end_time):
        result_df = {}
        data_length = self.encoder_length + self.decoder_length + 1
        for key,value in data.items():
            cur_df = value.loc[(value['real_time'] >= start_time) &
                                (value['real_time'] < end_time),:]
            result_df.update({key : cur_df})

        # if obs is missing
        for key,value in result_df.items():
            length_diff = value.shape[0] - data_length
            if abs(length_diff) > 2:
                raise Exception("more than one error")
            elif length_diff < 0:
                time_diff = value['real_time'].diff(periods = 1)
                diff_idx = np.where(time_diff > self.time_interval * 1000 )[0]
                if len(diff_idx) == 0: # if the last row is missing
                    diff_idx = [value.shape[0]] 
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

                    



        return result_df

    def convert_df2ts(self,data:Dict[str,pd.DataFrame]):
        result_data = []
        for key,value in data.itmes():
            cur_ts = torch.from_numpy(value.values, dtype = torch.float32)
            result_data.append(result_data)
        return result_data
    
    def pointnet_transform(self,data:List[pd.DataFrame],start_time,end_time):
        # cpu transform
        time_interval = self.time_interval # 1 second interval not 1 ms
        start_time *= 0.001
        end_time *= 0.001
        voxel_generator = Point2VoxelCPU1d(vsize_xyz=[time_interval],
                                           coors_range_xyz=[start_time,end_time],
                                           num_point_features = len(self.pairs)*3, # neads to be calculated later 
                                           max_num_voxels= int((end_time-start_time)/time_interval))

        voxels, coords, num_points = [],[],[]        
        for datum in data:
            time_col_loc = datum.columns.get_loc('Timestamp')
            datum = datum.to_numpy()
            datum = np.transpose
            tv_datum = tv.from_numpy(datum)
            voxels_tv, indices_tv, num_p_in_voxel = voxel_generator.point_to_voxel(tv_datum)
            voxels_tv, indices_tv, num_p_in_voxel = voxels_tv.numpy(), indices_tv.numpy(), num_p_in_voxel.numpy() 
            voxels.append(voxels_tv)
            coords.append(indices_tv)
            num_points.append(num_p_in_voxel)

        return [voxels, coords, num_points]
    
    def time_split(self,data:Dict[str,pd.DataFrame]):
        '''
        Time stamp should be always on the 0th column - set by select df
        '''
        result_data = dict()
        result_time = dict()
        for key,datum in data.items():
            time_stamp ,new_data = datum.loc[:,'real_time'],datum.loc[:,self.selected_cols] 
            result_data.update({key:new_data})
            result_time.update({key:time_stamp})

        return result_data,time_stamp

    def convert_np2ts(self, data:pd.DataFrame):
        new_data = torch.from_numpy(data.to_numpy().astype(np.float32))

        return new_data 


    def convert_np2ts_pointnet(self, data:List[List[np.array]]):
        voxels, coords, num_points = data
        voxel = np.concatenate(*voxels, axis = -1)
        coord = np.concatenate(*coords, axis = -1)
        num_point = np.concatenate(*num_points,axis =-1 )

        voxel_ts = torch.from_numpy(voxel)
        coord_ts = torch.from_numpy(coord)
        num_point_ts = torch.from_numpy(num_point)

        return [voxel_ts, coord_ts, num_point_ts]

    def select_columns(self,data:Dict[str,pd.DataFrame]):
        # make sure self.selected_header - time_stamp always on 0th col - 
        # later used by time_split
        result_df = {}

        for pair,datum in data.items():
            cur_data = datum.loc[:,self.selected_headers]
            result_df.update({pair : cur_data})

        return result_df
    
    def scaler(self,df_dict:Dict[str,pd.DataFrame]):

        selected_col = ['open','close','high','low','volume'] # this should be in init 
        result_df = {}
        for pair,datum in df_dict.items():
            scaler = StandardScaler()
            scaler.mean_ = [self.mean[pair][cur_col] for cur_col in selected_col]
            scaler.scale_ = [self.std[pair][cur_col] for cur_col in selected_col] 
            datum.loc[:,selected_col] = scaler.transform(datum.loc[:,selected_col].to_numpy())
            result_df.update({pair : datum})
        return result_df

    def diff_price(self,df_dict:Dict[str,pd.DataFrame]):
        selected_col = ['open','close','high','low','volume'] # this should be in init 
        result_df = {}
        for pair,datum in df_dict.items():
            ori_cols = list(datum.columns)
            cur_cols = [i for i in ori_cols if i not in selected_col]
            new_df = datum.loc[:,selected_col].diff(axis= 0,periods =1)
            new_df = pd.concat([new_df.iloc[1:,:],datum.loc[datum.index[1]:,cur_cols]],axis = 1)
            result_df.update({pair : new_df})

        return result_df

    def triple_barrier(self,data,barrier_width = 0.01):
        # consider encoder only product
        assert len(self.target_pair) == 1
        cur_data= data[self.target_pair[0]].loc[:,'open'].values
        #base_value = base_data[self.target_pair[0]]['open'] # this is series -> float 
        cur_data = cur_data.reshape(-1,1)
        # to-Do: change for enc-dec model
        horizon_value = cur_data[1:]
        base_value = cur_data[0].reshape(1,1)

        upper = np.where(horizon_value > base_value * (1 + barrier_width),1,0)
        lower = np.where(horizon_value < base_value * (1 - barrier_width),1,0)

        upper_a1 = np.argmax(upper)
        lower_a1 = np.argmax(lower)

        if upper_a1 > lower_a1:
            return torch.tensor([2], dtype = torch.float32)
        elif upper_a1 < lower_a1:
            return torch.tensor([0], dtype = torch.float32)
        else:
            return torch.tensor([1], dtype = torch.float32)
    
        
    def target_split(self,df_dict:Dict[str,pd.DataFrame]):
        # cov_result_df: [encoder_length,selected_head]
        # target_result_df: [decoder_length + 1,selected_head]
        encoder_length = self.encoder_length
        decoder_length = self.decoder_length

        cov_result_df = {}
        target_result_df = {}
        target_base_price= {}

        def diff_price(df_dict:Dict[str,pd.DataFrame]):
            selected_col = ['open','close','high','low','volume'] # this should be in init 
            result_df = {}
            for pair,datum in df_dict.items():
                ori_cols = list(datum.columns)
                cur_cols = [i for i in ori_cols if i not in selected_col]
                new_df = datum.loc[:,selected_col].diff(axis= 0,periods =1)
                new_df = pd.concat([new_df.iloc[1:,:],datum.loc[datum.index[1]:,cur_cols]],axis = 1)
                result_df.update({pair : new_df})

            return result_df

        for pair,datum in df_dict.items():
            new_df = datum.iloc[:-(decoder_length+1),:]
            cov_result_df.update({pair : new_df})

        # diff base price in encoder data
        cov_result_df = diff_price(cov_result_df)

        for pair,datum in df_dict.items():
            new_df = datum.iloc[-(decoder_length+2):,:]
            new_df_base = datum.iloc[-(decoder_length+2):,:]

            target_result_df.update({pair : new_df})
            target_base_price.update({pair : new_df_base})

        target_result_df= diff_price(target_result_df)
        #target_base_price = diff_price(target_base_price)
        
        return cov_result_df, target_result_df, target_base_price

    def cal_std(self,df_dict:Dict[str,pd.DataFrame]):
        self.mean = dict()
        self.std = dict()
        for key,value in df_dict.items():
            mean = dict()
            std = dict() 
            for col in value.columns:
                if col == 'real_time':
                    continue
                mean.update({col:value[col].mean()})
                std.update({col:value[col].std()})
            self.mean.update({key:mean})
            self.std.update({key:std})

        return df_dict

    def regular_concat(self,data:Dict[str,pd.DataFrame]):
        # make sure self.selected_header - time_stamp always on 0th col - 
        # later used by time_split
        
        df_list = []
        for pair,datum in data.items():
            cur_data = datum.loc[:,self.selected_cols]
            df_list.append(cur_data.reset_index(drop = True))
        result_df = pd.concat(df_list, axis = 1, ignore_index= True)

        return result_df


    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get sample for model

        Args:
            idx (int): index of prediction (between ``0`` and ``len(dataset) - 1``)

        Returns:
            Tuple[Dict[str, torch.Tensor], torch.Tensor]: x and y for model
        """
        try:
            start_time = self.index[idx]
            end_time = start_time + (self.encoder_length + self.decoder_length + 1) * 1000 * self.time_interval
            date_start = datetime.utcfromtimestamp(start_time//1000).strftime('%Y%m')
            date_before = (datetime.utcfromtimestamp(start_time//1000) - dt.timedelta(days=1)).strftime('%Y%m')
            date_end = datetime.utcfromtimestamp(end_time//1000).strftime('%Y%m')

            # load dataframe data for selected index
            if date_start == date_end:
                df_dict = self.load_dfs([date_start])
            elif date_start != date_end :
                df_dict = self.load_dfs([date_start, date_end])
            else:
                assert 1
                    #df_dict = self.load_dfs([date_before,date_end])
            


            # apply transformation here: TO-DO: make compose function 
            # current setting 
            #. 0 crop dataframe matching with the index
            #. 1 get point-pillar like formatting
            #. 1-1 split cov and target
            #. 2 get mid points add
            #. 3 transform the target function 
            #. 4 concat different pairs together. 
            
            df_dict = self.select_columns(df_dict)
            mean_data = self.diff_price(df_dict) 
            mean_data = self.cal_std(mean_data)
            df_dict = self.crop_df(df_dict,start_time,end_time)
            df_dict, target, target_base_price = self.target_split(df_dict) # also doing target generation - but for decoder input
            df_dict = self.scaler(df_dict)
            df_list, time_stamp = self.time_split(df_dict)
            #df_list,coords, num_points = self.pointnet_transform(df_dict,start_time,end_time)
            df_list = self.regular_concat(df_dict)
            #time_stamp = self.regular_concat(time_stamp)
            df_list = self.convert_np2ts(df_list) # later change to convert all items in the dict  - or selected 
            time_stamp = self.convert_np2ts(time_stamp)
            #coords = self.convert_np2ts(coords)
            #num_points = self.convert_np2ts(num_points)
            target = self.triple_barrier(target_base_price)

            assert df_list.shape[0] == self.encoder_length - 1

            # target - torch.Tensor - [B,1]
            # df_dict - torch.Tensor - [B,T,M,2*pair+1]
            return dict(x_data = df_list, 
                        y_data = target,
                        time_stamp = time_stamp,
                        ignore_flag = False 
                        #coords = coords,
                        #num_points = num_points,
                        )
        except:
            return dict(x_data = torch.randn((self.encoder_length-1,
                                            len(self.pairs) * len(self.selected_cols)),
                                            dtype = torch.float32), 
                        y_data = torch.randn((1), dtype = torch.float32),
                        time_stamp = torch.randn((self.encoder_length -1), dtype= torch.float32),
                        ignore_flag = True 
                        #coords = coords,
                        #num_points = num_points,
                        )



    def _collate_fn(
        self, batches: List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Collate function to combine items into mini-batch for dataloader.

        Args:
            batches (List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]): List of samples generated with
                :py:meth:`~__getitem__`.

        Returns:
            Tuple[Dict[str, torch.Tensor], Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]: minibatch
        """
        # collate function for dataloader
        # lengths

        x_data = torch.stack([batch['x_data'] for batch in batches if batch['ignore_flag'] is not True])
        # debug code
        if x_data.shape[0] != len(batches):
            print('more than two errors')
        time_stamp = torch.stack([batch['time_stamp'] for batch in batches if batch['ignore_flag'] is not True])
        #coords = torch.stack([batch['coords'] for batch in batches])
        #num_points = torch.stack([batch['num_points'] for batch in batches])
        y_data = torch.stack([batch['y_data'] for batch in batches if batch['ignore_flag'] is not True])

        return dict(x_data = x_data, 
                    y_data = y_data,
                    time_stamp = time_stamp,)
                    #coords = coords,
                    #num_points = num_points)

    def to_dataloader(
        self, train: bool = True, batch_sampler: Union[Sampler, str] = None, **kwargs
    ) -> DataLoader:
        """
        Get dataloader from dataset.

        The

        Args:
            train (bool, optional): if dataloader is used for training or prediction
                Will shuffle and drop last batch if True. Defaults to True.
            batch_size (int): batch size for training model. Defaults to 64.
            batch_sampler (Union[Sampler, str]): batch sampler or string. One of

                * "synchronized": ensure that samples in decoder are aligned in time. Does not support missing
                  values in dataset. This makes only sense if the underlying algorithm makes use of values aligned
                  in time.
                * PyTorch Sampler instance: any PyTorch sampler, e.g. the WeightedRandomSampler()
                * None: samples are taken randomly from times series.

            **kwargs: additional arguments to ``DataLoader()``

        Returns:
            DataLoader: dataloader that returns Tuple.
                First entry is ``x``, a dictionary of tensors with the entries (and shapes in brackets)

                * encoder_cat 
                * encoder_cont (batch_size x n_encoder_time_steps x n_features): float tensor of scaled continuous
                  variables for encoder


                Second entry is ``y``, a tuple of the form (``target``, `weight`)

                * target (batch_size x n_decoder_time_steps or list thereof with each entry for a different target):
                  unscaled (continuous) or encoded (categories) targets, list of tensors for multiple targets
                * weight (None or batch_size x n_decoder_time_steps): weight

        Example:

            Weight by samples for training:

            .. code-block:: python

                from torch.utils.data import WeightedRandomSampler

                # length of probabilties for sampler have to be equal to the length of the index
                probabilities = np.sqrt(1 + data.loc[dataset.index, "target"])
                sampler = WeightedRandomSampler(probabilities, len(probabilities))
                dataset.to_dataloader(train=True, sampler=sampler, shuffle=False)
        """
        default_kwargs = dict(
            shuffle=train,
            drop_last=train,
            collate_fn=self._collate_fn,
            batch_size=self.batch_size,
            batch_sampler=batch_sampler,
        )
        default_kwargs.update(kwargs)
        kwargs = default_kwargs

        return DataLoader(
            self,
            **kwargs,
        )

if __name__ == "__main__":
    data_path = '/media/ycc/Data2/aggTrades/'
    pairs = []
    target_pair = 'ETHUSDT'
    start_date = '20200101'
    end_date = '20220331'
    time_interval = 1
    encoder_length = 120
    decoder_length = 10
    val_cutoff = 0.8
    transform = []
    #predict_mode - updated in lt_model - determine train or val dataset
    data_type = 'aggTrade'
    max_obs = 100
    batch_size = 8

    cur_dataset = base_dataset(**kwargs)
    train_dataloader = cur_dataset.to_dataloader()
    for i,j in enumerate(train_dataloader):
        print(i)
        print(j)

        k = 100

