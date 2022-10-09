"""
Timeseries datasets.

Timeseries data is special and has to be processed and fed to algorithms in a special way. This module
defines a class that is able to handle a wide variety of timeseries data problems.
"""
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
class base_dataset(Dataset):
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
        max_obs: int = 1,
        batch_size: int = 8,
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

        # headers
        if data_type == 'ohlcv':
            self.headers =('real_time', 'open', 'high','low','close','volume',
            'Close_time','Quote_asset_volumne','Number_of_trades','Taker_buy_base_asset_volume',"Taker_buy_quote_asset_volume",'ignore') 
            self.selected_headers =  ("open","close","high","low","volume")
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
            df_dict.update(dict(pair = self.load_df(dates,pair)))

        return df_dict


    def load_df(self,dates,pair):
        assert len(dates) > 0
        df= None
        for date in dates:
            date = date + '.csv'
            data_path = os.path.join(self.data_path,pair,date)
            if df is None:
                df = pd.read_csv(data_path)
                df.columns = self.headers 
            df = pd.concat(df, axis = 1, ignore_index = True)

        
        return df


    def crop_df(self,data:Dict[str,pd.DataFrame],start_time,end_time):
        result_df = {}
        for key,value in data.items():
            cur_df = data.loc[(value['timestamp'] >= start_time) &
                                value['timestamp'] <= end_time]
            result_df.update(dict(key = cur_df))

        return result_df

    def convert_df2ts(self,data:Dict[str,pd.DataFrame]):
        result_data = []
        for key,value in data.itmes():
            cur_ts = torch.from_numpy(value.values)
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
    
    def time_split(self,data:List[np.array]):
        '''
        Time stamp should be always on the 0th column - set by select df
        '''
        result_data = []
        result_time = []
        for datum in data:
            time_stamp ,new_data = data[...,0],data[...,1:] 
            result_data.append(new_data)
            result_time.append(time_stamp)

        return result_data,time_stamp


    def convert_np2ts(self, data:List[List[np.array]]):
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
            result_df.update(dict(pair = cur_data))

        return result_df
    
    def scaler(self,df_dict:Dict[str,pd.DataFrame],mean,std,target):
        scaler = StandardScaler()
        scaler.mean_ = mean
        scaler.scale_ = std

        selected_col = [target]
        result_df = {}
        for pair,datum in df_dict.items():
            datum.loc[:,selected_col] = scaler.transfrom(datum.loc[:,selected_col])
            result_df.update(dict(pair = datum))
        return result_df

    def diff_price(self,df_dict:Dict[str,pd.DataFrame]):
        selected_col = ['Price']
        result_df = {}
        for pair,datum in df_dict.items():
            new_df = datum.loc[:,selected_col].diff(axis= 0,periods =1)
            new_df = pd.concat(new_df.loc[1:,:],datum.loc[:,-selected_col],ignore_index=True)
            result_df.update(dict(pair = new_df))

        return result_df

    def triple_barrier(self,data,barrier_width):
        # consider encoder only product
        cur_data= data[self.target_pair].loc[:,'Price'].values
        cur_data = cur_data.reshape(-1,1)
        # to-Do: change for enc-dec model
        horizon_value = cur_data[:-1]
        base_value = cur_data[0].reshape(1,1)

        upper = np.where(horizon_value > base_value * (1 + barrier_width),1,0)
        lower = np.where(horizon_value < base_value * (1 - barrier_width),1,0)

        upper_a1 = np.argmax(upper)
        lower_a1 = np.argmax(lower)

        if upper_a1 > lower_a1:
            return torch.tensor([2])
        elif upper_a1 < lower_a1:
            return torch.tensor([0])
        else:
            return torch.tensor([1])
    
        
    def target_split(self,df_dict:Dict[str,pd.DataFrame],target_transfrom):
        encoder_length = self.encoder_length
        decoder_length = self.decoder_length

        cov_result_df = {}
        target_result_df = {}

        for pair,datum in df_dict.items():
            new_df = datum.loc[:-(decoder_length+1),:]
            cov_result_df.update(dict(pair = new_df))

        for pair,datum in df_dict.items():
            new_df = datum.loc[-(decoder_length+1):,:]
            target_result_df.update(dict(pair = new_df))
        
        return cov_result_df, target_result_df

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get sample for model

        Args:
            idx (int): index of prediction (between ``0`` and ``len(dataset) - 1``)

        Returns:
            Tuple[Dict[str, torch.Tensor], torch.Tensor]: x and y for model
        """
        start_time = self.index[idx]
        end_time = start_time + (self.encoder_length + self.decoder_length + 1) * 1000 * self.time_interval
        date_start = datetime.utcfromtimestamp(start_time//1000).strftime('%Y%m%d')
        date_end = datetime.utcfromtimestamp(end_time//1000).strftime('%Y%m%d')

        # load dataframe data for selected index
        if date_start == date_end:
            df_dict = self.load_dfs([date_start])
        else:
            df_dict = self.load_dfs([date_start,date_end])

        # apply transformation here: TO-DO: make compose function 
        # current setting 
        #. 0 crop dataframe matching with the index
        #. 1 get point-pillar like formatting
        #. 1-1 split cov and target
        #. 2 get mid points add
        #. 3 transform the target function 
        #. 4 concat different pairs together. 
        
        df_dict = self.select_columns(df_dict)
        df_dict = self.crop_df(df_dict,start_time,end_time)
        df_dict, target = self.target_split(df_dict) # also doing target generation - but for decoder input
        df_dict = self.diff_price(df_dict) 
        df_dict = self.scaler(df_dict)
        df_list,coords, num_points = self.pointnet_transform(df_dict,start_time,end_time)
        df_list, time_stamp = self.time_split(df_dict)
        df_list = self.convert_np2ts(df_list) # later change to convert all items in the dict  - or selected 
        time_stamp = self.convert_np2ts(time_stamp)
        coords = self.convert_np2ts(coords)
        num_points = self.convert_np2ts(num_points)
        target = self.triple_barrier(target)

        # target - torch.Tensor - [B,1]
        # df_dict - torch.Tensor - [B,T,M,2*pair+1]
        return dict(x_data = df_list, 
                    y_data = target,
                    time_stamp = time_stamp,
                    coords = coords,
                    num_points = num_points)


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

        x_data = torch.tensor([batch['x_data'] for batch in batches])
        time_stamp = torch.tensor([batch['time_stamp'] for batch in batches])
        coords = torch.tensor([batch['coords'] for batch in batches])
        num_points = torch.tensor([batch['num_points'] for batch in batches])
        y_data = torch.tensor([batch['y_data'] for batch in batches])

        return dict(x_data = x_data, 
                    y_data = y_data,
                    time_stamp = time_stamp,
                    coords = coords,
                    num_points = num_points)

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

