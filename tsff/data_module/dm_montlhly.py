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
from sympy import O

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

from spconv.utils import Point2VoxelCPU1d
from cumm import tensorview as tv

from tsff.data_module.utils.builder import build_pipeline,DATASETS
from tsff.data_module.pipelines.compose  import Compose

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
        pipeline: List[Dict[str,str]] = [],
        num_workers = 4,
        selected_cols = [],
        selected_headers = [],
        load_memory = None,
        output_keys = []

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

        # 
        self.selected_cols = selected_cols
        self.selected_headers = selected_headers 

        self.num_workers = num_workers
        self.pipeline = Compose(pipeline)
        self.output_keys = output_keys

        self.memory_data = None
        if load_memory is not None:
            load_memory.update(start_time = self.start_date)
            load_memory.update(end_time = self.end_date)
            load_memory.update(headers = self.selected_headers)
            load_data_func = build_pipeline(load_memory)
            self.memory_data = load_data_func()
            print('memory_data_debug')
            


        # headers
        #if data_type == 'ohlcv':
        #    self.headers =('real_time', 'open', 'high','low','close','volume',
        #    'Close_time','Quote_asset_volumne','Number_of_trades','Taker_buy_base_asset_volume',"Taker_buy_quote_asset_volume",'ignore') 
        #    self.selected_headers =  ("real_time","open","close","high","low","volume")
        #    self.selected_cols = ("open","close","high","low","volume")
        #elif data_type == 'aggTrade':
        #    # change later 
        #    self.headers = ['Agg_tradeId','Price','Quantity','First_tradeID','Last_tradeID','Timestamp','maker','bestPrice'] # copy from github
        #    self.selected_headers= ['Timestamp', 'Price', 'Quantity']
        #else:
        #    AssertionError('data_type not implemented')

        ## filter data

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
        start_time = time.mktime(start_time.timetuple()) + GLOBAL_TIMEDIFF
        end_time = datetime.strptime(self.end_date, "%Y%m%d") + dt.timedelta(days=1)
        end_time = time.mktime(end_time.timetuple()) + GLOBAL_TIMEDIFF

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
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get sample for model

        Args:
            idx (int): index of prediction (between ``0`` and ``len(dataset) - 1``)

        Returns:
            Tuple[Dict[str, torch.Tensor], torch.Tensor]: x and y for model
        """
        #try:
        start_time = self.index[idx]
        data = {'start_time':start_time}

        if self.memory_data is not None:
            data.update(dict(x_data = self.memory_data))

        data = self.pipeline(data)
        data['ignore_flag'] = False
        #if data['x_data'].shape[0] != self.encoder_length -1:
        #    raise Exception("more than two errors")
        #if data['time_stamp'].shape[0] != self.encoder_length -1:
        #    raise Exception("more than two errors")
        #if data['target'].shape[0] != 1:
        #    raise Exception("more than two errors")
        return data
        #except:
        #    return dict(x_data = torch.zeros((self.encoder_length-1,
        #                                    len(self.pairs) * 5),
        #                                    dtype = torch.float32), 
        #                target = torch.zeros((1), dtype = torch.float32),
        #                time_stamp = torch.zeros((self.encoder_length -1), dtype= torch.float32),
        #                ignore_flag = True 
        #                #coords = coords,
        #                #num_points = num_points,
        #                )



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
        output_dict = {}

        for key in self.output_keys:
            output_dict[key] = torch.stack([batch[key] for batch in batches])
        
        return output_dict
                
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
            num_workers = self.num_workers
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


