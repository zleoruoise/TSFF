
import numpy as np
from copy import copy
from typing import Optional, Any, Union, Callable, Dict, List
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts, CosineAnnealingLR

import torch
from torch import Tensor


from pytorch_forecasting.data import TimeSeriesDataSet

from torch import functional as F

from pytorch_forecasting.utils import autocorrelation, create_mask, detach, integer_histogram, padded_stack, to_list


from pytorch_forecasting.models.base_model import BaseModelWithCovariates
from pytorch_forecasting import QuantileLoss,MultiHorizonMetric, SMAPE,RMSE, MAE, MAPE,SmoothL1Loss,FocalLoss,CrossEntropy
# old
from torchmetrics import Metric as LightningMetric

from torch import nn
from ..builder import * 
#from tsff.algorithm_module.models.builder import *

import pytorch_lightning as pl
from pytorch_forecasting.metrics import (
    MAE,
    MASE,
    SMAPE,
    DistributionLoss,
    Metric,
    MultiHorizonMetric,
    MultiLoss,
    QuantileLoss,
)
from pytorch_forecasting.optim import Ranger

# custom 



@MODEL.register_module()
class enc_model(pl.LightningModule):
    """
    Methods:
        init: --
        forward:
        training_step:
        validation_step:
        test_step:
        predict_step: 
        configure_optimizer:
    """

    def __init__(self,
                embedding = None,
                variable_selection = None,
                local_encoder = None,
                attention = None,
                post_attention = None,

                loss = 'RMSE',
                val_metrics: List[str] = [],

                optimizer: str = None,
                learning_rate: float = None,
                optimizer_params: str = None,
                lr_scheduler: str = None,
                weight_decay: float = None,
                **kwargs,
                 ) -> None:

        super().__init__()
        self.save_hyperparameters()
        # store loss function separately as it is a module

        # why do this here ? - now this is included in the cfg.model.optimizer 
        #self.hparams.custom_lr_scheduler = cfg.custom_lr_scheduler
        #self.hparams.optimizer = cfg.custom_optimizer


        # model construction
        if embedding is None:
            embedding = dict(type='empty_layer')
        if variable_selection is None:
            variable_selection = dict(type='empty_layer')
        if local_encoder is None:
            local_encoder = dict(type='empty_layer')
        if attention is None:
            attention = dict(type='empty_layer')
        if post_attention is None:
            post_attention = dict(type='empty_layer')

        self.embedding = build_embedding(embedding)
        self.variable_selection = build_variable_selection(variable_selection)
        self.local_encoder = build_local_encoder(local_encoder)
        self.attention = build_attention(attention)
        self.post_attention = build_post_attention(post_attention)

        # loss construction
        self.loss = build_loss_function(loss)

        self.val_metrics = {}
        for cur_metric in val_metrics:
            cur_metric_fn = build_loss_function(cur_metric)
            self.val_metrics.update(dict(cur_metric = cur_metric_fn))

            self.loss = build_loss_function(loss)

        # optimizer
        self.optimizer_params = optimizer_params
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.weight_decay = weight_decay

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        input dimensions: n_samples x time x variables
        """

        # move input_embedding into one module
        ## this include - embedding cat and cont, variable selection, 
        ### also group (embedding, prescaler before single variable grn in variable selection) 
        ### (context vector, cell, state, post enrichment, in static encoder)
        # input settings


        # input embedding - both cont and cat
        if self.embedding is not None:
            x = self.embedding(x)
        if self.variable_selection is not None:
            x = self.variable_selection(x)
        if self.local_encoder is not None:
            x = self.local_encoder(x)

        # mask = self.get_attention_mask(encoder_lengths,timesteps - self.hparams.max_encoder_length) - only consider encoder only
        if self.attention is not None:
            x = self.attention(x)
        # post-processing
        x = self.post_attention(x)
        
        return x['output_data']

    def training_step(self,batch,batch_idx):
        x = {k:v for k,v in batch.items() if k != 'y_data'}
        y = batch['y_data']
        y_hat = self(x)
        loss = self.loss(y_hat,y)
        self.log("loss", loss)
        return loss
        
    def validation_step(self,batch,batch_idx):
        x = {k:v for k,v in batch.items() if k != 'y_data'}
        y = batch['y_data']
        y_hat = self(x)
        loss = self.loss(y_hat,y)
        val_metrics = {key:cur_metric(y_hat,y) for key,cur_metric in self.val_metrics}

        self.log("val_loss", loss)
        self.log_dict(val_metrics)

    def test_step(self,batch,batch_idx):
        x,y = batch
        y_hat = self(x)
        loss = self.loss(y_hat,y)
        val_metrics = {key:cur_metric(y_hat,y) for key,cur_metric in self.val_metrics}

        self.log("loss", loss)
        self.log_dict(val_metrics)

    def prediction_step(self,batch,batch_idx):
        x,y = batch
        y_hat = self(x)
        return y_hat

    def configure_optimizers(self):
        """
        Configure optimizers.
        Args:
            self.optimizer_params
            self.learning_rate
            self.optimizer
            self.lr_scheduler
        Returns:
            Tuple[List]: first entry is list of optimizers and second is list of schedulers
        """
        # either set a schedule of lrs or find it dynamically
        optimizer_params = self.optimizer_params
        # set optimizer
        lrs = self.learning_rate
        lr = lrs
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=self.weight_decay)
        elif self.optimizer == "ranger":
            optimizer = Ranger(self.parameters(), lr=lr, weight_decay=self.hparams.weight_decay)
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=lr, weight_decay=self.weight_decay, **optimizer_params
            )
        else:
            raise ValueError(f"Optimizer of self.hparams.optimizer={self.optimizer} unknown")

        # set scheduler
        if self.lr_scheduler == "CosineAnnealingWarmRestarts":
            scheduler_config = {
                "scheduler":CosineAnnealingWarmRestarts(
                    optimizer,
                    30,
                    T_mult=2
                ) ,
                "monitor": "val_loss",  # Default: val_loss
                "interval": "epoch",
                "frequency": 1,
                "strict": False,
            }
        elif isinstance(lr, (list, tuple)):  # change for each epoch
            # normalize lrs
            lrs = np.array(lrs) / lrs[0]
            scheduler_config = {
                "scheduler": LambdaLR(optimizer, lambda epoch: lrs[min(epoch, len(lrs) - 1)]),
                "interval": "epoch",
                "frequency": 1,
                "strict": False,
            }
        else:  # find schedule based on validation loss
            scheduler_config = {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=0.2,
                    patience=self.hparams.reduce_on_plateau_patience,
                    cooldown=self.hparams.reduce_on_plateau_patience,
                    min_lr=self.hparams.reduce_on_plateau_min_lr,
                ),
                "monitor": "val_loss",  # Default: val_loss
                "interval": "epoch",
                "frequency": 1,
                "strict": False,
            }

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

# later make it as registery
def build_loss_function(loss):
    return eval(loss)()



