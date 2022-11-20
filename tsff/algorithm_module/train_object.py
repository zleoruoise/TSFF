# deep learning libraries
import os
from time import time
import glob
from matplotlib.transforms import Transform
import torch
from typing import List,Dict,Tuple,Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm

# conda
import numpy as np
import pandas as pd
from tsff.algorithm_module.models.builder import build_model
from tsff.data_module.utils.builder import build_dataset
from tsff.utils.utils import move_to_device

# custom


from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.migration import pl_legacy_patch



class train_object:
    def __init__(self,test_code_mode = False,**kwargs):
        """
        set params that are required to build dataset. 
        Args:
            max_prediction_length: 
            max_encoder_length: time length of the input
            target_list: target currency but in list - currency pairs that are meant to be predicted
            data_path: csv file path
        Returns:
            None
        """
        # set configs for each part
        self.forecast_type = kwargs['forecast_type']
        self.work_dir = kwargs['work_dir']
        self.dataset = kwargs['dataset'] 
        self.trainer = kwargs['trainer']
        self.cfg_model = kwargs['model']

        # this should be in dataset parsing - but we do in init
    
    def prepare_data(self,**kwargs):
        """
        wrapping all data loading and transformation
        Args:
            **kwargs
        Returns:
            train_dataloader: train dataset dataloader 
            val_dataloader: validation dataset dataloader 
        """

        # train_dataset 
        self.dataset.update(dict(predict_mode = True))
        train_dataset = build_dataset(self.dataset)
        # val_dataset
        self.dataset.update(dict(predict_mode = False))
        val_dataset = build_dataset(self.dataset)

        # dataset -> dataloader 

        train_dataloader = train_dataset.to_dataloader()
        val_dataloader = val_dataset.to_dataloader(train = False)

        return train_dataloader, val_dataloader

    def prepare_learning(self,**kwargs):
        """
        prepare dataset, model, optimizer, logger. Set everything ready for the fitting
        
        """

        self.train_dataloader, self.val_dataloader = self.prepare_data(**kwargs)
        print('data preparation done')
        self.trainer = self.prepare_trainer(**kwargs)
        print('data preparation done')
        self.model = self.prepare_model(**kwargs)
        #self.model.to("cuda")
    
    def fit(self):
        self.trainer.fit(self.model, 
            train_dataloader = self.train_dataloader, 
            val_dataloaders = self.val_dataloader)
        with open('./best_model_path', "w") as f:
            f.write(str(self.trainer.checkpoint_callback.best_model_path))


        
    def prepare_trainer(self,**kwargs):

        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=self.trainer['stop_patience'], verbose=False, mode="min")
        lr_logger = LearningRateMonitor()  # log the learning rate
        logger = TensorBoardLogger(self.work_dir)  # logging results to a tensorboard
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss", 
            dirpath= self.work_dir,
            filename="{epoch:03d}-{val_loss:.5f}",
            save_top_k=2,
            mode="min",
            save_last=True
)

        trainer = pl.Trainer(
                max_epochs=self.trainer['max_epochs'],
                gpus=self.trainer['gpus'],
                weights_summary=self.trainer['weights_summary'],
                gradient_clip_val=self.trainer['gradient_clip_val'],
                limit_train_batches=self.trainer['limit_train_batches'],
                limit_val_batches= 0.1,
                check_val_every_n_epoch=self.trainer['check_val_every_n_epoch'],
                fast_dev_run=self.trainer['fast_dev_run'],
                callbacks=[lr_logger, early_stop_callback,checkpoint_callback],
                logger=logger,
#                precision=16, accelerator="gpu", devices=1,
        )
        return trainer
    
    def prepare_model(self,**kwargs):
        # later make it call builder from model.__init__ registory

        model= build_model(self.cfg_model)
        return model
    
    def load_weights(self,path = None):
        # later this should be loaded from cfg files
        if path:
            self.best_weights = path
        if self.best_weights:
            #with open(self.best_weights, "r") as f:
            #    best_ckpt = f.readline()
                # to-Do: make selection of model possible
            best_ckpt = self.best_weights 
            with pl_legacy_patch():
                checkpoint = pl_load(best_ckpt, map_location="cuda")

            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.to("cuda")
            self.model.eval()

    def resume_training(self):
        cur_path = self.work_dir  + "/last.ckpt"
        if os.path.exists(cur_path):
            cur_ckpt = cur_path
        else:
            cur_path = self.work_dir  + "/*ckpt"
            cur_ckpt = glob.glob(cur_path)[-1]

        self.trainer.fit(self.model, 
            train_dataloader = self.train_dataloader, 
            val_dataloaders = self.val_dataloader,
            ckpt_path = cur_ckpt)



    
    def predict(self):
        # to-DO: implement load weights from ckpt
        # new_data should be implemented from real_time data obj
        # because we want to know the date and time, not just time idx, we use
        # raw dataframe as our input, and let self.model to handle the transformation
        # online dataset should also pass dataframe  
        #new_preds = self.model.predict(new_data, mode = "prediction")
        new_preds =[]
        for new_data in tqdm(self.val_dataloader):
            new_data = move_to_device(new_data,self.model.device)
            new_pred = self.model.prediction_step(new_data)
            new_pred = new_pred.detach()
            new_pred = move_to_device(new_pred,'cpu')
            new_preds.append(new_pred)
        return new_preds

    def eval_predict(self,new_df=None,raw_prediction = True,**kwargs):

        if isinstance(new_df,pd.DataFrame):
            raise ValueError("need to implement to set the start time")
        elif new_df is None:
            if 'training_cutoff' in kwargs:
                _,_,new_df = self.prepare_data(**kwargs)
            else:
                new_df = self.val_dataloader
        #raw_observations =  [[x['decoder_time_idx'],y[0]] for x,y in iter(new_df)]
        #for x,y in self.val_dataloader:
        #    x = x
        #    y = y
        #raw_observations =  [y[0] for x,y in iter(new_df)]



        time_idx = []
        observations = []
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for x,y in tqdm(new_df):
                _time_idx,obs, = x['decoder_time_idx'], y[0]
                x = move_to_device(x, self.model.device)
                pred = self.model(x)
                pred = move_to_device(pred[0], torch.device('cpu'))

                # stacking the data
                time_idx.append(_time_idx)
                observations.append(obs)
                predictions.append(pred)

        # transform to torch 
        time_idx = torch.cat(time_idx[:-1], axis = 0)
        # to-Do: change to incoperate with time idx
        observations = torch.cat(observations[:-1], axis = 0)
        predictions = torch.cat(predictions[:-1],axis = 0)
        # prediction 


        # return original scaled data or not 
        # check time index later 
        #predictions = predictions[:observations.shape[0],:]
        if predictions.dim() == 1:
            predictions = predictions.reshape(-1,1)
        
        if raw_prediction is True:
            #time_idx, predictions = self.reverse_transform(time_idx.cpu().numpy(),predictions.cpu().numpy()), 
            test1 = self.reverse_transform(time_idx.cpu().numpy(),predictions.cpu().numpy(),observations.cpu().numpy()), 
            # time_idx, preds, obs
            return test1[0][0],test1[0][1],test1[0][2]

            return time_idx, predictions, observations.cpu().numpy()

        return time_idx.cpu().numpy(),predictions.cpu().numpy(),observations.cpu().numpy()

    def reverse_transform(self,time_idx,predictions,obs):
        time_idx, predictions, obs = np.squeeze(time_idx), np.squeeze(predictions), np.squeeze(obs)

        if self.forecast_type == 'reg':
            original_pred = np.zeros_like(predictions)
            original_obs = np.zeros_like(predictions)
            for i in range(predictions.shape[1]):
                original_pred[:,i] = self.target_df.loc[time_idx[:,i]-self.max_prediction_length,self.target_col].values + predictions[:,i]
                original_obs[:,i] = self.target_df.loc[time_idx[:,i]-self.max_prediction_length,self.target_col].values + obs[:,i]
                #original_pred[:,i] =  predictions[:,i]
                #original_obs[:,i] =  obs[:,i]
        elif self.forecast_type == "cls":
            original_pred = predictions
            original_obs = obs 

        else:
            raise AssertionError('need to specify prediction task type')


        time_idx = time_idx * 60000 + self.start_time
        
        return time_idx,original_pred, original_obs

def move_to_device(
    x: Union[
        Dict[str, Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]],
        torch.Tensor,
        List[torch.Tensor],
        Tuple[torch.Tensor],
    ],
    device: Union[str, torch.DeviceObjType],
) -> Union[
    Dict[str, Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]],
    torch.Tensor,
    List[torch.Tensor],
    Tuple[torch.Tensor],
]:
    """
    Move object to device.
    Args:
        x (dictionary of list of tensors): object (e.g. dictionary) of tensors to move to device
        device (Union[str, torch.DeviceObjType]): device, e.g. "cpu"
    Returns:
        x on targeted device
    """
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(x, dict):
        for name in x.keys():
            x[name] = move_to_device(x[name], device=device)
    elif isinstance(x, torch.Tensor) and x.device != device:
        x = x.to(device)
    elif isinstance(x, (list, tuple)) and x[0].device != device:
        x = [move_to_device(xi, device=device) for xi in x]
    return x


if __name__ == "__main__":
    from  tsff.tools.train import train,parse_args,load_cfg,load_model

    args = parse_args()
    args.config = '/home/ycc/TSFF/tsff/algorithm_module/configs/pointformer/pointformer_960_CE_1.py'
    cfg = load_cfg(args)

    cur_model = load_model(cfg)

    cur_model.prepare_learning(batch_size = 16)
    cur_model.fit()
    #cur_model.load_weights("best_model_path.txt")
    #time_idx,predictions, observations = cur_model.eval_predict(new_df=None,new_training_cutoff=1638316800000,batch_size = 32*2)
    #from tsff.data_module.graph import evaluation_graph
    #evaluation_graph(time_idx,predictions,observations)

        

