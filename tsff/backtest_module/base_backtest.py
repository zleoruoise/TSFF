import os
from re import I
import time
import glob
from pathlib import Path
import json

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from tsff.algorithm_module.model import lt_model
from tsff.decision_module.zero_one_cls_decision import zero_one_cls_decision
from tsff.decision_module.base_decision import base_decision_module
from tsff import parse_args, render_mpl_table 
from tsff.algorithm_module import test


class base_backtest:
    def __init__(self,weights_dir, backtest_list,preds_path = None,work_dirs = None, long_bal= 5000,long_asset= 0,online = False,*args,**kwargs):
        self.weights_path = weights_dir
        self.backtest_list = pd.read_csv(backtest_list)

        if preds_path is None:
            self.preds_path = "/home/ycc/additional_life/tsff/backtest/preds.hdf5"
        if work_dirs is None:
            self.work_dirs = "/home/ycc/additional_life/work_dirs/graphs"

        self.remove_list = []
        self.black_list = []


        self.decision_module = base_decision_module()
        #self.decision_module = zero_one_cls_decision()

        self.long_bal = long_bal 
        self.long_asset = long_asset
        self.short_bal = 0
        self.short_asset = 0



    def _remove_list(self):
        total_list = self.backtest_list
        cur_models_list = total_list.loc[total_list['remove_list'] == 1,:]
        model_name_list = cur_models_list['model_conf'].to_list()

        self.remove_list = model_name_list


                

    def _black_list(self):
        total_list = self.backtest_list
        cur_models_list = total_list.loc[total_list['black_list'] == 1,:]
        model_name_list = cur_models_list['model_conf'].to_list()

        self.black_list = model_name_list

    def prepare_preds(self,new_scratch = False):

        model_list = self.backtest_list['model_conf'].to_list()
        self._remove_list()
        self._black_list()
        if os.path.exists(self.preds_path):
            open_type = 'r+'
            if new_scratch:
                open_type = "w"
        else:
            open_type = 'w'

        for models in model_list:
            if models in self.black_list:
                continue
            print(models)
            model_path = Path(models)
            parent_name,model_name = model_path.parts[-2:]
            with h5py.File(self.preds_path, open_type) as f:
                if parent_name not in f.keys():
                    cur_group = f.create_group(parent_name)
                else:
                    cur_group = f[parent_name]
                if (models in self.remove_list)  or (model_name not in cur_group.keys()):
                    time_idx, predictions, observations = self._predict(models)
                    sub_group = cur_group.create_group(model_name)
                    sub_group.create_dataset('time_idx', data = np.array(time_idx))
                    sub_group.create_dataset('predictions', data = np.array(predictions))
                    sub_group.create_dataset('observations', data = np.array(observations))
                    #cur_group.create_dataset(model_name,data = np.stack((time_idx,predictions)))
                    if "observations" not in f.keys():
                        f.create_dataset("observations",data = observations)

    def _predict(self,models):
        args = parse_args()
        args.config = models

        model_parts = Path(models).parts

        weights_path_parents = os.path.join('/home/ycc/additional_life/work_dirs',model_parts[-2],model_parts[-1].split('.')[0])
        if os.path.exists(weights_path_parents):
            cur_path = os.path.join(weights_path_parents,"*val_loss*.ckpt")
            weights_lists = glob.glob(cur_path)
            weights_lists = sorted(weights_lists, key= lambda x: float(x.split('=')[-1].split('.ckpt')[0]))



        result = test(args,weights_lists[0],batch_size = 8 * 10)
        return result

    def stats_backtest(self):
        model_list = self.backtest_list['model_conf'].to_list()

        result_list = []
        for model_path in model_list:
            model_parents,model = Path(model_path).parts[-2:]
            model_wo_py = model.split('.')[0]
            if model_path in self.black_list:
                continue
            with h5py.File(self.preds_path,"r") as f:
                cur_obs = np.array(f[model_parents][model]['observations'])
                cur_preds = np.array(f[model_parents][model]['predictions'])
                cur_times = np.array(f[model_parents][model]['time_idx'])
                target_labels = ['sell','stay','buy']
                # to have same length as obs
                cur_preds = cur_preds[:cur_obs.shape[0],:]
                # prob to label prediction - change to raw mode
                #cur_preds = np.argmax(cur_preds, axis = 1)

                cur_report_txt = classification_report(cur_obs,cur_preds,target_names = target_labels)

                cur_json_path = os.path.join('/',*self.work_dirs.split('/')[:-1],model_parents,model_wo_py,'results.txt')
                with open(cur_json_path, 'w') as g:
                    g.write(cur_report_txt)



    def backtest(self):
        # prepare_preds is done 
        model_list = self.backtest_list['model_conf'].to_list()
        observations = None
        model_results = []

        #fig,ax = plt.subplots(figsize = (20,10))
        
        for i,model_path in enumerate(model_list):
            model_parents,model = Path(model_path).parts[-2:]
            model_wo_py = model.split('.')[0]
            if model_path in self.black_list:
                continue

            with h5py.File(self.preds_path,"r") as f:
                if model not in f[model_parents].keys():
                    continue
                # saved observations are not the price
                #observations = np.array(f[model_parents][model]['observations'])
                #observations = np.array(f['observations'])

                cur_data = np.array(f[model_parents][model]['predictions'])
                time_idx = np.array(f[model_parents][model]['time_idx'])
                observations = np.array(f[model_parents][model]['observations'])
                # previous models saved time idx and predictions in the same array after concat
                # but we separated it
                #cur_data = cur_data[:time_idx.shape[0],:]
                observations = observations[:,0]

                # truncate first observation
                observations = observations[1:]
                cur_data = cur_data[:-1,:]
                

                buy, sell = self.decision_module.decide(cur_data,observations)
                total_balance = self._backtest(buy,sell,observations)
                # overall plot
                #ax = self.draw_timeplot(cur_data[0],total_balance,
                #    model_parents+ model, ax)

                # individual_plot
                fig_sub,axes_sub = plt.subplots(figsize=(20,10))
                axes_sub = self.draw_timeplot(time_idx[:-1,0],observations * self.long_bal/observations[0],'obs',axes_sub)
                axes_sub = self.draw_timeplot(time_idx[:-1,0],total_balance,model_parents + model,axes_sub)
                axes_sub.legend()
                fig_sub.savefig(self.work_dirs + "/" + model_parents + "_" + model +".png")

                model_results.append([model,total_balance[-1]/observations[0], buy.sum(), sell.sum()])

                #if i == 0:
                #    ax = self.draw_timeplot(cur_data[0],observations * self.long_bal/observations[0] ,'obs', ax)

        #ax.legend()
        #fig.savefig(self.work_dirs + "/" + 'total_plot.png')
        table_df = pd.DataFrame(model_results, columns=['model','profit','buy','sell'])
        table_fig, _ = render_mpl_table(table_df)
        table_fig.savefig(self.work_dirs + "/" + "total_table.png")

        

    def draw_timeplot(self,times,datas,title,ax):
        ax.plot(times,datas, label = title)
        return ax

    

    def _backtest(self,buy,sell,obs):
        self.long_asset = 0
        self.long_bal = 5000
        total_values = []

        for cur_price, cur_buy, cur_sell in zip(obs,buy,sell):
            if cur_buy == 1: 
                self.long_asset += self.long_bal/cur_price * 0.999
                self.long_bal = 0
            elif cur_sell == 1 and self.long_asset > 0:
                self.long_bal += cur_price * self.long_asset * 0.999
                self.long_asset = 0
            total_values.append(self._cur_bal(cur_price))

        self.long_asset = 0
        self.long_bal = 5000
        return total_values

    def _backtest_short(self,buy,sell,obs):
        total_values = []
        for _buy,_sell,_obs in zip(buy,sell,obs):
            cur_rel_bet = self._relative_bet_size(_obs)
            if _sell > 0:
                tmp_asset = self.long_asset - cur_rel_bet/_obs * _sell
                if tmp_asset >= 0:
                    self.long_bal += _obs * (self.long_asset - tmp_asset)
                    self.long_asset = tmp_asset

                elif tmp_asset < 0:
                    self.long_bal += self.long_asset * _obs
                    self.long_asset = 0
                    self.short_asset -= tmp_asset # tmp_asset is negative 
                    self.short_bal -= tmp_asset * _obs
                    self.long_bal  -= tmp_asset * _obs # subtract short amount from 
                else:
                    raise AssertionError("tmp_asset is nan")

            elif _buy > 0:
                tmp_asset = self.short_asset - cur_rel_bet/_obs * _buy

                if tmp_asset >= 0: # realise from the earlier short 
                    cur_short_bal = self.short_bal
                    self.long_bal += self.short_bal - tmp_asset * _obs

                    tmp_asset_change = self.short_asset - tmp_asset
                    self.short_bal = tmp_asset_change * self.short_bal
                    self.short_asset = tmp_asset_change

                elif tmp_asset < 0:
                    self.long_bal += self.short_asset *(self.short_bal - _obs)
                    self.short_asset = 0
                    self.long_asset -= tmp_asset # tmp_asset is negative 
                    self.long_bal += tmp_asset * _obs
                else:
                    raise AssertionError("tmp_asset is nan")
            total_values.append(self._cur_bal(_obs))

        return total_values

            

    def _cur_bal(self,obs):
        return self.long_bal + self.long_asset * obs + (self.short_bal - obs) * self.short_asset

    def _relative_bet_size(self,obs):

        return self._cur_bal(obs)


    def old_backtest(self,path = None,**kwargs):

        # parsing whether to predict or to load 
        if path is None:
            cur_time = time.gmtime()
            time_str = time.strftime('%Y-%m-%d', cur_time)
            cur_path = "tmp_preds" + "_" + time_str
            os.makedirs(cur_path,exist_ok= True)
        else:
            cur_path = path

        time_idx_path = os.path.join(cur_path,"time_idx.npy")
        predictions_path = os.path.join(cur_path,"predictions.npy")
        observations_path = os.path.join(cur_path,"observations.npy")

        if path is None:
            time_idx, predictions, observations = self.model.eval_predict(new_df = None, batch_size = 32 * 2,**kwargs)
            # save for later usage 
            np.save(time_idx_path,time_idx)
            np.save(predictions_path,predictions)
            np.save(observations_path,observations)

        else:
            time_idx, predictions, observations = np.load(time_idx_path), np.load(predictions_path),np.load(observations_path)
        
        obs = observations[:-1,0] # [[1,2,3,4,5,6],[2,3,4,5,6,7]] - take 1,2,... 
        preds = predictions[1:,:] # [[1,2,3,4,5,6],[2,3,4,5,6,7]] 

        obs = obs[:-100]
        preds = preds[:-100]
        
        buy, sell = self.decision_module.decide(preds,obs)

        init_price = obs[0]
        init_cash = self.cash_balance
        for cur_price, cur_buy, cur_sell in zip(obs,buy,sell):
            if cur_buy == 1 and self.cash_balance > 0:
                self.asset_balance = self.cash_balance/cur_price  * 0.9996
                self.cash_balance = 0
            elif cur_sell == 1 and self.asset_balance > 0:
                self.cash_balance = cur_price * self.asset_balance * 0.9996
                self.asset_balance = 0

        #for cur_price, cur_buy, cur_sell in zip(obs,buy,sell):
        #    if cur_buy == 1: 
        #        self.asset_balance = self.cash_balance/cur_price * 0.999
        #        self.cash_balance = 0
        #    elif cur_sell == 1 and self.asset_balance > 0:
        #        self.cash_balance = cur_price * self.asset_balance * 0.999
        #        self.asset_balance = 0

        final_price = self.cash_balance + cur_price * self.asset_balance
        print(f"number of buying: {buy.sum()}")
        print(f"number of selling: {sell.sum()}")
        
        print("intial_account_value",init_cash)
        print("final_account_value",final_price)

        print("yield rate of trading:",final_price/init_cash)
        print("yield rate of coin itself:",cur_price/init_price)
        print("account if not sell:",cur_price/init_price*5000)
        print(len(obs)/ (60 * 24)) # min * hours = days




   # def online_data(self):

   #     # get data from data_module online 
   #     pass

if __name__ == "__main__":
    #cur_model = base_backtest(weights_path = "/workspace/best_model_path_old.txt")
    #cur_model.backtest(path="tmp_preds_2022-03-23",new_training_cutoff=1632316800000)
    #cur_model.backtest(path=None,training_cutoff=1633086244000,training_ends = 1638356644000)
    weights_dir = '/home/ycc/additional_life/work_dirs'
    backtest_list = '/home/ycc/additional_life/tsff/backtest/backtest_list.csv'
    cur_model = base_backtest(weights_dir,backtest_list)
    cur_model.prepare_preds()
    #cur_model.stats_backtest()
    cur_model.backtest()
