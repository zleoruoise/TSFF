import os
import pickle
from collections import Counter
from datetime import date

from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from tsff.algorithm_module.train_object import train_object
from tsff.algorithm_module import train,parse_args,load_cfg,load_model
from tsff.data_module.dm_montlhly import monthly_dataset
from tsff.data_module.utils.builder import build_dataset


def label_counter(train_dataset,val_dataset, overall = True):
    counter_list = []

    for dataset in [train_dataset,val_dataset]:
        dataset_counter = Counter()
        for idx, batch in enumerate(tqdm(dataset)):
            y = batch['y_data']
            cur_counter = Counter(y.squeeze().numpy().tolist())
            dataset_counter += cur_counter
        counter_list.append(dataset_counter)
    
    if overall:
        output_counter = dataset_counter[0] + dataset_counter[1]
    else:
        output_counter = dataset_counter
    
    return output_counter

def dataframe_analysis(output_counter,graph = False):
    cur_df = [pd.DataFrame(i,columns = ['low','middle','high']) for i in output_counter]

    for i,_cur_df in enumerate(cur_df):
        print(f'current_sequence: {i}')
        print(_cur_df.value_counts())
    if graph:
        pass # to-Do:

def mean_std_preproc(cfg_dataset): 
    scaler_dict = {}
    _today = date.today()

    dataset = build_dataset(cfg_dataset)

    for pair in dataset.memory_data:
        scaler = StandardScaler()
        scaler.fit(dataset.memory_data[pair].loc[:,dataset.selected_cols])
        scaler_dict[pair] = scaler 
    date1 = _today.strftime('%Y%m')
    
    with open(f'/home/ycc/TSFF/scaler_{date1}', 'wb') as f:
        pickle.dump(scaler_dict,f)

def impute_missing_value_pair(cfg_dataset,output_path = '/home/ycc/TSFF/proc_data'):
    dataset = build_dataset(cfg_dataset)

    # pair loop
    for pair in dataset.memory_data:
        cur_df = dataset.memory_data[pair]
        time_diff = cur_df['real_time'].diff(periods = 1)
        # if more than 5, then truncate 
        diff_idx_trunc = list(np.where(time_diff > dataset.time_interval * 1000 *5)[0])


        diff_idx_trunc.append(cur_df.shape[0])
        diff_idx_trunc.insert(0,0)

        trunc_arr_list = []
        # truncated df loop
        for i in range(len(diff_idx_trunc)-1):
            trunc_arr_list.append(cur_df.loc[diff_idx_trunc[i]:diff_idx_trunc[i+1]-1,:])

        imputed_trunc_df = []
        # truncated df loop
        for cur_data in trunc_arr_list:
            cur_time_diff = cur_data['real_time'].diff(periods = 1).to_numpy()
            diff_idx_impute = list(np.where(cur_time_diff > dataset.time_interval * 1000)[0])
            cur_arr = cur_data.to_numpy()

            imputed_list = []
            diff_idx_impute.append(cur_arr.shape[0])
            diff_idx_impute.insert(0,0)
            # imputation in each truncated df 
            for i in range(len(diff_idx_impute)-1):
                imputed_list.append(cur_arr[diff_idx_impute[i]:diff_idx_impute[i+1]])
                if (i+1) != (len(diff_idx_impute)-1):
                    num_impute = int(cur_time_diff[diff_idx_impute[i+1]]//(dataset.time_interval * 1000) - 1)
                    imputed_list.append(np.tile(cur_arr[diff_idx_impute[i+1]:diff_idx_impute[i+1]+1],(num_impute,1)))
            imputed_arr = np.vstack(imputed_list)
            imputed_cur_df = pd.DataFrame(imputed_arr,columns = dataset.selected_headers)
            # truncated df - imputed => impuated_cur_df 
            imputed_trunc_df.append(imputed_cur_df)
        
        pair_path = os.path.join(output_path,pair)
        os.makedirs(pair_path,exist_ok=True)
        for idx,_imputed_trunc_df in enumerate(imputed_trunc_df):
            cur_df_path = os.path.join(pair_path,f"{idx}.csv" )
            _imputed_trunc_df.to_csv(cur_df_path, index=False)

if __name__ == "__main__":
    args = parse_args()

    args.config = '/home/ycc/TSFF/tsff/algorithm_module/configs/autoformer/autoformer_960_RMSE_15_stat_diff.py'

    cfg = load_cfg(args)
    #cur_model = load_model(cfg)

    impute_missing_value_pair(cfg.dataset)
    #mean_std_preproc(cfg.dataset)

    #train_dataset,val_dataset = cur_model.prepare_data()
    #output_counter = label_counter(train_dataset, val_dataset)
    #dataframe_analysis(output_counter)
