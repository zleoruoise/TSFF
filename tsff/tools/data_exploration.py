import pickle
from collections import Counter
from datetime import date

from tqdm import tqdm
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
        pair_dict = {}
        #for col in dataset.selected_cols:
        scaler = StandardScaler()
        scaler.fit(dataset.memory_data[pair].loc[:,dataset.selected_cols])
        #    pair_dict[col] = scaler
        scaler_dict[pair] = scaler 
    date1 = _today.strftime('%Y%m')
    
    with open(f'/home/ycc/TSFF/scaler_{date1}', 'wb') as f:
        pickle.dump(scaler_dict,f)




    


if __name__ == "__main__":
    args = parse_args()

    args.config = '/home/ycc/TSFF/tsff/algorithm_module/configs/autoformer/autoformer_960_RMSE_15_stat_diff.py'

    cfg = load_cfg(args)
    #cur_model = load_model(cfg)

    mean_std_preproc(cfg.dataset)

    #train_dataset,val_dataset = cur_model.prepare_data()
    #output_counter = label_counter(train_dataset, val_dataset)
    #dataframe_analysis(output_counter)
