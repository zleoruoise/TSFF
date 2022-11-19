import os
import shutil
from tsff.algorithm_module import train,resume_train
import pandas as pd
import numpy as np
import warnings
import argparse

import time 
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--options',
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def experiment_train(exp_path):
    # args from 
    args = parse_args()

    base_path = 'work_dirs'
    
    start_time = time.gmtime()
    start_time = time.strftime('%Y%m%d_%H',start_time)

    exp_name = os.path.join(Path(base_path))
    csv_name = os.path.join(Path(base_path),f"{start_time}_experiment_list.csv")

    os.makedirs(exp_name, exist_ok= True)

    cur_pd = pd.read_csv(exp_path)
    # save current experiment setting in work_dirs/
    cur_pd.to_csv(csv_name, index = False)

    resume_flag = False

    while True:
        # flag for cur loop
        resume_flag = False
        cur_pd = pd.read_csv(exp_path)
        if 1 not in cur_pd['train_flag'].values:
            break
        # train loop
        train_flag = cur_pd.index[cur_pd['train_flag'] == True].tolist()
        inv_train_flag = cur_pd.index[cur_pd['train_flag'] == False].tolist()

        # only for first loop
        no_train_pd = cur_pd.loc[inv_train_flag,:]
        no_train = no_train_pd.index[no_train_pd['train_start_time'].isna()].tolist()
        cur_pd.loc[no_train,"train_start_time"] = "Excluded"

        # general loop 
        cur_time = time.gmtime()
        cur_time = time.strftime('%Y%m%d_%H%M',cur_time)

        cur_idx = train_flag.pop(0)
        # save time in training file
        if isinstance(cur_pd.loc[cur_idx,'train_start_time'],str):
            resume_flag = True
        cur_pd.loc[cur_idx,'train_start_time'] = cur_time 

        ## set model conf and work_dir based on csv file
        args.config = cur_pd.loc[cur_idx,'model_conf']
        # save file save path 
        cur_work_dir = os.path.join(exp_name,args.config.split('/')[-2],args.config.split('/')[-1].split('.')[0])
        os.makedirs(cur_work_dir,exist_ok=True)
        args.work_dir = str(cur_work_dir)

        # model train start 
        if resume_flag is not True:
            train(args=args)
        else:
            resume_train(args=args)

        cur_pd.loc[cur_idx,'train_flag'] = 0

        # save in experiment_control
        cur_pd.to_csv(exp_path, index=False)

        # flag for next loop
        








if __name__ == "__main__":
    #exp_path = "/workspace/tsff/algorithm_module/experiment_control/experiment_list.csv"
    exp_path = "/home/ycc/additional_life/tsff/algorithm_module/experiment_control/experiment_list.csv"
    #exp_path = "/workspace/work_dirs/20220328_03/experiment.csv"
    experiment_train(exp_path)
