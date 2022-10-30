from pathlib import Path
import json
import argparse
import copy
import os
import os.path as osp
import time
import warnings

import glob
from tensorboard import program

from tsff.algorithm_module.train_object import train_object
from ..utils import Config 



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
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
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
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def train(args):

    cfg = load_cfg(args)
    model = load_model(cfg)
    #
    model.prepare_learning()
    # save updated cfg in file 
    cur_cfg = model.model.cfg
    file_name = os.path.join(Path(cfg.work_dir), "experiment_config.json")
    with open(file_name,'w') as f:
        json.dump(cur_cfg, f,indent=4)

    #tensorboard_launch(cfg.work_dir)
    model.fit()


def resume_train(args):

    cfg = load_cfg(args)
    model = load_model(cfg)
    #
    model.prepare_learning()

    #tensorboard_launch(cfg.work_dir)
    model.resume_training()
    # save updated cfg in file 
    

def test(args,weight_path,batch_size = 16,**kwargs):
    cfg = load_cfg(args)
    cfg.dataset.batch_size = cfg.dataset.batch_size * 3
    cfg.dataset.stop_val_randomization = False 

    model = load_model(cfg)
    model.prepare_learning()
    model.load_weights(weight_path)

    time_idx, predictions, observations = model.eval_predict()

    return time_idx, predictions, observations


def load_cfg(args):
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.auto_resume = args.auto_resume

    # init distributed env first, since logger depends on the dist info.
    return cfg


def load_model(cfg):
    # cuz pytorch lightning combines opt, dataset and torch model into one large instance  
    model = train_object(**cfg)
    return model


def tensorboard_launch(dirs):
    parents_path = dirs + '/default'
    prev_len = len(glob.glob(parents_path + "/version_*"))
    cur_path = parents_path + f"/version_{prev_len}"
    tb = program.TensorBoard()
    tb.configure(argv = [None, '--logdir', cur_path, '--port','5678'])
    url = tb.launch()
    print(f'cur url: {url}')
    #


if __name__ == '__main__':
    args = parse_args()
    test(args)