import warnings

from tsff.utils.registry import build_model_from_cfg, build_model_from_cfg_overall,Registry



PIPELINES = Registry('Pipeline', build_func=build_model_from_cfg)
DATASETS = Registry('Datasets', build_func=build_model_from_cfg)

def build_dataset(cfg):
    """Build backbone."""
    return DATASETS.build(cfg)

def build_pipeline(cfg):
    """Build backbone."""
    return PIPELINES.build(cfg)
