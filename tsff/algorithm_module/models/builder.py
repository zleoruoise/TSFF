import warnings

from tsff.utils.registry import build_model_from_cfg, build_model_from_cfg_overall,Registry



_MODEL = Registry('model', build_func=build_model_from_cfg)
# old_model is legacy and overall is also legacy
OLD_MODEL = Registry('total_model', build_func=build_model_from_cfg_overall)
MODEL = Registry('total_model', build_func=build_model_from_cfg)
MODELS = Registry('models', parent=_MODEL)

EMBEDDING = Registry('embedding', build_func=build_model_from_cfg)
EMBEDDING_LAYERS = Registry('embedding_layer', build_func=build_model_from_cfg)
VARIABLE_SELECTION= Registry('variable_selection', build_func=build_model_from_cfg)
VARIABLE_SELECTION_LAYER = Registry('variable_selection_layer', build_func=build_model_from_cfg)
STATIC_ENCODER= Registry('static_encoder', build_func=build_model_from_cfg)
LOCAL_ENCODER = Registry('local_encoder', build_func=build_model_from_cfg)
LOCAL_ENCODER_LAYER = Registry('local_encoder_layer', build_func=build_model_from_cfg)
ENRICHMENT = Registry('enrichment', build_func=build_model_from_cfg)
ATTENTION = Registry('attention', build_func=build_model_from_cfg)
ATTENTION_LAYERS = Registry('attention_layers', build_func=build_model_from_cfg)
POST_ATTENTION = Registry('post_attention', build_func=build_model_from_cfg)

def build_model(cfg):
    """Build backbone."""
    return MODEL.build(cfg)

def build_embedding(cfg):
    """Build backbone."""
    return EMBEDDING.build(cfg)

def build_embedding_layers(cfg):
    """Build backbone."""
    return EMBEDDING_LAYERS.build(cfg)

def build_variable_selection(cfg):
    """Build neck."""
    return VARIABLE_SELECTION.build(cfg)

def build_variable_selection_layer(cfg):
    """Build neck."""
    return VARIABLE_SELECTION_LAYER.build(cfg)


def build_local_encoder(cfg):
    """Build roi extractor."""
    return LOCAL_ENCODER.build(cfg)

def build_static_encoder(cfg):
    """Build shared head."""
    return STATIC_ENCODER.build(cfg)

def build_local_encoder(cfg):
    """Build shared head."""
    return LOCAL_ENCODER.build(cfg)

def build_local_encoder_layer(cfg):
    """Build head."""
    return LOCAL_ENCODER_LAYER.build(cfg)

def build_enrichment(cfg):
    """Build head."""
    return ENRICHMENT.build(cfg)

def build_attention(cfg):
    """Build head."""
    return ATTENTION.build(cfg)

def build_attention_layer(cfg):
    """Build head."""
    return ATTENTION_LAYERS.build(cfg)

def build_post_attention(cfg):
    """Build head."""
    return POST_ATTENTION.build(cfg)
