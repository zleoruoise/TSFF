# overal config size 
from tsff.algorithm_module.models.attention.attention_layer.informer_attn import ProbAttention

pairs = ['BTCUSDT','ETHUSDT',"BNBUSDT","XRPUSDT","ADAUSDT"]
selected_heads = ['Quantity','Price']

encoder_length = 960

d_model = 128
n_head = 8
dropout = 0.1


# model input output size 
max_encoder_length = 960 
max_prediction_length = 15

hidden_continuous_size = 16# used both in dataset and model
output_size = 10
log_interval = 10
custom_lr_scheduler = "CosineAnnealingWarmRestarts"


# model settings
dataset = dict(
    target_list = ['ADAUSDT','ALGOUSDT','ATOMUSDT','AVAXUSDT','BNBUSDT','BTCUSDT','DOGEUSDT','DOTUSDT','ETHUSDT','LTCUSDT'
    ,"LUNAUSDT",'SOLUSDT',"TRXUSDT","XLMUSDT",'XRPUSDT'],
    target = "ETHUSDT",
    transfrom = ['crop_df','convert_df2ts','pointnet_transform'],
    max_prediction_length = max_prediction_length,
    max_encoder_length = max_encoder_length,
    static_categoricals = [],
    static_real = [],
    time_varying_known_categorical = [],
    time_varying_known_real = ['time_idx'],
    time_varying_unknown_real =  [],
    time_varying_unknown_categorical = [],
    data_path = "./binance-public-data/data/data/spot/monthly/klines",
    transform = ['plain_target','stat_diff'],
    target_normalizer = None,
    barrier_width =  0.002,
    val_ratio = 0.7,
    training_cutoff = 1633014000000, # 211001-000000
    batch_size = 16,
    frac_ratio = 0.7,
    frac_window = 20,
    adf_test = False, 
    # new configure
    pairs = ['BTCUSDT','ETHUSDT',"BNBUSDT","XRPUSDT","ADAUSDT"],
    target_pair = ['ETHUSDT'],

    
)

trainer = dict(
    stop_patience = 5,
    max_epochs = 5000,
    gpus=1,
    weights_summary="top",
    gradient_clip_val=0.1,
    limit_train_batches=0.02,
    check_val_every_n_epoch= 5,
    fast_dev_run=False,  

)

lstm_layers_num = 3

model = dict(
    type = 'TFTransformer',
    loss = 'RMSE',

    lr_scheduler = "CosineAnnealingWarmRestarts",
    optimizer = "adamw",
    learning_rate = 0.001,
    weight_decay = 0.01,

    # first embedding layers
    embedding = dict(
        type = "base_embedding",
        #cat embedding
        value_embedding = dict(
            type = 'pillar_vfe',
            d_model = d_model, # should be loaded in from_dataset
            num_point_features= len(pairs) * len(selected_heads),
            use_norm = True,
            use_relative_distance = True 
        ),
        positional_embedding = dict(
            type = 'pillar_positional',
            d_model = d_model, # should be loaded in from_dataset
            max_len = 5000 if encoder_length < 5000 else encoder_length * 2
        ),
        temporal_embedding = dict(
            type = 'pillar_temporal',
            d_model = d_model, # should be loaded in from_dataset
        ),
    ),

    # variable seelction = None
    #local encoder = None
    # attn
    attention = dict(
        type = 'decomp_enc',
        # encoder 
        encoder_attention = dict(
            type = 'autoformer_encoder',
            layer_type = dict(
                type = 'autoformer_encoder_layer',
                sa_layer_type = dict(
                    type = 'AttentionLayer',
                    attn_type = dict(
                        type = "AutoCorrelation"))),
            hidden_size = d_model,
            dim_ff = d_model * 4,
            nhead = n_head,
            dropout = dropout,
            activation = 'gelu',
            layer_num = 3,
        ),
        moving_avg = 25
    ),
    post_attention = dict(
        type = 'single_pred',
        d_model = d_model,
        hidden_size = d_model*2,
        num_classes = 3,
        time_steps = encoder_length,
        num_layers = 2 # from dataset
    ),


    )