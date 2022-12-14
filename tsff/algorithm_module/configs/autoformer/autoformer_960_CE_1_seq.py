# overal config size 

# General variable 

from base64 import decode
from json import encoder
from symbol import encoding_decl


d_model = 128
n_head = 8
dropout = 0.1

# model input output size 
encoder_length = 960 
decoder_length = 0

hidden_continuous_size = 128 # used both in dataset and model
output_size = 10
log_interval = 10
custom_lr_scheduler = "CosineAnnealingWarmRestarts"

pairs = ['BTCUSDT','ETHUSDT',"BNBUSDT","XRPUSDT","ADAUSDT"]
target_pair = ['ETHUSDT']
selected_cols = ("open","close","high","low","volume")
lstm_layers_num = 3

# forecast type
forecast_type = 'reg'

# work_dir - should be full path
work_dir = '/home/ycc/TSFF/work_dir'
train_pipeline = [dict(type = 'select_columns',
                       selected_headers =  ("real_time","open",
                            "close","high","low","volume")),
                  dict(type = 'crop_df_seq',
                        encoder_length = encoder_length + 1,
                        decoder_length = decoder_length,
                        time_interval = 2),
                  dict(type = 'scaler',
                        pickle_path = '/home/ycc/TSFF/scaler_202210.pkl',
                        selected_cols = selected_cols), 
                  dict(type = 'target_split',
                        selected_cols = selected_cols,
                        encoder_length = encoder_length + 1,
                        decoder_length = decoder_length),
#                  dict(type = 'scaler',
#                        value_pickle = ''),
                  dict(type = 'time_split',
                        selected_cols = selected_cols),
                  dict(type = 'regular_concat',
                        selected_cols = selected_cols),
                  dict(type = 'convert_np2ts',
                        df_keys = ['x_data','time_stamp']),
                  dict(type = 'triple_barrier',
                        selected_cols = selected_cols,
                        target_pair = target_pair,
                        barrier_width = 0.001)
]

# model settings
dataset = dict(
    type = 'dm_seq',
    data_path = "/home/ycc/additional_life/binance-public-data/data/data/spot/monthly/klines",
    pairs = pairs,
    target_pair = ['ETHUSDT'],
    start_date = "20210101",
    end_date = "20220331",
    time_interval = 2,
    encoder_length = encoder_length + 1, # encoder 
    decoder_length = decoder_length,
    val_cutoff = 0.8,
    transforms = ['plain_target','stat_diff'],
    data_type = 'ohlcv',
    #barrier_width =  0.002,
    batch_size = 16,
    pipeline = train_pipeline,
    num_workers =22, 
    selected_cols = selected_cols,
    selected_headers = ('real_time', 'open', 'high','low','close','volume',
                                   'Close_time','Quote_asset_volumne','Number_of_trades',
                                   'Taker_buy_base_asset_volume',"Taker_buy_quote_asset_volume",'ignore'), 
    load_memory = dict(type = 'load_memory_seq',pairs =  pairs,
                        data_path = "/home/ycc/TSFF/proc_data/",
                        ),
    output_keys = ['target','x_data','time_stamp']
)

# trainer settings
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
# model settings
model = dict(
    type = 'enc_model',
    loss = 'CrossEntropy',
    #val_metrics = ['RMSE'],
    lr_scheduler = "CosineAnnealingWarmRestarts",
    optimizer = "adamw",
    learning_rate = 0.001,
    weight_decay = 0.01,

    # first embedding layers
    embedding = dict(
        type = "multitype_embedding",
        #cat embedding
        categorical_embedding_layer = None,        # cont_embdding
        continuous_embedding_layer = dict(
            type = 'continuous_embedding_layer',
            hidden_continuous_size = hidden_continuous_size,
            num_cov = len(pairs) * 5,
        ),
    ),

    # variable seelction
    variable_selection = None,
    # local encoder
    local_encoder = None, 
    # attn
    attention = dict(
        type = 'decomp_enc',
        # enrichment before attn
        local_enrichment_layer = None,
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
        decoder_attention = None,
        moving_avg = 25
    ),
    post_attention = dict(
        type = 'collapse_mlp_output',
        hidden_size = d_model,
        encoder_length = encoder_length,
        time_interval = 1,
        dropout = dropout,
        output_size = 3 # from dataset
    ),
    )