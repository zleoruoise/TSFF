# overal config size 

# General variable 

from base64 import decode
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

pairs = ['BTCUSDT','ETHUSDT']
target_pair = ['ETHUSDT']
selected_cols = ('Quantity','Price')
lstm_layers_num = 3

# forecast type
forecast_type = 'reg'

# work_dir - should be full path
work_dir = '/home/ycc/TSFF/work_dir'
train_pipeline = [dict(type = 'set_time', 
                       encoder_length = encoder_length + 1,
                       decoder_length = decoder_length,
                       time_interval = 60),
                  #dict(type = 'load_dfs',pairs =  pairs,
                  #      data_path = "/home/ycc/additional_life/binance-public-data/data/data/spot/monthly/klines",
                  #      headers = ('real_time', 'open', 'high','low','close','volume',
                  #                 'Close_time','Quote_asset_volumne','Number_of_trades',
                  #                 'Taker_buy_base_asset_volume',"Taker_buy_quote_asset_volume",'ignore'),
                  #      ),
                  dict(type = 'select_columns',
                       selected_headers =  ('real_time','Price','Quantity')),
                  dict(type = 'crop_df',
                        encoder_length = encoder_length + 1,
                        decoder_length = decoder_length,
                        time_interval = 60,
                        pointnet = True),
                  #dict(type = 'scaler',
                  #      pickle_path = '/home/ycc/TSFF/scaler_202210.pkl',
                  #      selected_cols = selected_cols), 
                  dict(type = 'target_split',
                        selected_cols = selected_cols,
                        encoder_length = encoder_length + 1,
                        decoder_length = decoder_length),
                  dict(type = 'time_split',
                        selected_cols = selected_cols),
                  dict(type = 'pointnet_transform',
                        selected_cols = selected_cols,
                        time_interval = 60,
                        pairs = pairs
                        ),
                  dict(type = 'convert_np2ts',
                        keys = ['x_data','time_stamp','voxels','coords','num_points']),
                  dict(type = 'triple_barrier',
                        selected_cols = selected_cols,
                        target_pair = target_pair,
                        barrier_width = 0.001)
]

# model settings
dataset = dict(
    type = 'monthly_dataset',
    data_path = "/home/ycc/additional_life/binance-public-data/data_agg/spot/monthly/aggTrades",
    pairs = pairs,
    target_pair = ['ETHUSDT'],
    start_date = "20210801",
    end_date = "20210930",
    time_interval = 60,
    encoder_length = encoder_length + 1, # encoder 
    decoder_length = decoder_length,
    val_cutoff = 0.8,
    transforms = ['plain_target','stat_diff'],
    data_type = 'ohlcv',
    #barrier_width =  0.002,
    batch_size = 16,
    pipeline = train_pipeline,
    num_workers =0, 
    selected_cols = selected_cols,
    selected_headers =  ['Agg_tradeId','Price','Quantity','First_tradeID','Last_tradeID','real_time','maker','bestPrice'], # Timestamp -> real_time cohersion

    load_memory = dict(type = 'load_memory_time',pairs =  pairs,
                        data_path = "/home/ycc/additional_life/binance-public-data/data_agg/spot/monthly/aggTrades",
                        data_type = 'aggtrades'
                        
                        ),
    output_keys = ['x_data','target','voxels','num_points','coords']
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
        type = "base_embedding",
        #cat embedding
        value_embedding = dict(
            type = 'pillar_vfe',
            d_model = d_model, # should be loaded in from_dataset
            num_point_features= len(pairs) * len(selected_cols),
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
        type = 'single_mlp_output',
        hidden_size = d_model,
        dropout = dropout,
        output_size = 1 # from dataset
    ),
    )