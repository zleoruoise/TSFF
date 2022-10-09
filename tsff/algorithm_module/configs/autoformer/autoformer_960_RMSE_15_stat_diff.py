# overal config size 

# General variable 

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
lstm_layers_num = 3

# forecast type
forecast_type = 'reg'

# work_dir - should be full path
work_dir = '/home/ycc/TSFF/work_dir'

# model settings
dataset = dict(
    type = 'monthly_dataset',
    data_path = "/home/ycc/additional_life/binance-public-data/data/data/spot/monthly/klines",
    pairs = ['BTCUSDT','ETHUSDT',"BNBUSDT","XRPUSDT","ADAUSDT"],
    target_pair = ['ETHUSDT'],
    start_date = "20210101",
    end_date = "20220331",
    time_interval = 60,
    encoder_length = encoder_length + 1, # encoder 
    decoder_length = decoder_length,
    val_cutoff = 0.8,
    transforms = ['plain_target','stat_diff'],
    data_type = 'ohlcv',
    #barrier_width =  0.002,
    batch_size = 16,
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
    loss = 'RMSE',
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
        type = 'dec_post_attention',
        hidden_size = d_model,
        dropout = dropout,
        output_size = 1 # from dataset
    ),
    )