# TSFF

## Design Patterns

## Module structure
```
.
├── README.md
├── algorithm_module - module that controls deep learning training,validation and test
│   ├── PF_test.ipynb
│   ├── configs - configs controls dataset preprocessing, which submodule to call (composite model), their hyperparams
│   │   ├── autoformer
│   │   ├── informer
│   │   ├── pointformer
│   │   └── tft
│   ├── experiment_control - training different models sequentially by calling different configs. script for easy model comparison.
│   │   ├── experiment_list.csv
│   │   └── experiment_training.py
│   ├── models - pytorch modules, submodule structure follows tft.
│   │   ├── attention - enc-dec module
│   │   │   ├── attention_layer - individual attention layer module
│   │   │   ├── decomp_enc.py
│   │   │   ├── decomp_enc_dec.py
│   │   ├── embedding - module that convert raw data to embedded vectors
│   │   ├── local_encoder - local encoder parts that accords with tft. bypass embedding vectors to attention module if not used
│   │   ├── utils - general func (decoder masks and etc.)
│   │   ├── models - submodule wrapper
│   │   │   ├── base_model.py - one that follows tft structure
│   │   ├── post_attention - final layers which yield multi-ahead reg scalar value or labels for classification.
│   │   └── variable_selection - variable selection on embed vector. accords with tft bypass if not used.
│   ├── train.py - script that help to train, train resume and predict. Called by experiment_control or user directly. 
│   ├── train_object.py - object that wraps not only model computation graph, but also controls data preparation.
│   └── utils
├── backtest_module - module that evaluate the trained model either statistical metrics (RMSE,etc...) or with backtest training strategy 
│   ├── backtest_list.csv
│   └── base_backtest.py
├── data_module - module that handles storing and communicating with exchange using ccxt and loads stored or downloaded dataset for model training.
│   ├── Data_exploring_test.ipynb - notebook files that shows how to connect and handles data from exchange using ccxt.
│   ├── async_record.py - recording market data from exchange using async connection and writing.
│   ├── multiproc_record.py - to-Do: use multiproccess library to record market data from market data.
│   ├── dm.py - base module that handles market data.
│   ├── online_dm.py - module that handles online market data.
│   ├── stored_dm.py - module that handles stored market data.
│   ├── pipelines - module that preprocess raw data for pytorch learning.
│   └── utils
├── decision_module - module that handles trading strategy based on prediction from algorithm_module.
│   ├── base_decision.py - base mechanisms for trading strategy
│   ├── dynamic_betting_decision.py - to:Do - based on softmaxed probability, decide the side and proportion of trades
│   └── zero_one_cls_decision.py - based on probability, decide only side of trading, but not the size.
└── utils.py
```
## Supported Features

## How to install

## Getting Starts

