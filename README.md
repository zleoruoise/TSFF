# TSFF
Time Series Forecasting for Financial data (TSFF) is an open source project for time series forecasting and classification using deep learning models, such as transformer and RNN/LSTM.

Compared to other deep learning open source project on time series forecasting, such as pytorch forecasting, TSFF has a modular design. We 
decomposed components as described in Temporal fusion transformers for inter- pretable multi-horizon time series forecasting.

In addition, model training configuration setting and interaction between model and datasets are followed Object Detection open source projects, such as OpenPCDet and MMDetection. 

In order to reduce repeated stuff, we select to use pytorch lightning for handling training, validation hook. Learning scheduler and loggings are directly brought from Pytorch lighning.

Currently, I have used only one type of financial data, cryptocurrency data from Binance, but it is because those dataset is the easiest to access. It is planned to add other types of financial data, such as stock and foreign exchange rate.
## Design Patterns
![Design](design_pattern.png)

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
│   │   ├── wrapped_models - submodule wrapper
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
└── utils - module that handle registry and general utility function
```
## Supported Features

## How to install
### requirements
All the codes are tested in the following environment:
* Linux (tested on Ubuntu 20.04)
* Python 3.8+
* PyTorch 1.7 or higher (tested on PyTorch 1.7, 1.11)
* PyTorch Lightning 1.7.1
* CUDA 11.0 or higher 
*  [`spconv v2.x`](https://github.com/traveller59/spconv)


### Install `TSFF`

a. Clone this repository.
```shell
git clone https://github.com/zleoruoise/TSFF.git
```

b. Install the dependent libraries as follows:
```shell
pip install -r requirements.txt 
```


c. Install this `pcdet` library and its dependent libraries by running the following command:
```shell
python setup.py develop
```

## Quick Demo
## Getting Starts

