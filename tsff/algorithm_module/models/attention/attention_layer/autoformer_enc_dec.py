from calendar import c
from json import encoder
import torch
from torch import nn
import torch.nn.functional as F
from ....models.builder import ATTENTION_LAYERS, build_attention_layer, build_enrichment, build_local_encoder_layer 

from .base_encoder import encoder_layer
from .base_decoder import base_decoder, decoder_layer


class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x



class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

@ATTENTION_LAYERS.register_module()
class autoformer_encoder(nn.Module):

    # set 
    def __init__(self, 
        layer_type,
        hidden_size,
        dim_ff,
        nhead,
        dropout,
        activation,
        layer_num):


        super(autoformer_encoder,self).__init__()

        encoder_layers = []

        if isinstance(layer_type, dict):
            new_args = dict(
                dim_ff = dim_ff,
                nhead = nhead,
                d_model = hidden_size,
                dropout = dropout,
                activation = activation)

            layer_type.update(new_args)

        for i in range(layer_num):
            encoder_layers.append(
                build_attention_layer(layer_type))


        self.attn_layers = nn.ModuleList(encoder_layers)
        self.norm = nn.LayerNorm(hidden_size)

        
    def forward(self,x,attn_mask = None):
        # x [B, L, D]
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

@ATTENTION_LAYERS.register_module()
class autoformer_encoder_layer(encoder_layer):
    def __init__(self, sa_layer_type, d_model, nhead,dim_ff=None, dropout=0.1, moving_avg =25,activation="relu"):
        super(autoformer_encoder_layer, self).__init__(sa_layer_type,d_model,
            nhead,dim_ff,dropout,activation)

        dim_ff = dim_ff or 4*d_model

        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)


    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))

        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res, attn

@ATTENTION_LAYERS.register_module()
class autoformer_decoder(base_decoder):
    """
    Autoformer encoder
    """
    def __init__(self, 
        layer_type,
        hidden_size,
        dim_ff,
        nhead,
        dropout,
        activation,
        layer_num, 
        norm_layer=None, 
        projection=None):
        super(autoformer_decoder, self).__init__(layer_type,hidden_size,dim_ff,nhead,dropout,activation,layer_num)
        self.projection = projection

    def forward(self, x, mems, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, mems, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend

@ATTENTION_LAYERS.register_module()
class autoformer_decoder_layer(decoder_layer):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, sa_layer_type, cross_layer_type, d_model, nhead, dim_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(autoformer_decoder_layer, self).__init__(sa_layer_type,cross_layer_type,d_model,nhead)
        dim_ff = dim_ff or 4 * d_model
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x, trend1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend
