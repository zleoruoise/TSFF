from json import encoder
import torch
from torch import nn
import torch.nn.functional as F
from ....models.builder import ATTENTION_LAYERS, build_attention_layer, build_enrichment, build_local_encoder_layer 
from .full_attn import AttentionLayer

@ATTENTION_LAYERS.register_module()
class base_encoder(nn.Module):

    # set 
    def __init__(self, 
        layer_type,
        hidden_size,
        dim_ff,
        nhead,
        dropout,
        activation,
        layer_num):


        super(base_encoder,self).__init__()

        encoder_layers = []
        conv_layers = []

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

        for i in range(layer_num - 1):
            conv_layers.append(ConvLayer(hidden_size))

        self.attn_layers = nn.ModuleList(encoder_layers)
        self.conv_layers = nn.ModuleList(conv_layers)
        self.norm = nn.LayerNorm(hidden_size)

        
    def forward(self,x,attn_mask = None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


@ATTENTION_LAYERS.register_module()
class encoder_layer(nn.Module):
    def __init__(self, sa_layer_type, d_model, nhead,dim_ff=None, dropout=0.1, activation="relu"):
        super(encoder_layer, self).__init__()

        if isinstance(sa_layer_type, dict):
            new_args = dict(
                nhead = nhead,
                d_model = d_model)

            sa_layer_type.update(new_args)



        dim_ff = dim_ff or 4*d_model
        self.attention = build_attention_layer(sa_layer_type) 
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=dim_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=dim_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        new_x, attn = self.attention(
            x, x, x,
            attn_mask = attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm2(x+y), attn

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1,2)
        return x

