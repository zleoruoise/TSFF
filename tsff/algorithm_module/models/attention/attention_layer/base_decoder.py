from base64 import decode
import torch
import torch.nn as nn
import torch.nn.functional as F
from ....models.builder import ATTENTION_LAYERS, build_attention_layer, build_enrichment, build_local_encoder_layer 
from .full_attn import AttentionLayer



@ATTENTION_LAYERS.register_module()
class base_decoder(nn.Module):
    def __init__(self, 
        layer_type,
        hidden_size,
        dim_ff,
        nhead,
        dropout,
        activation,
        layer_num):
        super(base_decoder, self).__init__()


        
        if isinstance(layer_type, dict):
            new_args = dict(
                dim_ff = dim_ff,
                nhead = nhead,
                d_model = hidden_size,
                dropout = dropout,
                activation = activation)

            layer_type.update(new_args)



        layers = []
        for i in range(layer_num):
            layers.append(build_attention_layer(layer_type))
        self.norm = nn.LayerNorm(hidden_size)
        self.layers = nn.ModuleList(layers)

    def forward(self, x, mems, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, mems, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x

@ATTENTION_LAYERS.register_module()
class decoder_layer(nn.Module):
    def __init__(self, sa_layer_type, cross_layer_type, d_model, nhead,dim_ff=None,
                 dropout=0.1, activation="relu"):
        super(decoder_layer, self).__init__()


        if isinstance(sa_layer_type, dict):
            new_args = dict(
                nhead = nhead,
                d_model = d_model)

            sa_layer_type.update(new_args)

        if isinstance(cross_layer_type, dict):
            new_args = dict(
                nhead = nhead,
                d_model = d_model)

            cross_layer_type.update(new_args)


        dim_ff = dim_ff or 4*d_model
        self.self_attention = build_attention_layer(sa_layer_type) 
        self.cross_attention = build_attention_layer(cross_layer_type) 
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=dim_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=dim_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm3(x+y)