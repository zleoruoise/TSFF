import torch
from torch import nn
from ..builder import ATTENTION, build_attention_layer, build_enrichment
from .attention_layer.autoformer_enc_dec import series_decomp

@ATTENTION.register_module()
class decomp_enc(nn.Module):

    # set 
    def __init__(self, 
                local_enrichment_layer = None,
                encoder_attention = None,
                decoder_attention = None,
                moving_avg = 25):

        super().__init__()

        if local_enrichment_layer is None:
            local_enrichment_layer = dict(type ='empty_layer')
        if encoder_attention is None:
            encoder_attention = dict(type = 'empty_layer')
        if decoder_attention is None:
            decoder_attention = dict(type = 'empty_layer')

        self.static_enrichment = build_enrichment(local_enrichment_layer)
        self.encoder_attention = build_attention_layer(encoder_attention)
        self.decoder_attention = build_attention_layer(decoder_attention)

        self.decomp = series_decomp(moving_avg)

        
    # base follows tft- others will reimplement forward    #
    def forward(self,x):
        # input
        enc_in = x['encoded_data'] 

        # module computation
        enc_out , attns = self.encoder_attention(enc_in)
        x['attn_data'] = enc_out

        return x