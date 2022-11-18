import torch
from torch import nn
from tsff.algorithm_module.models.builder import VARIABLE_SELECTION, build_embedding, build_enrichment, build_static_encoder,build_variable_selection, build_variable_selection_layer

@VARIABLE_SELECTION.register_module()
class base_variable_selection(nn.Module):

    # set 
    def __init__(self,
            max_encoder_length,
            static_variable_selection,
            encoder_variable_selection,
            decoder_variable_selection,
            static_encoder,
            ):

        super().__init__()
        self.max_encoder_length = max_encoder_length
        # variable selection from TFT - 
        # to-DO: implement class that just pass 
        ## current option: VariableSelectionNetwork, NoSelection
        ## group input size update function -


        self.static_variable_selection = build_variable_selection(static_variable_selection,)
        self.encoder_variable_selection = build_variable_selection_layer(encoder_variable_selection,)
        self.decoder_variable_selection = build_variable_selection_layer(decoder_variable_selection,)
        self.static_encoder = build_static_encoder(static_encoder)



    def forward(self,input_vectors,timesteps,static_variables,encoder_variables,decoder_variables):

        # variable name selection form input vectors are done inside the varaible_selection - change it 
        
        #static_varaible selection
        static_embedding, static_variable_selection = self.static_variable_selection(input_vectors,
            selected_variables = static_variables,
            max_encoder_length = self.max_encoder_length,
            encoder_flag =  "static")

        # one of 4 context vectors of static embedding
        static_context_variable_selection, static_input_hidden, static_input_cell, static_context_enrichment = \
             self.static_encoder(static_embedding,timesteps)

        if static_context_enrichment is not None:
            encoder_static_variable_selection = static_context_variable_selection[:,:self.max_encoder_length]
        else:
            encoder_static_variable_selection = None

        if static_context_enrichment is not None:
            decoder_static_variable_selection = static_context_variable_selection[:,self.max_encoder_length:]
        else:
            decoder_static_variable_selection = None
            
        #encoder variable_selection
        embeddings_varying_encoder, encoder_sparse_weights = self.encoder_variable_selection(
            embeddings = input_vectors,
            static_context_vector = encoder_static_variable_selection,
            selected_variables = encoder_variables,
            max_encoder_length = self.max_encoder_length, 
            encoder_flag = "encoder")

        #decoder variable_selection
        embeddings_varying_decoder, decoder_sparse_weights = self.decoder_variable_selection(
            embeddings = input_vectors,
            static_context_vector = decoder_static_variable_selection,
            selected_variables = decoder_variables,
            max_encoder_length = self.max_encoder_length, 
            encoder_flag = "decoder")


        # return embeddings (3), 4 static enrichment vectors, 3 sparse_weights
        return static_embedding, embeddings_varying_encoder, embeddings_varying_decoder, \
            static_context_variable_selection, static_input_hidden, static_input_cell, static_context_enrichment, \
                static_variable_selection, encoder_sparse_weights, decoder_sparse_weights


        


    def input_size_update(self,cat,real):
        input_sizes = {
            name: self.input_embeddings.output_size[name] for name in cat
        }
        input_sizes.update(
            {
                name: self.hparams.hidden_continuous_sizes.get(name, self.hparams.hidden_continuous_size)
                for name in real           }
        )

    def expand_static_context(self, context, timesteps):
        """
        add time dimension to static context
        """
        return context[:, None].expand(-1, timesteps, -1)