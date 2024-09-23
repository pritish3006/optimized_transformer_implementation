import torch
import torch.nn as nn
from models.embeddings import TransformerEmbedding

class Transformer(nn.Module):
    """
    
    """
    def __init__(self, src_vocab_size, tgt_vocab_sized_model, num_layers, num_heads, d_ff, dropout, max_len=5000):
        super(Transformer, self).__init__()
        self.encoder_embedding = TransformerEmbedding(src_vocab_size, d_model, max_len, dropout)
        self.decoder_embedding = TransformerEmbedding(src_vocab_size, d_model, max_len, dropout)
        # initializing endocer and decoder stacks
        # self.encoder
        # self.decoder
        # declare final linear layer and softmax if needed

    def forward(self, src, tgt, src_mask, tgt_mask):
        src_embedded = self.encoder_embeddings(src)
        tgt_embedded = self.decoder_embeddings(tgt)
        # pass through encoder and decoder
        # encoder_output
        # decoder_output
        # apply final linear layer
        # output = self.final_layer(decoder_output)
        # return output
