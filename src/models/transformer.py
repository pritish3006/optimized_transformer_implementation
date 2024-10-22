import torch
import torch.nn as nn
from .embeddings import TransformerEmbedding
from .encoder import Encoder
from .decoder import Decoder
from .feed_forward import PositionwiseFeedForward
from src.utils.helpers import initialize_weight

class Transformer(nn.Module):
    """
    implements the full transformer model with encoder, decoder, and final output layers.

    Args:
        num_encoder_layers (int): number of encoder layers.
        num_decoder_layers (int): number of decoder layers.
        d_model (int): dimension of the input embeddings and model size.
        num_heads (int): number of attention heads in the multi-head attention mechanism.
        d_ff (int): dimension of the inner feed-forward layer.
        src_vocab_size (int): size of the input vocabulary.
        tgt_vocab_size (int): size of the output vocabulary.
        max_len (int): maximum sequence length.
        dropout (float): dropout probability applied after each sub-layer.

    Shape:
        input: (batch_size, src_seq_len), (batch_size, tgt_seq_len)
        output: (batch_size, tgt_seq_len, tgt_vocab_size)
    """
    def __init__(self, num_encoder_layers, num_decoder_layers, d_model, num_heads, d_ff, src_vocab_size, tgt_vocab_size, max_len=5000, dropout=0.1):
        super(Transformer, self).__init__()
        # initializing endocer and decoder stacks
        self.encoder = Encoder(num_encoder_layers, d_model, num_heads, d_ff, src_vocab_size, max_len, dropout)      # encoder layer
        self.decoder = Decoder(num_decoder_layers, d_model, num_heads, d_ff, tgt_vocab_size, max_len, dropout)      # decoder layer
        self.linear = nn.Linear(d_model, tgt_vocab_size)                                                            # final linear layer for output logits
        self.softmax = nn.Softmax(dim=1)                                                                            # softmax for generating token probabilities

        # Initialize weights
        self.apply(initialize_weight)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        """
        forward pass through the full transformer model
        Args:
            src (tensor): input source tensor of shape (batch_size, src_seq_len)
            tgt (tensor): target input tensor of shape (batch_size, tgt_seq_len)
            src_mask (tensor, optional): mask for the input sequence (batch_size, src_seq_len, src_seq_len)
            tgt_mask (tensor, optional): mask for the target sequence (batch_size, tgt_seq_len, tgt_seq_len)
            memory_mask (tensor, optional): mask for the memory (encoder output) (batch_size, tgt_seq_len, src_seq_len)
        Returns:
            tensor: output tensor of shape (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # pass through encoder and decoder
        memory = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, memory, tgt_mask, memory_mask)
        # apply final linear layer
        output = self.linear(decoder_output)
        # return output
        return output
