import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward
from .embeddings import TransformerEmbedding

class DecoderLayer(nn.Module):
    """
    implements a single layer of the transformer decoder layer

    this layer consists of:
    1. masked multi-head self-attention mechanism for target sequences
    2. multi-head cross-attention mechanism with the encoder's output
    3. position-wise feed-forward network layer
    4. layer normalization and residual connections around each sub-layer

    Args:
        d_model (int): dimension of the input embeddings and model size
        num_heads (int): number of attention heads in the multi-head attention mechanism
        d_ff (int): dimension of the inner feed-forward layer
        dropout (float): dropout probability applied after each sub-layer
    Shape:
        input: (batch_size, tgt_seq_len, d_model), (batch_size, src_seq_len, d_model)
        output: (batch_size, tgt_seq_len, d_model)
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        # masked multi-head self-attention for target sequences
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        # multi-head cross-attention with encoder output
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        # position-wise feed-forward network layer
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        # layer normalization applied before each sub-layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        # dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        """
        Forward pass through the decoder layer
        Args:
            x (tensor): input tensor of shape (batch_size, tgt_seq_len, d_model)
            memory (tensor): encoder output tensor of shape (batch_size, src_seq_len, d_model)
            tgt_mask (tensor, optional): mask for target sequences (batch_size, tgt_seq_len, tgt_seq_len)
            memory_mask (tensor, optional): mask for encoder output (batch_size, tgt_seq_len, src_seq_len)
        Returns:
            tensor: output tensor of shape (batch_size, tgt_seq_len, d_model)
        """
        # masked self-attention for target sequence with residual connection
        self_attention_output, _ = self.self_attention(x, x, x, tgt_mask)               # output of multi-head self-attention mechanism
        x += self.dropout(self_attention_output)                                        # apply dropout probability after attention layer
        x = self.layer_norm1(x)                                                         # apply layer normalization

        # cross-attention with encoder output (memory) with residual connection
        cross_attention_output, _ = self.cross_attention(x, memory, memory, memory_mask)
        x += self.dropout(cross_attention_output)                                       # apply dropout probability after attention mechanism
        x = self.layer_norm2(x)                                                         # apply layer normalization

        # apply position-wise feed-forward network with residual connection
        ff_output = self.feed_forward(x)
        x += self.dropout(ff_output)                                                    # apply dropout probability after FFN layer
        x = self.layer_norm3(x)                                                         # apply layer normalization

        return x 

# defining the decoder stack
class Decoder(nn.Module):
    """
    implements the full decoder stack of the transformer
    Args:
        num_layers (int): number of decoder layers
        d_model (int): dimension of the input embeddings and model size
        num_heads (int): number of attention heads in the multi-head attention mechanism
        d_ff (int): dimension of the inner feed-forward layer
        vocab_size (int): size of the output vocabulary
        max_len (int): maximum sequence length
        dropout (float): dropout probability applied after each sub-layer
    Shape:
        input: (batch_size, tgt_seq_len)
        output: (batch_size, tgt_seq_len, d_model)
    """
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, max_len=5000, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = TransformerEmbedding(vocab_size, d_model, max_len, dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        forward pass through the decoder stack.
        Args:
            tgt (tensor): target input tensor of shape (batch_size, tgt_seq_len)
            memory (tensor): encoder output tensor (memory) of shape (batch_size, src_seq_len, d_model)
            tgt_mask (tensor, optional): mask for encoder output (batch_size, tgt_seq_len, src_seq_len)
        Returns:
            tensor: output tensor of shape (batch_size, tgt_seq_len, d_model)
        """
        x = self.embedding(tgt)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)                 # applying the decoder on every layer of data passing through the decoder stack
        return self.layer_norm(x)
    

