import torch
import torch.nn as nn
# importing relevant modules
from models.attention import MultiHeadAttention
from models.feed_forward import PositionwiseFeedForward
from models.embeddings import TransformerEmbedding

class EncoderLayer(nn.Module):
    """
    implements a single layer of the transformer encoder.

    this layer consists of:
    1. multi-head self-attention mechanism
    2. position-wise feed-forward network
    3. layer normalization and residual connection around each sub-layer

    Args:
        d_model (int): dimension of the input embeddings and model-size
        num_head (int): number of attention heads in the multi-head attention mechanism
        d_ff (int): dimension of the inner feed-forward layer
        dropout (float): dropout probability applied after each sub-layer
    Shape:
        input: (batch_size, seq_len, d_model)
        output: (batch_size, seq_len, d_model)
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        forward pass through the encoder layer
        Args: 
            x (tensor): input tensor of shape (batch_size, seq_len, d_model)
            mask (tensor, optional): mask tensor of shape (batch_size, seq_len, seq_len)
                                     positions with True values indicating positions to be masked.
        Returns:
            tensor: output tensor of shape (batch_size, seq_len, d_model)
        """
        # applying multi-head attention with residual connection
        attention_output, _ = self.self_attention(x, x, x, mask)
        x += self.dropout(attention_output)                             # apply dropout
        x = self.layer_norm1(x)                                         # apply layer normalization

        # apply position-wise feed-forward network with residual connections
        ff_output = self.feed_forward(x)
        x += self.dropout(ff_output)                                    # apply dropout
        x = self.layer_norm2(x)                                         # apply layer normalization

        return x
    
class Encoder(nn.Module):
    """
    implements the full encoder stack of the transformer
    Args:
        num_layers (int): number of encoder layers
        d_models (int): dimension of the input embeddings and model size
        num_heads (int): number of attention heads in the multi-head attention mechanism
        d_ff (int): dimension of the inner feed-forward layer
        vocab_size (int): size of the input vocabulary
        mex_len (int): maximum sequence length
        dropout (float): dropout probability applied after each sub-layer
    Shape:
        input: (batch_size, seq_len)
        output: (batch_size, seq_len, d_model)
    """
    def __init__(self, num_layer, d_model, num_heads, d_ff, vocab_size, max_len=5000, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = TransformerEmbedding(vocab_size, d_model, max_len, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layer)])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, src, mask=None):
        """
        forward pass through the encoder stack
        Args:
            src (tensor): input tensor of shape (batch_size, seq_len)
            mask (tensor, optional): mask tensor of shape (batch_size, seq_len, seq_len)
        Returns:
            tensor: output tensor of shape (batch_size, seq_len, d_model)
        """
        x = self.embedding(src)
        for layer in self.layers:
            x = layer(x, mask)
        return self.layer_norm(x)
    
