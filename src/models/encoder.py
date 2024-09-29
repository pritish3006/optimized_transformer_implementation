import torch
import torch.nn as nn
# importing relevant modules
from models.attention import MultiHeadAttention
from models.feed_forward import PositionwiseFeedForward

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