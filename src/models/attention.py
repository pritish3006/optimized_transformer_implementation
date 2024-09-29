import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScaledDotProductAttention(nn.Module):
    """
        implements the scaled dot-product attention mechanism.
    """
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, query, key, value, mask=None):
        """
        Args: 
            query: tensor of shape (..., seq_len_q, d_k)
            key: tensor of shape (..., seq_len_k, d_k)
            value: tensor of shape (..., seq_len_v, d_v)
            mask: tensor of shape (..., seq_len_q, seq_len_k) with dtype torch.bool
                    True values indicate positions to be masked
        Returns: 
            output: tensor after applying attention
            attention_weights: tensor of attention weights
        """
        d_k = query.size(-1)
        # compute the scaled dot-products
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        logger.debug(f"scores shape: {scores.shape}")

        # apply mask if provided
        if mask is not None:
            # mask where mask is true (masked positions are the positions that need to be ignored)
            scores = scores.masked_fill(mask, float('-inf'))

        # compute the attention weights
        attention_weights = F.softmax(scores, dim=-1)
        logger.debug(f"arrention weights shape: {attention_weights.shape}")

        # apply attention weights to values
        output = torch.matmul(attention_weights, value)
        logger.debug(f"output shape: {output.shape}")

        return output, attention_weights
    
class MultiHeadAttention(nn.Module):
    """
        implements the multi-head self-attention mehanism
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by the number of heads num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # linear layers for query, key, value projections
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)

        # scaled dot-product attention
        self.attention = ScaledDotProductAttention()

        # final linear layer
        self.linear_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, average_weights=True):
        """
        Args:
            query: tensor of shape (batch_size, seq_len_q, d_model)
            key: tensor of shape (batch_size, seq_len_k, d_model)
            value: tensor of shape (batch_size, seq_len_v, d_model)
            mask: tensor of shape (batch_size, num_heads, seq_len_q, seq_len_k) with dtype torch.bool
                    True value indicates positions to be masked
        Returns:
            output: tensor after applying multi-head attention
            attention_weights: tensor of attention weights from all heads
        """
        batch_size = query.size(0)

        # Linear projections for query, key, and value
        query = self.linear_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.linear_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Adjust the mask dimensions to match (batch_size, num_heads, seq_len_q, seq_len_k)
        if mask is not None:
           mask = mask.unsqueeze(1)                            # Add a new dimension for num_heads

        # Apply scaled dot-product attention independently for each head
        attention_output, attention_weights = self.attention(query, key, value, mask=mask)

        # Conditionally average the attention weights
        if average_weights:
            attention_weights = attention_weights.mean(dim=1)  # Shape: (batch_size, seq_len, seq_len)

        # Concatenate all the attention outputs and project to d_model
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        output = self.linear_out(attention_output)
        output = self.dropout(output)

        return output, attention_weights 