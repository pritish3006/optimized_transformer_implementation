import pytest 
import torch
from src.models.attention import ScaledDotProductAttention, MultiHeadAttention

def test_scaled_dot_product_attention():
    batch_size = 2
    seq_len = 5
    d_k = 64

    query = torch.rand(batch_size, seq_len, d_k)
    key = torch.rand(batch_size, seq_len, d_k)
    value = torch.rand(batch_size, seq_len, d_k)
    mask = torch.zeros(batch_size, seq_len, seq_len).bool()             # no masking

    attention = ScaledDotProductAttention()
    output, attention_weights = attention(query, key, value, mask=mask)

    assert output.shape == (batch_size, seq_len, d_k), "output shape is incorrect"
    assert attention_weights.shape == (batch_size, seq_len, seq_len), "attention weights shape is incorrect"


def test_multi_head_attention():
    batch_size = 2
    seq_len = 5
    d_model = 64
    num_heads = 8

    query = torch.rand(batch_size, seq_len, d_model)
    key = torch.rand(batch_size, seq_len, d_model)
    value = torch.rand(batch_size, seq_len, d_model)
    mask = torch.zeros(batch_size, seq_len, seq_len).bool()             # no masking

    attention = MultiHeadAttention(d_model, num_heads)
    output, attention_weights = attention(query, key, value, mask=mask, average_weights=True)

    assert output.shape == (batch_size, seq_len, d_model), "output shape is incorrect"
    assert attention_weights.shape == (batch_size, seq_len, seq_len), "attention weight shape is incorrect"