import pytest
import torch
from src.models.embeddings import TokenEmbedding, PositionalEncoding, TransformerEmbedding

def test_token_embedding():
    vocab_size = 100
    d_model = 64
    embedding = TokenEmbedding(vocab_size, d_model)

    input_tokens = torch.randint(0, vocab_size, (10, 20))
    output = embedding(input_tokens)

    assert output.shape == (10, 20, d_model), "TokenEmbedding output shape is incorrect"

def test_positional_encoding():
    vocab_size = 100
    d_model = 64
    max_len = 100
    encoding = TransformerEmbedding(vocab_size, d_model, max_len)

    input_tensor = torch.zeros(10, 20, d_model)
    output = encoding(input_tensor)

    assert output.shape == (10, 20, d_model), "PositionalEncoding output shape is incorrect"
    assert not torch.equal(output, input_tensor), "PositionalEncoding did not change the input tensor"

def test_transformer_embedding():
    vocab_size = 100
    d_model = 64
    max_len = 100
    embedding = TransformerEmbedding(vocab_size, d_model, max_len)

    input_tokens = torch.randint(0, vocab_size, (10, 20))               # batch of 10 sequences of length 20
    output = embedding(input_tokens)

    assert output.shape == (10, 20, d_model), "TransformerEmbedding output shape is incorrect"
