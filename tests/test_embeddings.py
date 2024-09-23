<<<<<<< HEAD
import torch
import pytest
from src.models.embeddings import (
    TokenEmbedding,
    PositionalEncoding,
    TransformerEmbedding,
)


def test_token_embedding():
    vocab_size = 1000
    d_model = 512
    embedding = TokenEmbedding(vocab_size, d_model)
    x = torch.randint(0, vocab_size, (32, 20))
    output = embedding(x)
    assert output.shape == (32, 20, d_model)


def test_positional_encoding():
    d_model = 512
    max_len = 100
    pe = PositionalEncoding(d_model, max_len)
    x = torch.randn(32, 50, d_model)
    output = pe(x)
    assert output.shape == (32, 50, d_model)


def test_transformer_embedding():
    vocab_size = 1000
    d_model = 512
    max_len = 100
    batch_size = 32
    seq_len = 50

    embedding_layer = TransformerEmbedding(vocab_size, d_model, max_len)

    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    output = embedding_layer(x)

    assert output.shape == (batch_size, seq_len, d_model), "Output shape is incorrect"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains infinity values"
=======
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
>>>>>>> master
