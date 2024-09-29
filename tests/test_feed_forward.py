import torch
import pytest 
from src.models.feed_forward import PositionwiseFeedForward

def test_feed_forward():
    batch_size = 2
    seq_len = 5
    d_model = 64
    d_ff = 256

    # defining the input tensor and initializing the feedforward network
    input_tensor = torch.rand(batch_size, seq_len, d_model)
    feed_forward = PositionwiseFeedForward(d_model, d_ff)

    # running forward pass to capture the output
    output = feed_forward(input_tensor)

    # check output shape
    assert output.shape == (batch_size, seq_len, d_model), "Feed forward output shape is incorrect"

    # check that the feed forward network transforms the input tensor
    assert not torch.equal(output, input_tensor), "Feed forward network performed no transformations on the input tensor"


