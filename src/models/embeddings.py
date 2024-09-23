import torch
import torch.nn as nn
import math
import logging

# setting up the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TokenEmbedding(nn.Module):
    """
        implementing the token embeddings
    """
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
     
class PositionalEncoding(nn.module):
    """
    implements the sinusoidal positional encoding function.

    Args:
        d_model (int): The dimension of the model embeddings.
        max_len (int): The maximum length of input sequences.

    Attributes:
        pe (Tensor): A buffer containing positional encodings of shape (1, max_len, d_model).
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)  # Shape: (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # Shape: (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )  # Shape: (d_model//2,)
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Shape: (max_len, d_model//2)
        pe[:, 1::2] = torch.cos(position * div_term)  # Shape: (max_len, d_model//2)
        
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]  # Slicing pe to match seq_len
        logger.debug(f"PositionalEncoding forward output shape: {x.shape}")
        return x    
class TransformerEmbedding(nn.Module):
    """
        implementing transformer embeddings
    """
    def __init__(self, vocab_size, d_model, max_len=5000, dropout=0.1):
        super(TransformerEmbedding, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        embedded = self.token_embedding(x)
        embedded_with_pos = self.positional_encoding(embedded)
        return self.dropout(embedded_with_pos)
    

    
