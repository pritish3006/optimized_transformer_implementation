import torch
import torch.nn as nn 
import logging

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PositionwiseFeedForward(nn.Module):
    """
        defining the positionwise feedforward network used in a transformer
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # apply the first linear transformation
        x = self.linear1(x)
        logger.debug(f"shape after first linear operation: {x.shape}, values: {x}")
        # applying relu activation function
        x = self.activation(x)
        logger.debug(f"shape after activation: {x.shape}, value: {x}")
        # applying the dropout layer
        x = self.dropout(x)
        logger.debug(f"shape after the dropout layer {x.shape}, value:  {x}")
        # applying the second linear transformation
        x = self.linear2(x)
        logger.debug(f"shape after second lienar transformation: {x.shape}, value: {x}")

        return x
    


