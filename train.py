import torch
from src.utils.helpers import create_padding_mask, create_look_ahead_mask

# sample input tensor with token indices
src_input = torch.tensor([[1, 2, 3, 0, 0],[4, 5, 6, 7, 0]])
tgt_input = torch.tensor([[1, 2, 3, 4, 0],[5, 6, 7, 8, 0]])

# creating padding mask for source input (src_input)
src_mask = create_padding_mask(src_input)                                       #dtype torch.bool

# creating padding mask for target input (tgt_input)
tgt_padding_mask = create_padding_mask(tgt_input)                               #dtype torch.bool

# creating look ahead mask for target input
tgt_look_ahead_mask = create_look_ahead_mask(tgt_input.size(1))                 #dtype torch.bool

# combining both target masks for the decoder
tgt_mask = tgt_padding_mask & tgt_look_ahead_mask                               #using boolean AND operation. dtype torch.bool

# passing the masks to the model during forward pass operation
output = model(src_input, tgt_input, src_mask, tgt_mask)                        # TODO: Define model