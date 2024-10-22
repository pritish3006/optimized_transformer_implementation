import torch
import os
import torch.nn as nn

def create_masked_padding(seq, pad_token=0):
    """
    creates a mask for padding tokens
    Args:
        seq: tensor of shape (batch_size, seq_len)
        pad_token: the token used for padding (default is 0)
    Returns: 
        mask: tensor of shape (batch_size, 1, 1, seq_len) with dtype torch.bool
        true values indicate padding positions to be masked.
    """
    # creating a mask where true indicates positions that are padded tokens
    mask = (seq == pad_token).unsqueeze(1).unsqueeze(2)                     # shape: (batch_size, 1, 1, seq_len)
    print(f"mask shape in create_masked_padding: {mask.shape}")
    return mask                                                             # true where pad tokens are present (dtype torch.bool)

def create_look_ahead_mask(size):
    """
    creates a look-ahead mask for the decoder to prevent attending to future tokens.
    Args:
        size: the size of the sequence (seq_len)
    Returns:
        mask: tensor of shape (1, size, size) with dtype torch.bool
                True values indicate positions to be masked (future tokens)
    """
    # create a matrix with ones above the main diagonal (look-ahead mask)
    mask = torch.triu(torch.ones((size, size)), diagonal=1).bool()          # true above the diagonal
    print(f"mask shape in create_look_ahead_mask: {mask.shape}")
    return mask                                                             # true value indicates position to be masked

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir="checkpoints"):
    """
    saves a checkpoint of the model's state including model weights, optimizer state and loss
    Args:
        model: transformer model being trained
        optimizer: the optimizer being used during training
        epoch: current epoch number
        loss: training loss at the time of checkpointing
        checkpoint_dir: directory where checkpoint files will be saved
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)                                         # create checkpoint dir if it doesn't already exist

    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, checkpoint_path)

    print(f"checkpoint saved: {checkpoint_path}")                

def initialize_weight(x):
    if isinstance(x, nn.Linear):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)
    elif isinstance(x, nn.LayerNorm):
        nn.init.constant_(x.weight, 1.0)
        nn.init.constant_(x.bias, 0)
