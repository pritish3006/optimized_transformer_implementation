import torch

<<<<<<< HEAD

def create_padding_mask(seq, pad_token=1):
    """
    Create a mask to hide the padding token
    Args:
        seq: tensor of shape (batch_size, seq_len)
        pad_token: token used for padding (default is 0)
    Returns:
        mask: tensor of size (batch_size, 1, 1, seq_len)
    """
    # mask the positions where the sequence value is equal to the padding token
    return (
        (seq == pad_token).unsqueeze(1).unsqueeze(2)
    )  # output shape: (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    """
    creates a look ahead mask for the decoder to prevent attention from applying to future tokens
    Args:
        size: the size of the sequence (seq_len)
    Returns:
        mask: tensor of shape (1, size, size)
    """
    # creating a matrix with ones above the diagonal (this will be the look ahead mask)
    mask = torch.triu(torch.ones((size, size)), diagonal=1).bool()
    return mask
=======
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
    return mask                                                             # trie where pad tokens are present (dtype torch.bool)

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
    return mask                                                             # true value indicates position to be masked
>>>>>>> master
