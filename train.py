import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import BertTokenizer
from src.models.transformer import Transformer
from src.data.dataloader_factory import DataLoaderFactory
from src.utils.helpers import create_look_ahead_mask, create_masked_padding, save_checkpoint
import logging

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_epoch(model, data_loader, optimizer, criterion, device):
    """
    Train the model for one epoch.
    
    Args:
        model: The Transformer model being trained.
        data_loader: The DataLoader providing the training data.
        optimizer: The optimizer used for backpropagation.
        criterion: Loss function for training.
        device: The device (CPU or GPU) to run the training on.
    
    Returns:
        The average loss over the entire epoch.
    """
    model.train()  # Set the model to training mode
    total_loss = 0.0  # Initialize total loss for this epoch

    for batch_size, (src, tgt) in enumerate(data_loader):
        src, tgt = src.to(device), tgt.to(device)

        # Prepare input and target sequences for the decoder
        tgt_input = tgt[:, :-1]  # Exclude the last token for decoder input
        tgt_output = tgt[:, 1:]  # Shift the target sequence by one for the decoder output

        # Generate masks
        src_mask = create_masked_padding(src)  # Masking for source input
        tgt_padding_mask = create_masked_padding(tgt_input)  # Masking for target input
        tgt_look_ahead_mask = create_look_ahead_mask(tgt_input.size(1))  # Look-ahead mask for the decoder
        tgt_mask = tgt_padding_mask & tgt_look_ahead_mask  # Combine padding and look-ahead masks

        # Zero out the gradients before each forward pass
        optimizer.zero_grad()

        # Forward pass through the model
        output = model(src, tgt_input, src_mask, tgt_mask)

        # Compute the loss
        loss = criterion(output.view(-1, output.size(-1)), tgt_output.view(-1))
        total_loss += loss.item()

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

    # Return the average loss for this epoch
    return total_loss / len(data_loader)


def validate_epoch(model, data_loader, criterion, device):
    """
    Validate the model for one epoch (no gradient computation).
    
    Args:
        model: The Transformer model being validated.
        data_loader: The DataLoader providing the validation data.
        criterion: Loss function for validation.
        device: The device (CPU or GPU) to run the validation on.
    
    Returns:
        The average validation loss over the entire epoch.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0  # Initialize total validation loss

    with torch.no_grad():  # Disable gradient computation for validation
        for src, tgt in data_loader:
            src, tgt = src.to(device), tgt.to(device)

            # Prepare input and target sequences for the decoder
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # Generate masks
            src_mask = create_masked_padding(src)
            tgt_padding_mask = create_masked_padding(tgt_input)
            tgt_look_ahead_mask = create_look_ahead_mask(tgt_input.size(1))
            tgt_mask = tgt_padding_mask & tgt_look_ahead_mask

            # Forward pass
            output = model(src, tgt_input, src_mask, tgt_mask)
            loss = criterion(output.view(-1, output.size(-1)), tgt_output.view(-1))
            total_loss += loss.item()

    # Return the average validation loss
    return total_loss / len(data_loader)


def train(model, train_loader, val_loader, epochs, optimizer, criterion, device, checkpoint_dir="checkpoints"):
    """
    Train the model for a specified number of epochs, including validation and checkpointing.
    
    Args:
        model: The Transformer model being trained.
        train_loader: DataLoader providing the training data.
        val_loader: DataLoader providing the validation data.
        epochs: Number of training epochs.
        optimizer: Optimizer for backpropagation.
        criterion: Loss function for training and validation.
        device: The device (CPU or GPU) to run the training on.
        checkpoint_dir: Directory where model checkpoints will be saved.
    """
    for epoch in range(epochs):
        # Training for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validation after each epoch
        val_loss = validate_epoch(model, val_loader, criterion, device)

        logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Save the model checkpoint
        save_checkpoint(model, optimizer, epoch, train_loss, checkpoint_dir)


if __name__ == "__main__":
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Load the dataset using DataLoaderFactory
    train_loader = DataLoaderFactory.get_dataloader(
        data_type="text",
        text_data=["Sample sentence 1", "Sample sentence 2", "Sample sentence 3"],       # TODO: replace with actual dataset
        tokenizer=tokenizer,
        batch_size=32,
        max_len=128
    )

    val_loader = DataLoaderFactory.get_dataloader(
        data_type="text",
        text_data=["Validation sentence 1", "Validation sentence 2"],                   # TODO: replace with actual dataset  
        tokenizer=tokenizer,
        batch_size=32,
        max_len=128
    )

    # Initialize the Transformer model
    model = Transformer(
        num_encoder_layers=6,  # Number of encoder layers
        num_decoder_layers=6,  # Number of decoder layers
        d_model=512,           # Dimensionality of the model
        num_heads=8,           # Number of attention heads
        d_ff=2048,             # Dimensionality of feedforward layers
        src_vocab_size=10000,  # Source vocabulary size
        tgt_vocab_size=10000   # Target vocabulary size
    )

    # Move the model to the chosen device (GPU if available, otherwise CPU)
    model.to(device)

    # Set up optimizer and loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index in loss calculation
    optimizer = Adam(model.parameters(), lr=0.001)

    # Set the number of training epochs
    train(model, train_loader, val_loader, epochs=10, optimizer=optimizer, criterion=criterion, device=device)