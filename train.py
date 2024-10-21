import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertTokenizer
from src.models.transformer import Transformer
from src.data.dataloader_factory import DataLoaderFactory
from accelerate import Accelerator
from src.utils.helpers import create_look_ahead_mask, create_masked_padding, save_checkpoint
import logging
import wandb
from pathlib import Path
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.ribes_score import sentence_ribes
from rouge import Rouge
from torchmetrics.text import BLEUScore, CharErrorRate
from torchmetrics import Precision, Recall, F1Score, Accuracy, Perplexity
import numpy as np

# Initialize wandb
wandb.init(project="transformer_training", config={
    "learning_rate": 1e-4,
    "batch_size": 32,
    "epochs": 10,
    # Add other hyperparameters here
})

# Setup model checkpointing
checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

# Function to save model checkpoint with wandb
def save_model_checkpoint(model, optimizer, scheduler, epoch, loss, total_steps):
    """
    Save a checkpoint of the model's state during training.

    This function saves the model state, optimizer state, scheduler state,
    current epoch, loss, and total steps to a checkpoint file. It also logs
    the checkpoint file to Weights & Biases (wandb) for experiment tracking.

    Args:
        model (nn.Module): The model being trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        epoch (int): The current epoch number.
        loss (float): The current loss value.
        total_steps (int): The total number of training steps completed.

    Returns:
        None

    Side effects:
        - Creates a checkpoint file in the `checkpoint_dir` directory.
        - Logs the checkpoint file to wandb.
    """
    checkpoint_path = checkpoint_dir / f"model_epoch_{epoch}_steps_{total_steps}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'total_steps': total_steps,
    }, checkpoint_path)
    wandb.save(str(checkpoint_path))  # Log the checkpoint file to wandb

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_metrics(output, target, tokenizer):
    # Convert output probabilities to token ids
    pred_tokens = torch.argmax(output, dim=-1)
    
    # Convert tensors to lists
    pred_tokens = pred_tokens.cpu().numpy().tolist()
    target_tokens = target.cpu().numpy().tolist()
    
    # Convert token ids to words
    pred_words = [tokenizer.decode(sent, skip_special_tokens=True) for sent in pred_tokens]
    target_words = [tokenizer.decode(sent, skip_special_tokens=True) for sent in target_tokens]
    
    # Calculate BLEU score
    bleu = BLEUScore(n_gram=4)
    bleu_score = bleu(pred_words, [[t] for t in target_words])
    
    # Calculate METEOR score
    meteor = np.mean([meteor_score([t], p) for p, t in zip(pred_words, target_words)])
    
    # Calculate ROUGE score
    rouge = Rouge()
    rouge_scores = rouge.get_scores(pred_words, target_words, avg=True)
    
    # Calculate TER (Translation Edit Rate)
    ter = CharErrorRate()
    ter_score = ter(pred_words, target_words)
    
    # Calculate Precision, Recall, F1
    precision = Precision(task="multiclass", num_classes=tokenizer.vocab_size)
    recall = Recall(task="multiclass", num_classes=tokenizer.vocab_size)
    f1 = F1Score(task="multiclass", num_classes=tokenizer.vocab_size)
    
    precision_score = precision(pred_tokens, target_tokens)
    recall_score = recall(pred_tokens, target_tokens)
    f1_score = f1(pred_tokens, target_tokens)
    
    # Calculate Sequence-level and Token-level accuracy
    seq_accuracy = Accuracy(task="multiclass", num_classes=tokenizer.vocab_size)
    token_accuracy = Accuracy(task="multiclass", num_classes=tokenizer.vocab_size, average='micro')
    
    seq_acc = seq_accuracy(pred_tokens, target_tokens)
    token_acc = token_accuracy(pred_tokens, target_tokens)
    
    # Calculate Perplexity
    perplexity = Perplexity()
    ppl = perplexity(output.view(-1, output.size(-1)), target.view(-1))
    
    return {
        'bleu': bleu_score,
        'meteor': meteor,
        'rouge': rouge_scores,
        'ter': ter_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score,
        'seq_accuracy': seq_acc,
        'token_accuracy': token_acc,
        'perplexity': ppl
    }

def get_transformer_scheduler(optimizer, d_model, warmup_steps):
    def lr_lambda(step):
        if step == 0:
            return 0
        return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))
    return LambdaLR(optimizer, lr_lambda)

def train_epoch(model, data_loader, optimizer, criterion, scheduler, device,tokenizer):
    """
    Train the model for one epoch.
    
    Args:
        model: The Transformer model being trained.
        data_loader: The DataLoader providing the training data.
        optimizer: The optimizer used for backpropagation.
        criterion: Loss function for training.
        device: The device (CPU or GPU) to run the training on.
        tokenizer: The tokenizer used for decoding predictions.
    
    Returns:
        The average loss and metrics over the entire epoch.
    """
    model.train()                                                                   # Set the model to training mode
    total_loss = 0.0                                                                # Initialize total loss for this epoch
    all_metrics = {}    
    steps = 0

    for batch_idx, (src, tgt) in enumerate(data_loader):
        # Prepare input and target sequences for the decoder
        tgt_input = tgt[:, :-1]                                                     # Exclude the last token for decoder input
        tgt_output = tgt[:, 1:]                                                     # Shift the target sequence by one for the decoder output

        # Generate masks
        src_mask = create_masked_padding(src)                                       # Masking for source input
        tgt_padding_mask = create_masked_padding(tgt_input)                         # Masking for target input
        tgt_look_ahead_mask = create_look_ahead_mask(tgt_input.size(1))             # Look-ahead mask for the decoder
        tgt_padding_mask = tgt_padding_mask.to(device)                              # Move the padding mask to the device
        tgt_look_ahead_mask = tgt_look_ahead_mask.to(device)                        # Move the look-ahead mask to the device
        tgt_mask = tgt_padding_mask & tgt_look_ahead_mask                           # Combine padding and look-ahead masks

        # Zero out the gradients before each forward pass
        optimizer.zero_grad()

        # Forward pass through the model
        print(f"Source shape: {src.shape}")                                         # adding print statements to check the shapes of the source input tensors
        print(f"Target input shape: {tgt_input.shape}")                             # adding print statements to check the shapes of the target input tensors
        output = model(src, tgt_input, src_mask, tgt_mask)

        # Compute the loss
        loss = criterion(output.view(-1, output.size(-1)), tgt_output.view(-1))
        total_loss += loss.item()

        # Calculate metrics
        batch_metrics = calculate_metrics(output, tgt_output, tokenizer)
        for key, value in batch_metrics.items():
            all_metrics[key] = all_metrics.get(key, 0) + value

        # Backpropagation and optimization
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        steps += 1

        # Log metrics every 100 batches
        if batch_idx % 100 == 0:
            wandb.log({f"train_{k}": v for k, v in batch_metrics.items()})

    # Calculate average metrics
    avg_metrics = {k: v / len(data_loader) for k, v in all_metrics.items()}
    avg_loss = total_loss / len(data_loader)

    # Log average metrics for the epoch
    wandb.log({"train_loss": avg_loss, **{f"train_{k}": v for k, v in avg_metrics.items()}})

    return avg_loss, avg_metrics, steps


def validate_epoch(model, data_loader, criterion, device, tokenizer):
    """
    Validate the model for one epoch (no gradient computation).
    
    Args:
        model: The Transformer model being validated.
        data_loader: The DataLoader providing the validation data.
        criterion: Loss function for validation.
        device: The device (CPU or GPU) to run the validation on.
        tokenizer: The tokenizer used for decoding predictions.
    
    Returns:
        The average validation loss and metrics over the entire epoch.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0  # Initialize total validation loss
    all_metrics = {}

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
            # Forward pass through the model
            output = model(src, tgt_input, src_mask, tgt_mask)
            
            # Calculate the loss
            # Reshape the output and target tensors to 2D for the loss calculation
            # output shape: (batch_size * seq_len, vocab_size)
            # tgt_output shape: (batch_size * seq_len)
            loss = criterion(output.view(-1, output.size(-1)), tgt_output.view(-1))
            
            # Accumulate the total loss for the epoch
            total_loss += loss.item()

            # Calculate metrics
            batch_metrics = calculate_metrics(output, tgt_output, tokenizer)
            for key, value in batch_metrics.items():
                all_metrics[key] = all_metrics.get(key, 0) + value

    # Calculate average metrics
    avg_metrics = {k: v / len(data_loader) for k, v in all_metrics.items()}
    avg_loss = total_loss / len(data_loader)

    # Log average metrics for the validation
    wandb.log({"val_loss": avg_loss, **{f"val_{k}": v for k, v in avg_metrics.items()}})

    return avg_loss, avg_metrics


def train(model, train_loader, val_loader, epochs, optimizer, criterion, tokenizer, d_model=512, warmup_steps = 1000, checkpoint_dir="checkpoints"):
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
        tokenizer: The tokenizer used for decoding predictions.
        checkpoint_dir: Directory where model checkpoints will be saved.
    """

    # initializing accelerator instance
    accelerator = Accelerator()

    # initializing the scheduler
    scheduler = get_transformer_scheduler(optimizer, d_model, warmup_steps)

    # prepare the model, optimizer, data loaders, and scheduler to work with the accelerator
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        val_loader, 
        scheduler
    )

    # set the device using the accelerator
    device = accelerator.device

    best_val_loss = float('inf')
    total_steps = 0

    for epoch in range(epochs):
        # Training for one epoch
        train_loss, train_metrics, steps = train_epoch(model, train_loader, optimizer, criterion, scheduler, device, tokenizer)
        total_steps += steps
        
        # Validation after each epoch
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device, tokenizer)

        logger.info(f"epoch {epoch+1}/{epochs}")
        logger.info(f"train loss: {train_loss:.4f}, validation loss: {val_loss:.4f}")
        logger.info(f"train metrics: {train_metrics}")
        logger.info(f"validation metrics: {val_metrics}")
        logger.info(f"current learning rate: {scheduler.get_last_lr()[0]:.6f}")

        # Save the model checkpoint if it is the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model_checkpoint(model, optimizer, scheduler, epoch, val_loss, total_steps)

if __name__ == "__main__":
    
    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    
    # Define paths to the training source and target files
    train_src_file_path = 'Datasets/raw/en-hi/opus.en-hi-train.en'
    train_tgt_file_path = 'Datasets/raw/en-hi/opus.en-hi-train.hi'

    # Load training source and target data
    try:
        with open(train_src_file_path, 'r', encoding='utf-8') as src_file, \
             open(train_tgt_file_path, 'r', encoding='utf-8') as tgt_file:
            train_src_data = src_file.readlines()
            train_tgt_data = tgt_file.readlines()
    except FileNotFoundError as e:
        logger.error(f"Error loading training dataset: {e}")
        raise

    # Strip newline characters and remove any empty lines
    train_src_data = [line.strip() for line in train_src_data if line.strip()]
    train_tgt_data = [line.strip() for line in train_tgt_data if line.strip()]

    # Log the number of training samples loaded
    logger.info(f"Loaded {len(train_src_data)} training source samples and {len(train_tgt_data)} training target samples")

    # Create the training dataloader
    train_loader = DataLoaderFactory.get_dataloader(
        data_type="text",
        text_data=(train_src_data, train_tgt_data),  # Passing as a tuple for translation task
        tokenizer=tokenizer,
        batch_size=32,
        max_len=128,
        shuffle=True,  # Shuffle training data
        is_translation=True  # Indicate that this is a translation task
    )

    # Log the successful creation of the training dataloader
    logger.info(f"Training dataloader created with {len(train_src_data)} samples")

    # Define paths to the source and target files
    src_file_path = 'Datasets/raw/en-hi/opus.en-hi-dev.en'
    tgt_file_path = 'Datasets/raw/en-hi/opus.en-hi-dev.hi'

    # Load source and target data
    try:
        with open(src_file_path, 'r', encoding='utf-8') as src_file, \
             open(tgt_file_path, 'r', encoding='utf-8') as tgt_file:
            src_data = src_file.readlines()
            tgt_data = tgt_file.readlines()
    except FileNotFoundError as e:
        logger.error(f"Error loading dataset: {e}")
        raise

    # Strip newline characters and remove any empty lines
    src_data = [line.strip() for line in src_data if line.strip()]
    tgt_data = [line.strip() for line in tgt_data if line.strip()]

    # Log the number of samples loaded
    logger.info(f"Loaded {len(src_data)} source samples and {len(tgt_data)} target samples")

    # Create the validation dataloader
    val_loader = DataLoaderFactory.get_dataloader(
        data_type="text",
        text_data=(src_data, tgt_data),  # Passing as a tuple for translation task
        tokenizer=tokenizer,
        batch_size=32,
        max_len=128,
        shuffle=False,  # Usually, we don't shuffle validation data
        is_translation=True  # Indicate that this is a translation task
    )

    # Log the successful creation of the validation dataloader
    logger.info(f"Validation dataloader created with {len(src_data)} samples")

    # Initialize the Transformer model
    model = Transformer(
        num_encoder_layers=6,  # Number of encoder layers
        num_decoder_layers=6,  # Number of decoder layers
        d_model=512,           # Dimensionality of the model
        num_heads=8,           # Number of attention heads
        d_ff=2048,             # Dimensionality of feedforward layers
        src_vocab_size=len(tokenizer.vocab),  # Source vocabulary size
        tgt_vocab_size=len(tokenizer.vocab)   # Target vocabulary size
    )

    # Move the model to the chosen device (GPU if available, otherwise CPU)
    #model.to(device)

    # Set up optimizer and loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index in loss calculation
    optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

    # Set the number of training epochs
    train(model, train_loader, val_loader, epochs=1, optimizer=optimizer, criterion=criterion, tokenizer=tokenizer, d_model=512, warmup_steps=1000)

