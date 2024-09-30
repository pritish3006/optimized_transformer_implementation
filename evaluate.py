import torch
from torch.nn import CrossEntropyLoss
from torchmetrics.functional import bleu_score
from transformers import BertTokenizer
from src.models.transformer import Transformer
from src.data.dataloader_factory import DataLoaderFactory
from src.utils.helpers import create_masked_padding, create_look_ahead_mask
import logging
import os

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_checkpoint(model, checkpoint_path, device):
    """
    Load model weights from a saved checkpoint.
    """
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model weights from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found.")
    
    return model

def compute_accuracy(predicted_tokens, target_tokens):
    """
    Computes the accuracy of the model's predictions by comparing the predicted tokens to the target tokens.
    """
    correct_predictions = (predicted_tokens == target_tokens).float().sum()
    total_predictions = target_tokens.numel()
    return correct_predictions / total_predictions

def evaluate_model(model, data_loader, criterion, device, verbose=False):
    """
    Evaluates the performance of the model on a validation dataset.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_accuracy = 0.0
    total_bleu_score = 0.0
    count = 0

    with torch.no_grad():  # Disable gradient computation during evaluation
        for i, (src, tgt) in enumerate(data_loader):
            src, tgt = src.to(device), tgt.to(device)

            tgt_input = tgt[:, :-1]  # Exclude the last token for decoder input
            tgt_output = tgt[:, 1:]  # Shift by one position for target output

            src_mask = create_masked_padding(src)
            tgt_padding_mask = create_masked_padding(tgt_input)
            tgt_look_ahead_mask = create_look_ahead_mask(tgt_input.size(1))
            tgt_mask = tgt_padding_mask & tgt_look_ahead_mask

            output = model(src, tgt_input, src_mask, tgt_mask)

            loss = criterion(output.view(-1, output.size(-1)), tgt_output.view(-1))
            total_loss += loss.item()

            predicted_tokens = torch.argmax(output, dim=-1)
            accuracy = compute_accuracy(predicted_tokens, tgt_output)
            total_accuracy += accuracy.item()

            predicted_tokens_list = [pred.tolist() for pred in predicted_tokens]
            target_tokens_list = [tgt.tolist() for tgt in tgt_output]
            bleu = bleu_score(predicted_tokens_list, target_tokens_list)
            total_bleu_score += bleu.item()

            # Log batch metrics
            logger.info(f"Batch {i+1}/{len(data_loader)} - Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}, BLEU: {bleu:.4f}")

            # Verbose output: Print predictions for each batch
            if verbose:
                logger.info(f"Predicted tokens: {predicted_tokens_list}")
                logger.info(f"Target tokens: {target_tokens_list}")

            count += 1

    avg_loss = total_loss / count
    avg_accuracy = total_accuracy / count
    avg_bleu_score = total_bleu_score / count

    return {
        "loss": avg_loss,
        "accuracy": avg_accuracy,
        "bleu_score": avg_bleu_score
    }


if __name__ == "__main__":
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    
    model = Transformer(
        num_encoder_layers=6,                                                   
        num_decoder_layers=6,                                                   
        d_model=512, 
        num_heads=8,
        d_ff=2048,
        src_vocab_size=10000,
        tgt_vocab_size=10000
    )
    
    model.to(device)

    # Load the best model checkpoint
    checkpoint_path = "checkpoints/best_model.pth"
    model = load_model_checkpoint(model, checkpoint_path, device)

    # Load the validation data
    val_loader = DataLoaderFactory.get_dataloader(
        data_type="text",
        text_data=["Validation sentence 1", "Validation sentence 2"],  # Placeholder data
        tokenizer=BertTokenizer.from_pretrained('bert-base-uncased'),
        batch_size=32,
        max_len=128
    )

    # Initialize the loss function
    criterion = CrossEntropyLoss(ignore_index=0)

    # Evaluate model performance
    eval_metrics = evaluate_model(model, val_loader, criterion, device, verbose=True)

    # Print final evaluation metrics
    logger.info(f"Validation Loss: {eval_metrics['loss']:.4f}")
    logger.info(f"Validation Accuracy: {eval_metrics['accuracy']:.4f}")
    logger.info(f"BLEU Score: {eval_metrics['bleu_score']:.4f}")
