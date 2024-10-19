import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, get_linear_schedule_with_warmup
from src.models.transformer import Transformer
from src.data.text_dataloader import TextDataLoader
from src.data.dataloader_factory import DataLoaderFactory
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
from torch.utils.data import Dataset
from accelerate import Accelerator
from accelerate.utils import set_seed
import nltk

nltk.download('punkt')

# Initialize wandb
wandb.init(project="transformer_training", config={
    "learning_rate": 1e-4,
    "batch_size": 32,
    "epochs": 10,
})

# initialize accelerator
accelerator = Accelerator()

# Setup model checkpointing
checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# implementing a custom callback to log all metrics
class MultiMetricCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if state.is_world_process_zero:
            print(f"epoch {state.epoch}: metrics: {metrics}")
            for metric_name, value in metrics.items():
                if metric_name != "epoch":
                    wandb.log({f"eval/{metric_name}": value}, step=state.global_step)


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

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metrics = calculate_metrics(torch.tensor(logits), torch.tensor(labels), tokenizer) 

    # compute a weighted combined score for best model performance
    combined_score = (
        0.3 * metrics['bleu'] +
        0.1 * metrics['rouge']['rouge-l']['f'] +
        0.25 * metrics['meteor'] +
        0.35 * (1 / metrics['perplexity'])  # Invert perplexity as lower is better
    )

    metrics['combined_score'] = combined_score
    return metrics


if __name__ == "__main__":
    set_seed(42)
    device = accelerator.device
    
    # Load the tokenizer (BPE tokenizer within AutoTokenizer)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')  # GPT-2 uses BPE tokenization
    tokenizer.pad_token = tokenizer.eos_token  # Set the padding token to the end-of-sequence token
    
    with open('Datasets/raw/en-hi/opus.en-hi-train.en', 'r') as f:
        src_text_data = f.readlines()
    with open('Datasets/raw/en-hi/opus.en-hi-train.hi', 'r') as f:
        tgt_text_data = f.readlines()

    train_dataset = TextDataLoader(
        src_text_data=src_text_data,
        tgt_text_data=tgt_text_data,
        tokenizer=tokenizer,
        batch_size=32,
        max_len=128,
        shuffle=True
    ).load_data()

    val_dataset = TextDataLoader(
        src_text_data=open('Datasets/raw/en-hi/opus.en-hi-dev.en', 'r').readlines(),
        tgt_text_data=open('Datasets/raw/en-hi/opus.en-hi-dev.hi', 'r').readlines(),
        tokenizer=tokenizer,
        batch_size=32,
        max_len=128,
        shuffle=False
    ).load_data()

    # Initialize the Transformer model
    model = Transformer(
        num_encoder_layers=6,  # Number of encoder layers
        num_decoder_layers=6,  # Number of decoder layers
        d_model=512,           # Dimensionality of the model
        num_heads=8,           # Number of attention heads
        d_ff=2048,             # Dimensionality of feedforward layers
        src_vocab_size=tokenizer.vocab_size,  # Source vocabulary size
        tgt_vocab_size=tokenizer.vocab_size   # Target vocabulary size
    )

    # Move the model to the chosen device (GPU if available, otherwise CPU)
    model.to(device)

    # defining the optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=1e-5)

    # prepare the training setup with accelerator
    model, optimizer, train_dataset, val_dataset = accelerator.prepare(model, optimizer, train_dataset, val_dataset)


    num_training_steps = len(train_dataset) * 1 // 32                                                                               # for 1 epoch, batch size 32 - hardcoded
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=num_training_steps)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="results",
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        warmup_steps=500,
        weight_decay=0.001,
        logging_dir="logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="combined_score",
        load_best_model_at_end=True,
        report_to="wandb"
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        optimizers=(optimizer, scheduler),
        compute_metrics=compute_metrics,
        callbacks=[MultiMetricCallback()]
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")

    # Save the model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained("model_save")
    trainer.save_model("trainer_save")

    # End wandb run
    wandb.finish()