import torch
import torch.nn as nn
from torch.optim import Adam
from src.models.transformer import Transformer
from src.data.dataset import load_data                                          # TODO: implement the modular dataloader
from src.utils.helpers import create_look_ahead_mask, create_masked_padding, save_checkpoint    # TODO: implement save_checkpoint
import logging

# setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_epoch(model, data_loader, optimizer, criterion, device):
    model.train()                                                               # set the model to training mode
    total_loss = 0.0                                                            # init training loss @ 0.0

    for batch_size, (src, tgt) in enumerate(data_loader):
        src, tgt = src.to(device), tgt.to(device)

        # input and target sequences for the decoder
        tgt_input = tgt[:, :-1]                                                 # exclude the last token for decoder input
        tgt_output = tgt[:, 1:]                                                 # shift by one position for target output

        # generate masks
        src_mask = create_masked_padding(src)                                   # TODO: look into default mask and play around with different masks
        tgt_padding_mask = create_masked_padding(tgt_input)                     # TODO: look into default mask val and play around with different masks
        tgt_look_ahead_mask = create_look_ahead_mask(tgt_input.size(1))
        tgt_mask = tgt_padding_mask & tgt_look_ahead_mask                       # combining padding and look ahead masks using boolean AND operation

        # forward pass
        optimizer.zero_grad()
        output = model(src, tgt_input, src_mask, tgt_mask)

        # compute the loss
        loss = criterion(output.view(-1, output.size(-1)), tgt_output.view(-1))
        total_loss += loss.item()

        # backpropagation and optimization
        loss.backward()
        optimizer.step()
    
    return total_loss / len(data_loader)

def validate_epoch(model, data_loader, criterion, device):
    model.eval()                                                                # set the model to eval mode
    total_loss = 0.0                                                            # set the eval loss to 0.0

    with torch.no_grad():
        for src, tgt in data_loader:
            src, tgt = src.to(device), tgt.to(device)

            # input and target sequences
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # generate masks
            src_mask = create_masked_padding(src)
            tgt_padding_mask = create_masked_padding(tgt_input)
            tgt_look_ahead_mask = create_look_ahead_mask(tgt_input.size(1))
            tgt_mask = tgt_padding_mask & tgt_look_ahead_mask

            # forward pass
            output = model(src, tgt_input, src_mask, tgt_mask)
            loss = criterion(output.view(-1, output.size(-1)), tgt_output.view(-1))
            total_loss += loss.item()

    return total_loss / len(data_loader)        

def train(model, train_loader, val_loader, epochs, optimizer, criterion, device, checkpoint_dir="checkpoints"):
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)

        logger.info(f"epoch {epoch+1}/{epochs}, train loss: {train_loss: .4f}, validation loss: {val_loss:.4f}")

        # save model checkpoint
        save_checkpoint(model, optimizer, epoch, train_loss, checkpoint_dir)

if __name__ == "__main__":
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')   # moving the trainig to GPU if available
    
    # load the dataset
    train_loader, val_loader = load_data(data_type="text", batch_size=32)

    # initialize model, optimizer, loss
    model = Transformer(                                                        
        num_encoder_layers=6,                                                   # 6 as defined in the AIAYN paper
        num_decoder_layers=6,                                                   # 6 as defined in the AIAYN paper
        d_model=512, 
        num_heads=8,
        d_ff=2048,
        src_vocab_size=10000,
        tgt_vocab_size=10000                                  
    )

    # move the model to the chosen device
    model.to(device)

    # setting up optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = Adam(model.parameter(), lr=0.001)

    # set the model training epochs and train the model
    train(model, train_loader, val_loader, epochs=10, optimizer=optimizer, criterion=criterion, device=device)
    
