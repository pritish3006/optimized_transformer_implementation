import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer                                          # using BertTokenizer from HF tokenizers

class TextDataset(Dataset):
    """
    dataset class for tokenized text data using BertTokenizer
    """
    def __init__(self, src_text_data, tgt_text_data, tokenizer, max_len=512):
        self.src_text_data = src_text_data
        self.tgt_text_data = tgt_text_data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.src_text_data)
    
    def __getitem__(self, idx):
        src_text = self.src_text_data[idx]
        src_tokens = self.tokenizer(
            src_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        src_input_ids = src_tokens['input_ids'].squeeze(0)                      # remove the batch dimension
        src_attention_mask = src_tokens['attention_mask'].squeeze(0)            # remove the batch dimensions from the attention

        if self.tgt_text_data is not None:
            tgt_text = self.tgt_text_data[idx]
            tgt_tokens = self.tokenizer(
                tgt_text,
                truncation=True,
                padding='max_length',
                max_length=self.max_len,
                return_tensors='pt'
            )
            tgt_input_ids = tgt_tokens['input_ids'].squeeze(0)
            tgt_attention_mask = tgt_tokens['attention_mask'].squeeze(0)
            return src_input_ids, tgt_input_ids
        else:
            return src_input_ids

class TextDataLoader:
    """
    dataloader subclass for text data using BertTokenizer
    """
    def __init__(self, src_text_data, tgt_text_data, tokenizer, batch_size=32, max_len=512, shuffle=True):
        # Store input data and parameters
        self.src_text_data = src_text_data
        self.tgt_text_data = tgt_text_data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len
        self.shuffle = shuffle

    def load_data(self):
        """
        overriding the load_data() method defined in the BaseDataLoader class
        """
        # Create TextDataset instance for source and target data
        dataset = TextDataset(self.src_text_data, self.tgt_text_data, self.tokenizer, max_len=self.max_len)
        
        # create and return a DataLoader with the dataset
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)

