import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class TextDataset(Dataset):
    """
    dataset class for tokenized text data using AutoTokenizer
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
        src_tokens = self.tokenizer.encode_plus(
            src_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        src_input_ids = src_tokens['input_ids'].squeeze(0)
        src_attention_mask = src_tokens['attention_mask'].squeeze(0)

        if self.tgt_text_data is not None:
            tgt_text = self.tgt_text_data[idx]
            tgt_tokens = self.tokenizer.encode_plus(
                tgt_text,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            tgt_input_ids = tgt_tokens['input_ids'].squeeze(0)
            tgt_attention_mask = tgt_tokens['attention_mask'].squeeze(0)
            return src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask
        else:
            return src_input_ids, src_attention_mask

class TextDataLoader:
    """
    dataloader subclass for text data using AutoTokenizer
    """
    def __init__(self, src_text_data, tgt_text_data, tokenizer, batch_size=32, max_len=512, shuffle=True):
        # Store input data and parameters
        self.src_text_data = src_text_data
        self.tgt_text_data = tgt_text_data
        self.batch_size = batch_size
        self.max_len = max_len
        self.shuffle = shuffle
        self.tokenizer = tokenizer

    def load_data(self):
        """
        overriding the load_data() method defined in the BaseDataLoader class
        """
        # Create TextDataset instance for source and target data
        dataset = TextDataset(self.src_text_data, self.tgt_text_data, self.tokenizer, max_len=self.max_len)
        
        # create and return a DataLoader with the dataset
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)
