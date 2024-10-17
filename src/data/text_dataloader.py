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
            return src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask
        else:
            return src_input_ids, src_attention_mask

class TextDataLoader:
    """
    dataloader subclass for text data using BertTokenizer
    """
    def __init__(self, src_text_data, tgt_text_data, batch_size=32, max_len=512, shuffle=True):
        # Store input data and parameters
        self.tgt_text_data = tgt_text_data
        self.batch_size = batch_size
        self.max_len = max_len
        self.shuffle = shuffle

        
        # Initialize tokenizers for source and target languages
        self.src_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        self.tgt_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

    def load_data(self):
        """
        overriding the load_data() method defined in the BaseDataLoader class
        """
        # Create TextDataset instances for source and target data
        src_dataset = TextDataset(self.src_text_data, self.src_tokenizer, max_len=self.max_len)
        tgt_dataset = TextDataset(self.tgt_text_data, self.tgt_tokenizer, max_len=self.max_len)
        
        # Combine source and target datasets into a single Tensor Dataset
        dataset = torch.utils.data.TensorDataset(
            *[torch.stack(tensors) for tensors in zip(src_dataset, tgt_dataset)]
        )
        
        # Create and return a DataLoader with the combined dataset
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)        

