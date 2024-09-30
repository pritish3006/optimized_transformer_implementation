import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer                                          # using BertTokenizer from HF tokenizers

class TextDataset(Dataset):
    """
    dataset class for tokenized text data using BertTokenizer
    """
    def __init__(self, text_data, tokenizer, max_len=512):
        self.text_data = text_data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text_data)
    
    def __getitem__(self, idx):
        text = self.text_data[idx]
        tokens = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        input_ids = tokens['input_ids'].squeeze(0)                              # remove the batch dimension          
        attention_mask = tokens['attention_mask'].squeeze(0)                    # remove the batch dimensions from the attention mask
        return input_ids, attention_mask
    
    class TextDataLoader:
        """
        dataloader subclass for text data using BertTokenizer
        """
        def __init__(self, text_data, batch_size=32, max_len=512, shuffle=True):
            self.text_data = text_data
            self.batch_size = batch_size
            self.max_len = max_len
            self.shuffle = shuffle
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        def load_data(self):
            """
            overriding the load_data() method defined in the BaseDataLoader class
            """
            dataset = TextDataset(self.text_data, self.tokenizer, max_len=self.max_len)
            return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        