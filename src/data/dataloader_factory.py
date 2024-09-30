from src.data.text_dataloader import TextDataLoader

class DataLoaderFactory:
    """
    factory class to handle multiple data-types through a unified interface
    new data-types can be added to the dataloader factory in a modular manner
    gives experimental flexibility
    """
    @staticmethod
    def get_dataloader(data_type, **kwargs):
        """
        returns the appropriate dataloader based on the data_type argument
        Params:
            data_type: 'text', 'image', 'pattern'
            kwargs: additional arguments required for the specific data type
        Returns:
            dataloader instance for the specified data type
        """
        if data_type == 'text':
            return DataLoaderFactory._get_text_dataloader(**kwargs)
        elif data_type == 'image':
            return DataLoaderFactory._get_image_dataloader(**kwargs)
        elif data_type == 'pattern':
            return DataLoaderFactory._get_pattern_dataloader(**kwargs)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
        
    @staticmethod
    def _get_text_dataloader(text_data, tokenizer, batch_size=32, max_len=512, shuffle=True):
        """
        returns the dataloader for the text data
        """
        text_loader = TextDataLoader(text_data=text_data, tokenizer=tokenizer, batch_size=batch_size, max_len=max_len, shuffle=shuffle)
        return text_loader.load_data()
    
    @staticmethod
    def _get_image_dataloader(image_dir, batch_size=32, img_size=224, shuffle=True):
        """
        returns the image dataloader (TODO: for future use)
        """
        raise NotImplementedError(f"dataloader not implemented")
    
    @staticmethod
    def _get_pattern_dataloader(data_file, batch_size=32, shuffle=True):
        """
        returns the dataloader for pattern/structured data (TODO: for future use)
        """
        raise NotImplementedError(f"dataloader not implemented")
    