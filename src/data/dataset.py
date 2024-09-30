from torch.utils.data import DataLoader

class BaseDataLoader:
    """
    base abstract class for modular data-loader
    sub-classes will implement `load_data` method to handle specific data types
    """
    def __init__(self, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def load_data(self):
        raise NotImplementedError("Subclasses will implement this method")                  # this is intended to be overridden by specific type based subclass.
    
    def create_dataloader(self, dataset):
        """
        create pytorch dataloader from the given dataset
        """
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)