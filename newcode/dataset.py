# dataset
import torch

### Now for dataloader abstractions

# Generally set pin_memory to True for gpu use,
#  and overload pin_memory function in data return


# See https://pytorch.org/docs/stable/data.html
class DataLoaderToDevice(torch.utils.data.DataLoader):
    def __init__(self, dataset, device='cpu', **kwargs):
        super().__init__(dataset, **kwargs)
        self.device = device
    def __iter__(self):
        return self
    def __next__(self):
        return super().__next__().to(self.device)


# To be returned by dataset / dataloader
class DataBatchBase:
    # Derived classes need to implement data_keys()
    @staticmethod
    def data_keys():
        pass #Return a list of keys

    def __init__(self, **kwargs):
        keys = self.data_keys()
        for k in keys:
            setattr(self,k,kwargs.get(k,None))

    # Use in dataloader for collate function
    @classmethod
    def collate(cls, point_list):
        keys = cls.data_keys()
        collate_dict = {}
        for k in keys:
            val_list = []
            for p in point_list:
                val_list.append(getattr(p, k))
            collate_dict[k] = torch.cat(val_list, dim=0) #Presumes points already unsqueezed in batch dim
        return DataBatch(**collate_dict)

    # So can set pin_memory to true
    def pin_memory(self):
        keys = self.data_keys()
        for k in keys:
            setattr(self, k, getattr(self,k).pin_memory())

