# dataset classes
import torch

############################################################################
# General Classes
############################################################################

# See https://pytorch.org/docs/stable/data.html
class DataLoaderToDevice(torch.utils.data.DataLoader):
    def __init__(self, dataset, device='cpu', **kwargs):
        super().__init__(dataset, **kwargs)
        self.device = device
    def __iter__(self):
        self.iterator = super().__iter__()
        return self
    def __next__(self):
        return self.iterator.__next__().to(self.device)


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
        return cls(**collate_dict)

    # So can set pin_memory to true
    def pin_memory(self):
        keys = self.data_keys()
        for k in keys:
            setattr(self, k, getattr(self,k).pin_memory())
        return self

    def to(self, device):
        keys = self.data_keys()
        for k in keys:
            setattr(self, k, getattr(self,k).to(device))
        return self

    def cuda(self):
        return self.to('cuda')

    def cpu(self):
        return self.to('cpu')

    def __getitem__(self, key):
        return getattr(self,key)

    def get(self, key, default=None):
        try:
            return self.__getitem__(key)
        except AttributeError:
            return default



