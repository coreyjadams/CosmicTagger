import torch

from torch.utils import data

class TorchLarcvDataset(data.IterableDataset):

    def __init__(self, larcv_dataset, global_batch_size):
        super(TorchLarcvDataset).__init__()
        self.ds = larcv_dataset
        self.global_batch_size = global_batch_size
    def __iter__(self):
        for batch in self.ds:
            yield batch

    def __len__(self):
        return 10
        return int(len(self.ds) / self.global_batch_size)


def create_torch_larcv_dataloader(larcv_ds, global_batch_size):

    ids =  TorchLarcvDataset(larcv_ds, global_batch_size)

    torch_dl = data.DataLoader(ids, 
        num_workers    = 0, 
        batch_size    = None, 
        batch_sampler = None,
        pin_memory    = True)

    return torch_dl