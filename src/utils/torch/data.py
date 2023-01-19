import torch

import collections
import contextlib
import re

from typing import Callable, Dict, Optional, Tuple, Type, Union
from torch._six import string_classes

np_str_obj_array_pattern = re.compile(r'[SaUO]')

from torch.utils import data

# Turn off the warning for num_workers=0
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")

class TorchLarcvDataset(data.IterableDataset):

    def __init__(self, larcv_dataset, global_batch_size):
        super(TorchLarcvDataset).__init__()
        self.ds = larcv_dataset
        self.global_batch_size = global_batch_size

    def __iter__(self):
        for i, batch in enumerate(self.ds):
            if i < len(self.ds):
                yield batch

    def image_size(self):
        return self.ds.image_size()

    def image_meta(self):
        return self.ds.image_meta

    def __len__(self):
        return 10
        return int(len(self.ds) / self.global_batch_size)



def custom_convert(data, device):

    """
    This is modified from the default version to move data to the device.
    """

    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        # array of string classes and object
        if elem_type.__name__ == 'ndarray' \
                and np_str_obj_array_pattern.search(data.dtype.str) is not None:
            return data
        return torch.as_tensor(data).to(device)
    elif isinstance(data, collections.abc.Mapping):
        try:
            return elem_type({key: custom_convert(data[key], device=device) for key in data})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: custom_convert(data[key], device=device) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(custom_convert(d, device=device) for d in data))
    elif isinstance(data, tuple):
        return [custom_convert(d, device=device) for d in data]  # Backwards compatibility.
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, string_classes):
        try:
            return elem_type([custom_convert(d, device=device) for d in data])
        except TypeError:
            # The sequence type may not support `__init__(iterable)` (e.g., `range`).
            return [custom_convert(d, device=device) for d in data]
    else:
        return data


def create_torch_larcv_dataloader(larcv_ds, global_batch_size, device=None):

    ids =  TorchLarcvDataset(larcv_ds, global_batch_size)

    if device is not None:
        target_collate_fn = lambda x : custom_convert(x, device)
    else:
        target_collate_fn = data.default_convert

    torch_dl = data.DataLoader(ids,
        num_workers    = 0,
        batch_size    = None,
        batch_sampler = None,
        pin_memory    = False,
        collate_fn    = target_collate_fn
    )

    return torch_dl
