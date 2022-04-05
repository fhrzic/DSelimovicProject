import random
import torch
from torch import nn as nn
from math import floor

class augmentation_model_per_row(nn.Module):
    """
        Augmentation that shuffles the input data signals in random manner - each separately
    """

    def __init__(self, input_signal_length, patch_size):
        super().__init__()
        self.patch_size = floor(input_signal_length / patch_size)


    def forward(self, input_batch):
        _t = torch.Tensor.split(input_batch, 1, dim = 0)
        _t = list(_t)
        _t = [self.shuffle_signal(_x) for _x in _t]
        _t = torch.cat(_t, dim = 3)
        return _t

    # Create function that shuffles signal
    def shuffle_signal(self, row_tensor):
        _t = torch.Tensor.split(row_tensor, self.patch_size, dim = 3)
        _t = list(_t)
        random.shuffle(_t)
        return torch.cat(_t, dim = 3)


class augmentation_model_per_batch(nn.Module):
    """
        Augmentation that shuffles the input data signals in random manner - all rows on the same
    """

    def __init__(self, input_signal_length, patch_size):
        super().__init__()
        self.patch_size = floor(input_signal_length / patch_size)

    def forward(self, input_batch):
        return self.shuffle_signal(input_batch)

    # Create function that shuffles signal
    def shuffle_signal(self, row_tensor):
        _t = torch.Tensor.split(row_tensor, self.patch_size, dim = 3)
        _t = list(_t)
        random.shuffle(_t)
        return torch.cat(_t, dim = 3)
