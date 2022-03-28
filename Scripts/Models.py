import torch
from torch import nn
import torch.nn.functional as F
import math

class CNN_REG(nn.Module):
    """
    Implementation of first model proposed in paper 
    https://www.researchgate.net/publication/333681910_SHIP_AS_A_WAVE_BUOY_-_ESTIMATING_RELATIVE_WAVE_DIRECTION_FROM_IN-SERVICE_SHIP_MOTION_MEASUREMENTS_USING_MACHINE_LEARNING#fullTextFileContent
    
    Args:
        * Max Pooling ratio is uknown so it is make customable
        * Input size is based on ship_dataset and is set to 1501
    """
    def __init__(self, max_pooling_ratio = 2):
        super(CNN_REG, self).__init__()
        # Calculating number of neurons after the convolutions and max pooling
        self.dense_number = math.floor((math.floor((1501-15+1) / max_pooling_ratio) - 9 + 1) / max_pooling_ratio)
        self.max_pooling_ratio = max_pooling_ratio

        # Layers definition
        self.first_conv = torch.nn.Conv2d(1, 48, kernel_size = (3, 15))
        self.second_conv = torch.nn.Conv2d(48, 48, kernel_size = (1,9))
        self.dense = torch.nn.Linear(self.dense_number * 48, 30)
        self.dropout = nn.Dropout(p=0.25)
        self.output_layer = torch.nn.Linear(30, 3)
        
    def forward(self, x):
        # Neural network build
        _out = self.first_conv(x)
        _out = torch.tanh(_out)
        _out = F.max_pool2d(_out, kernel_size = (1, self.max_pooling_ratio))
        _out = self.second_conv(_out)
        _out = torch.tanh(_out)
        _out = F.max_pool2d(_out, kernel_size = (1, self.max_pooling_ratio))
        
        # Flatten
        _out = _out.view(-1, self.dense_number * 48)
        _out = self.dense(_out)
        _out = torch.tanh(_out)
        _out = self.dropout(_out)
        _out = self.output_layer(_out)
        _out = torch.tanh(_out)
        
        return _out