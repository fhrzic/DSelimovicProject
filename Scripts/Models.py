from numpy import float32
import torch
from torch import dropout, nn
import torch.nn.functional as F
import math

class CNN_REG(nn.Module):
    """
    Implementation of first model proposed in paper 
    https://www.researchgate.net/publication/333681910_SHIP_AS_A_WAVE_BUOY_-_ESTIMATING_RELATIVE_WAVE_DIRECTION_FROM_IN-SERVICE_SHIP_MOTION_MEASUREMENTS_USING_MACHINE_LEARNING#fullTextFileContent
    * Input size is based on ship_dataset and is set to 1501

    Args:
        * Max Pooling ratio is uknown so it is make customable
        * Scaler - to linearly increase number of neurons

    """
    def __init__(self, max_pooling_ratio = 2, scaler = 1):
        super(CNN_REG, self).__init__()
        # Calculating number of neurons after the convolutions and max pooling
        self.dense_number = math.floor((math.floor((1501-15+1) / max_pooling_ratio) - 9 + 1) / max_pooling_ratio)
        self.max_pooling_ratio = max_pooling_ratio

        # Layers definition
        self.scaler = scaler
        self.first_conv = torch.nn.Conv2d(1, 48 * self.scaler, kernel_size = (3, 15))
        self.second_conv = torch.nn.Conv2d(48 * self.scaler, 48 * self.scaler, kernel_size = (1,9))
        self.dense = torch.nn.Linear(self.dense_number * 48 * self.scaler, 30 * self.scaler)
        self.dropout = nn.Dropout(p=0.25)
        self.output_layer = torch.nn.Linear(30 * self.scaler, 3)
        
    def forward(self, x):
        # Neural network build
        _out = self.first_conv(x)
        _out = torch.tanh(_out)
        _out = F.max_pool2d(_out, kernel_size = (1, self.max_pooling_ratio))
        _out = self.second_conv(_out)
        _out = torch.tanh(_out)
        _out = F.max_pool2d(_out, kernel_size = (1, self.max_pooling_ratio))

        # Flatten
        _out = _out.view(-1, self.dense_number * 48 * self.scaler)
        _out = self.dense(_out)
        _out = torch.tanh(_out)
        _out = self.dropout(_out)
        _out = self.output_layer(_out)
        _out = torch.tanh(_out)        
        return _out

class SE_Block(nn.Module):
        "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
        def __init__(self, c, r=16):
            super().__init__()
            self.squeeze = nn.AdaptiveAvgPool2d(1)
            self.excitation = nn.Sequential(
                nn.Linear(c, c // r, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(c // r, c, bias=False),
                nn.Sigmoid()
            )

        def forward(self, x):
            bs, c, _, _ = x.shape
            y = self.squeeze(x).view(bs, c)
            y = self.excitation(y).view(bs, c, 1, 1)
            return x * y.expand_as(x)

class MLSTM_CNN(nn.Module):
    """
    Implementation of the second model proposed in paper 
    https://www.researchgate.net/publication/333681910_SHIP_AS_A_WAVE_BUOY_-_ESTIMATING_RELATIVE_WAVE_DIRECTION_FROM_IN-SERVICE_SHIP_MOTION_MEASUREMENTS_USING_MACHINE_LEARNING#fullTextFileContent
    * Input size is based on ship_dataset and is set to 1501    

    Args:
        * Scaler - to linearly increase number of neurons
    """
    def __init__(self, scaler = 1):
        super(MLSTM_CNN, self).__init__()
        self.scaler = scaler

        # Conv layers
        self.first_conv = torch.nn.Conv2d(1, 16 * self.scaler, kernel_size = (3, 11))
        self.second_conv = torch.nn.Conv2d(16 * self.scaler, 32 * self.scaler, kernel_size = (1, 6))
        self.third_conv = torch.nn.Conv2d(32 * self.scaler, 32 * self.scaler, kernel_size = (1, 3))

        # Calc features
        self.first_features = (1501-11 +1 )
        self.second_features = self.first_features - 6 + 1
        self.third_features = self.second_features - 3 + 1  

        # Batch-nrom layers
        self.first_batch_norm = nn.BatchNorm2d(16 * self.scaler)
        self.second_batch_norm = nn.BatchNorm2d(32 * self.scaler)
        self.third_batch_norm = nn.BatchNorm2d(32 * self.scaler)

        # SE Block
        self.first_SE_block = SE_Block(16 * self.scaler, r = 16)
        self.second_SE_block = SE_Block(32 * self.scaler, r = 16)

        # Global Pooling
        self.global_max_pooling = nn.MaxPool2d((1, self.third_features))
        
        ## Dense layers
        self.dense = torch.nn.Linear(64 * self.scaler, 8 * self.scaler)
        self.dropout = nn.Dropout(p=0.1)

        # Output layer
        self.output_layer = torch.nn.Linear(8 * self.scaler, 3)

        # LSTM
        self.lstm = nn.LSTM(input_size = 3, 
                hidden_size = 32 * self.scaler,
                batch_first = True,
                num_layers = 8)
        self.dropout_lstm = nn.Dropout(p=0.1)


    def forward(self, x):
        # LSTM Block
        _out_lstm = torch.squeeze(x, dim = 1)
        _out_lstm = _out_lstm.permute(0, 2, 1)
        
        self.lstm.flatten_parameters()
        _ , (_out_lstm, _) = self.lstm(_out_lstm)
        _out_lstm = _out_lstm[-1]
        _out_lstm = self.dropout_lstm(_out_lstm)

        # Convolution block
        # First block
        _out_cnn = self.first_conv(x)
        _out_cnn = self.first_batch_norm(_out_cnn)
        _out_cnn = torch.tanh(_out_cnn)
        _out_cnn = self.first_SE_block(_out_cnn)

        # Second block
        _out_cnn = self.second_conv(_out_cnn)
        _out_cnn = self.second_batch_norm(_out_cnn)
        _out_cnn = torch.tanh(_out_cnn)
        _out_cnn = self.second_SE_block(_out_cnn)
        

        # Third block
        _out_cnn = self.third_conv(_out_cnn)
        _out_cnn = self.third_batch_norm(_out_cnn)
        _out_cnn = torch.tanh(_out_cnn)

        # Global max pooling
        _out_cnn = self.global_max_pooling(_out_cnn)
        _out_cnn = torch.squeeze(_out_cnn, dim = 3)
        _out_cnn = torch.squeeze(_out_cnn, dim = 2)

        # Concat
        _out = torch.cat((_out_lstm, _out_cnn), dim = 1)
        
        # Head part
        _out = self.dense(_out)
        _out = torch.tanh(_out)
        _out = self.dropout(_out)
        # Output
        _out = self.output_layer(_out)
        _out = torch.tanh(_out)
    
        return _out

    
class SP_NN(nn.Module):
    """
    Implementation of the third model proposed in paper 
    https://www.researchgate.net/publication/333681910_SHIP_AS_A_WAVE_BUOY_-_ESTIMATING_RELATIVE_WAVE_DIRECTION_FROM_IN-SERVICE_SHIP_MOTION_MEASUREMENTS_USING_MACHINE_LEARNING#fullTextFileContent
    * Input size is based on ship_dataset and is set to 1501
    Args:
        * Max Pooling ratio is uknown so it is make customable
        * Scaler - to linearly increase number of neurons
    """

    def __init__(self, max_pooling_ratio = 2, scaler = 1):
        # Init
        super(SP_NN, self).__init__()
        self.scaler = scaler
        self.pool_ratio = max_pooling_ratio
        self.dense_number = math.floor((((1501-25+1) - 1 + 1) / self.pool_ratio)) * 3

        # Conv
        self.first_conv = torch.nn.Conv2d(1, 64 * self.scaler, kernel_size = (1, 25))
        self.second_conv = torch.nn.Conv2d(64 * self.scaler, 128 * self.scaler, kernel_size = (3, 1))

        # SE Block
        self.se_block = SE_Block(64 * self.scaler, r = 16)

        # Dense
        self.dense = torch.nn.Linear(self.dense_number * 128 * self.scaler, 30 * self.scaler)
        self.dropout = nn.Dropout(p=0.25)

        # Output layer
        self.output_layer = torch.nn.Linear(30 * self.scaler, 3)

    def forward(self, x):
        # Forward pass

        # First block
        _out = self.first_conv(x)
        _out = torch.tanh(_out)
        _out = self.se_block(_out)

        # Second block
        _out = self.second_conv(_out)
        _out = torch.tanh(_out)
        
        # Min max and avg pooling
        _out_min = torch.mul(_out, -1)
        _out_min = F.max_pool2d(_out_min, kernel_size = (1, self.pool_ratio))
        _out_min = torch.mul(_out_min, -1)
        

        _out_max = F.max_pool2d(_out, kernel_size = (1, self.pool_ratio))
        _out_avg = F.avg_pool2d(_out, kernel_size = (1, self.pool_ratio))

        # Concat and flatten
        _out = torch.cat((_out_max, _out_min, _out_avg), dim = 3)
        _out = _out.view(-1, self.dense_number * 128 * self.scaler)

        # Dense and head
        _out = self.dense(_out)
        _out = torch.tanh(_out)
        _out = self.dropout(_out)

        _out = self.output_layer(_out)
        _out = torch.tanh(_out)

        return _out