from numpy import float32
import torch
from torch import Tensor, dropout, nn
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
        self.first_conv = torch.nn.Conv2d(1, 48 * self.scaler, kernel_size = (1, 25))
        self.second_conv = torch.nn.Conv2d(48 * self.scaler, 48 * self.scaler, kernel_size = (3, 1))

        # SE Block
        self.se_block = SE_Block(64 * self.scaler, r = 16)

        # Dense
        self.dense = torch.nn.Linear(self.dense_number * 48 * self.scaler, 30 * self.scaler)
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
        _out = _out.view(-1, self.dense_number * 48 * self.scaler)

        # Dense and head
        _out = self.dense(_out)
        _out = torch.tanh(_out)
        _out = self.dropout(_out)

        _out = self.output_layer(_out)
        _out = torch.tanh(_out)

        return _out

class Positional_Encoding(nn.Module):
    """
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html#define-the-model
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Normalization_block(nn.Module):
    """
    https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec#d554
    """
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class Attention_block(nn.Module):
    def __init__(self, number_of_heads=4, number_of_embedings = 128):
        super(Attention_block, self).__init__()
        
        # First mha
        self.first_mha = torch.nn.MultiheadAttention(embed_dim = number_of_embedings, 
                                                    num_heads = number_of_heads)
        self.first_normalize = Normalization_block(d_model = number_of_embedings)

        # Second mha
        self.second_mha = torch.nn.MultiheadAttention(embed_dim = number_of_embedings, 
                                                    num_heads = number_of_heads)
        self.second_normalize = Normalization_block(d_model = number_of_embedings)

    def forward(self, input_1):
        _out_1, _ = self.first_mha(input_1, input_1, input_1)
        
        _input_2 = input_1 + _out_1
        _input_2 = self.first_normalize(_input_2)

        _out_2, _ = self.second_mha(_input_2, _input_2, _input_2)
        
        _input_3 = _input_2 + _out_2
        _input_3 = self.second_normalize(_input_3)

        return _input_3

class head_block(nn.Module):
    def __init__(self, number_of_neurons: list):
        super(head_block, self).__init__()
        # Dense layers
        
        self.dense_1 = torch.nn.Linear(number_of_neurons[0], number_of_neurons[1])
        self.dense_2 = torch.nn.Linear(number_of_neurons[1], number_of_neurons[2])
        self.dense_3 = torch.nn.Linear(number_of_neurons[2], number_of_neurons[3])
        self.dense_4 = torch.nn.Linear(number_of_neurons[3], number_of_neurons[4])
        self.output = torch.nn.Linear(number_of_neurons[4], 1)
        
        self.a_1 = torch.nn.ReLU()
        self.a_2 = torch.nn.ReLU()
        self.a_3 = torch.nn.ReLU()
        self.a_4 = torch.nn.ReLU()
        self.a_o = torch.nn.Sigmoid()

        self.drop_1 = torch.nn.Dropout(p = 0.1)
        self.drop_2 = torch.nn.Dropout(p = 0.1)
        self.drop_3 = torch.nn.Dropout(p = 0.1)
        self.drop_4 = torch.nn.Dropout(p = 0.1)

        self.bn_1 = torch.nn.BatchNorm1d(number_of_neurons[1])
        self.bn_2 = torch.nn.BatchNorm1d(number_of_neurons[2])
        self.bn_3 = torch.nn.BatchNorm1d(number_of_neurons[3])
        self.bn_4 = torch.nn.BatchNorm1d(number_of_neurons[4])

    def forward(self, x):
        # First block
        _out = self.dense_1(x)
        _out = self.a_1(_out)
        _out = self.bn_1(_out)
        _out = self.drop_1(_out)

        # Second block
        _out = self.dense_2(_out)
        _out = self.a_2(_out)
        _out = self.bn_2(_out)
        _out = self.drop_2(_out)
        
        # Third block
        _out = self.dense_3(_out)
        _out = self.a_3(_out)
        _out = self.bn_3(_out)
        _out = self.drop_3(_out)

        # Four block
        _out = self.dense_4(_out)
        _out = self.a_4(_out)
        _out = self.bn_4(_out)
        _out = self.drop_4(_out)

        # Output block
        _out = self.output(_out)
        _out = self.a_o(_out)

        return _out



class ATT_NN(nn.Module):
    def __init__(self, number_of_blocks = 1, number_of_heads=4, embeding_scale = 30, 
                number_of_embedings = 128, batch_size = 32, head_neurons = [4096, 4096, 2048, 512, 1]):
        # Init
        super(ATT_NN, self).__init__()

        # Embedding
        self.create_embeding = torch.nn.Conv2d(1, number_of_embedings , kernel_size = (3, embeding_scale))
        self.seq_length = 1501-embeding_scale + 1 
        self.number_of_embedings = number_of_embedings
        self.positional_encoding = Positional_Encoding(d_model = number_of_embedings, 
                                                        max_len = self.seq_length)
        
       
        # Multiheads attentins
        self.block_list = nn.ModuleList()
        for i in range(number_of_blocks):
            self.block_list.append(Attention_block(number_of_heads, number_of_embedings))                            

        # Decision heads
        self.head_Hs = head_block([self.seq_length * self.number_of_embedings, 4096, 4096, 2048, 512])
        self.head_Ts = head_block([self.seq_length * self.number_of_embedings, 4096, 4096, 2048, 512])
        self.head_Dp = head_block([self.seq_length * self.number_of_embedings, 4096, 4096, 2048, 512])

    def forward(self, x):
        # Fix input and create embeding
        _out = self.create_embeding(x)
        print(_out.shape)        
        _out = torch.squeeze(_out, dim = 2)
        _out = _out.permute(2,0,1)
        _out = self.positional_encoding(_out)
        
        # Go trough attentions
        for _i, _att_block in enumerate(self.block_list):
            _out = _att_block(_out)

        # Rearange output
        _out = _out.permute(1,2,0)
        _out = torch.unsqueeze(_out, dim = 2)
        
        # Flatten
        _out = _out.reshape(-1, self.seq_length * self.number_of_embedings)

        
        # Heads parts
        _head_Hs = self.head_Hs(_out)
        _head_Ts = self.head_Ts(_out)
        _head_Dp = self.head_Dp(_out)

        _out = torch.cat((_head_Hs, _head_Ts, _head_Dp), dim = 1)
        return _out