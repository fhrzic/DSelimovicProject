import os
import sys
import numpy as np
from collections import namedtuple
import torch
from Models import *
import pandas as pd
import xlsxwriter
import ast


model_params = namedtuple(
    'model_params',
    'name, gpu, max_pooling_rate, scaler')

model_params_att = namedtuple(
    'model_params_att',
    'name, gpu, blocks, heads, emb_scale, num_emb, heads_shape')

model_params_head = namedtuple(
    'model_params',
    'name, gpu, neurons, module',
)

class evaluate_model():
    """
    Class for models evaluation

    Args: 
        * path, str, valid path to model ".pth" file.
    """

    def __init__(self, weights_path: str = None):
        # Grab params
        self.model_params = self.__get_params(weights_path)

        
        # Use cuda if possible        
        self.use_cuda = torch.cuda.is_available()        
        
        self.device = torch.device("cuda" if self.use_cuda and self.model_params.gpu else "cpu")

        self.model = self.init_model()
        
        self.__load_weights(weights_path)
        self.model.eval()


    def init_model(self):

        assert self.model_params.name in ["CNN_REG", "MLSTM_CNN", "SP_NN", "ATT_NN", "HEAD_NN"], f"Wrong model name, got: {self.model_params.name}"
        print("**************************************")
        print(f"USING MODEL: {self.model_params.name}")

        if self.model_params.name == "CNN_REG":
            _model = CNN_REG(max_pooling_ratio = self.model_params.max_pooling_rate, 
                scaler = self.model_params.scaler)
        
        if self.model_params.name == "MLSTM_CNN":
            _model = MLSTM_CNN(scaler = self.model_params.scaler)

        if self.model_params.name == "SP_NN":
            _model = SP_NN(max_pooling_ratio = self.model_params.max_pooling_rate, 
                scaler = self.model_params.scaler)

        if self.model_params.name == "ATT_NN":
            _model = ATT_NN(self.model_params.blocks, self.model_params.heads, self.model_params.emb_scale,
                            self.model_params.num_emb, self.model_params.heads_shape)

        if self.model_params.name == 'HEAD_NN':
            _model = HEAD_NN(self.model_params.neurons)

        if self.model_params.gpu:
            print(f"USING GPU: {self.device}")
            _model = _model.to(self.device)
        else:
            _model.to(self.device)
        return _model

    def __get_params(self, path):
        # Part that is exctracting params from the path file
        _ , file = os.path.split(path)
        _splitted = file.split(':')
        _name = _splitted[0]
        _cuda = True # OVDJE BIRAŠ ZA CUDU-JAKOOO BITNOOO!!!
        
        if _name == 'HEAD_NN':
            _params = model_params_head(_name, _cuda,  ast.literal_eval(_splitted[2]), _splitted[1])
            return _params
        
        if _name == 'ATT_NN':
            _params = model_params_att(_name, _cuda, int(_splitted[1]), 
                                    int(_splitted[2]), int(_splitted[3]), 
                                    int(_splitted[4]), ast.literal_eval(_splitted[5].split('_')[0]))
        else:
            if _splitted[1] != 'na':
                _splitted[1] = int(_splitted[1])
            _params = model_params(_name, _cuda, _splitted[1], 
                                int(_splitted[2].split("_")[0]))
        
        return _params
        
    def __load_weights(self, path):
        """
            Function that loads model.

            Args:
                * path, string, path to the model checkpoint
        """
        print("Loading model!")
        _state_dict = torch.load(path, map_location = self.device)
        self.model.load_state_dict(_state_dict['model_state'])
        self.model.name = _state_dict['model_name']

    
    def evaluate(self, sample):
        self.model.eval()

        with torch.no_grad():
            _sample = sample.to(self.device)
            _prediction = self.model(_sample)
        return _prediction

    def evaluate_dropout(self, sample): 
        self.model.train()
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.eval()
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        with torch.no_grad():
            _sample = sample.to(self.device)
            _prediction = self.model(_sample)
        return _prediction



    def evalaute_and_export(self, dl_valid, dl_test):
        # Dictionaries
        _validation_dict = {'Hs-true': [], 'Hs-pred': [], 'Hs-mse': [], 'Hs-mae': [],
                'Tz-true': [], 'Tz-pred': [], 'Tz-mse': [], 'Tz-mae': [],
                'Tp-true': [], 'Tp-pred': [], 'Tp-mse': [], 'Tp-mae': [],
                }

        _test_dict = {'Hs-true': [], 'Hs-pred': [], 'Hs-mse': [], 'Hs-mae': [],
                        'Tz-true': [], 'Tz-pred': [], 'Tz-mse': [], 'Tz-mae': [],
                        'Tp-true': [], 'Tp-pred': [], 'Tp-mse': [], 'Tp-mae': [],
                }

        # Validation
        for _i, _batch in enumerate(dl_valid):
            print(f"Validation: {_i}", end = '\r')
            _input, _output, _info = _batch
            if self.model_params.name == 'ATT_NN':
                _predictions, _ = self.evaluate(_input)
            else:
                _predictions = self.evaluate(_input)
            
            if self.model_params.gpu:
                _output = _output.to('cpu')
                _predictions = _predictions.to('cpu')

            _o_hs, _o_tz, _o_tp = _output[0].detach().numpy()
            _p_hs, _p_tz, _p_tp = _predictions[0].detach().numpy()
            
            _validation_dict['Hs-true'].append(_o_hs)
            _validation_dict['Hs-pred'].append(_p_hs)
            _validation_dict['Hs-mse'].append(np.square(np.subtract(_o_hs, _p_hs)))
            _validation_dict['Hs-mae'].append(np.abs((np.subtract(_o_hs, _p_hs))))
                                        
            _validation_dict['Tz-true'].append(_o_tz)
            _validation_dict['Tz-pred'].append(_p_tz)
            _validation_dict['Tz-mse'].append(np.square(np.subtract(_o_tz, _p_tz)))
            _validation_dict['Tz-mae'].append(np.abs((np.subtract(_o_tz, _p_tz))))
            
            _validation_dict['Tp-true'].append(_o_tp)
            _validation_dict['Tp-pred'].append(_p_tp)
            _validation_dict['Tp-mse'].append(np.square(np.subtract(_o_tp, _p_tp)))
            _validation_dict['Tp-mae'].append(np.abs((np.subtract(_o_tp, _p_tp))))
    
        # Test
        for _i,_batch in enumerate(dl_test):
            print(f"Test: {_i}", end = '\r')
            _input, _output, _info = _batch
            _predictions = self.evaluate(_input)
            if self.model_params.name == 'ATT_NN':
                _predictions, _ = self.evaluate(_input)
            else:
                _predictions = self.evaluate(_input)
            
            if self.model_params.gpu:
                _output = _output.to('cpu')
                _predictions = _predictions.to('cpu')
            
            _o_hs, _o_tz, _o_tp = _output[0].detach().numpy()
            _p_hs, _p_tz, _p_tp = _predictions[0].detach().numpy()
            
            _test_dict['Hs-true'].append(_o_hs)
            _test_dict['Hs-pred'].append(_p_hs)
            _test_dict['Hs-mse'].append(np.square(np.subtract(_o_hs, _p_hs)))
            _test_dict['Hs-mae'].append(np.abs((np.subtract(_o_hs, _p_hs))))
                                        
            _test_dict['Tz-true'].append(_o_tz)
            _test_dict['Tz-pred'].append(_p_tz)
            _test_dict['Tz-mse'].append(np.square(np.subtract(_o_tz, _p_tz)))
            _test_dict['Tz-mae'].append(np.abs((np.subtract(_o_tz, _p_tz))))
            
            _test_dict['Tp-true'].append(_o_tp)
            _test_dict['Tp-pred'].append(_p_tp)
            _test_dict['Tp-mse'].append(np.square(np.subtract(_o_tp, _p_tp)))
            _test_dict['Tp-mae'].append(np.abs((np.subtract(_o_tp, _p_tp))))

        
        _writer = pd.ExcelWriter(self.model_params.name+ "-results.xlsx", engine='xlsxwriter')
        # Generate dataframes
        _df_test = pd.DataFrame.from_dict(_test_dict)
        _df_valid = pd.DataFrame.from_dict(_validation_dict)

        _df_valid.to_excel(_writer, sheet_name="Validation", index = False)
        _df_test.to_excel(_writer, sheet_name="Test", index = False)
        _writer.save() 

    
    def evalaute_and_export_HEAD(self, dl_valid, dl_test):
        # Dictionaries
        _validation_dict = {'True': [], 'Pred': [], 'mse': [], 'mae': []}

        _test_dict = {'True': [], 'Pred': [], 'mse': [], 'mae': []}

        # Validation
        for _i, _batch in enumerate(dl_valid):
            print(f"Validation: {_i}", end = '\r')
            _input, _output, _info = _batch

            #_input = torch.squeeze(_input, dim = 0)
            #_input = torch.squeeze(_input, dim = 0)

            if self.model_params.name == 'ATT_NN':
                _predictions, _ = self.evaluate(_input)
            else:
                _predictions = self.evaluate(_input)
            
            if self.model_params.gpu:
                _output = _output.to('cpu')
                _predictions = _predictions.to('cpu')

            _o = _output[0].detach().numpy()
            _p = _predictions[0].detach().numpy()

            if self.model_params.module == 'Hs':
                _o = _o[0]
            
            if self.model_params.module == 'Tz':
                _o = _o[1]
            
            if self.model_params.module == 'Dp':
                _o = _o[2]
            
            _validation_dict['True'].append(_o)
            _validation_dict['Pred'].append(_p)
            _validation_dict['mse'].append(np.square(np.subtract(_o, _p)))
            _validation_dict['mae'].append(np.abs((np.subtract(_o, _p))))
                                        
        # Test
        for _i,_batch in enumerate(dl_test):
            print(f"Test: {_i}", end = '\r')
            _input, _output, _info = _batch
            _predictions = self.evaluate(_input)
            if self.model_params.name == 'ATT_NN':
                _predictions, _ = self.evaluate(_input)
            else:
                _predictions = self.evaluate(_input)
            
            if self.model_params.gpu:
                _output = _output.to('cpu')
                _predictions = _predictions.to('cpu')
            
            _o = _output[0].detach().numpy()
            _p = _predictions[0].detach().numpy()
            
            if self.model_params.module == 'Hs':
                _o = _o[0]
            
            if self.model_params.module == 'Tz':
                _o = _o[1]
            
            if self.model_params.module == 'Dp':
                _o = _o[2]
            

            _test_dict['True'].append(_o)
            _test_dict['Pred'].append(_p)
            _test_dict['mse'].append(np.square(np.subtract(_o, _p)))
            _test_dict['mae'].append(np.abs((np.subtract(_o, _p))))

        
        _writer = pd.ExcelWriter(self.model_params.name+"-"+self.model_params.module+ "-results.xlsx", engine='xlsxwriter')
        # Generate dataframes
        _df_test = pd.DataFrame.from_dict(_test_dict)
        _df_valid = pd.DataFrame.from_dict(_validation_dict)

        _df_valid.to_excel(_writer, sheet_name="Validation", index = False)
        _df_test.to_excel(_writer, sheet_name="Test", index = False)
        _writer.save()

def return_top_k(path: str, k: int)-> dict:
    """
    Standalone method that grabs k-best results from the folder that contains formate ".xlsx" 
    resutls of model training. Returns dictionary

    Args:
        * path, str --> path to the dir contatining results.
        * k, int --> how many results to return.
    """
    # Grab xlsx
    xlsx_list = os.listdir(path)
    
    # Set save dict
    unsorted_dict = {}
    for xlsx in xlsx_list:
        mse = float(xlsx.split(":")[-1].split(".xlsx")[0]) 
        unsorted_dict[mse] = xlsx

    # Sort dict
    sort_dict = dict(sorted(unsorted_dict.items()))
    
    # Top k keys
    return_dict = {}
    for i, key in enumerate(sort_dict):
        if i == k:
            break
        return_dict[sort_dict[key]] = key
    
    return (return_dict)

