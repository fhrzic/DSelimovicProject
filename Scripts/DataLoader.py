# Import libs
import sys
import copy
import functools
import os
import glob
import random
import json
import diskcache
import pickle
import zlib
import numpy as np
import torch

# Cache function
# Cashing
class dump_disk(diskcache.Disk):
    """
        Overrides diskcache Disk class with implementation of zlib library for compression.
    """
    def store(self, value, read, key=None):
        """
            Override from base class diskcache.Disk.
            
            :param value: value to convert
            :param bool read: True when value is file-like object
            :return: (size, mode, filename, value) tuple for Cache table
        """
        
        if read is True:
            value = value.read()
            read = False

        value = pickle.dumps(value)
        value = zlib.compress(value, zlib.Z_BEST_SPEED)
        return super(dump_disk, self).store(value, read)
    
    def fetch(self, mode, filename, value, read):
        """
            Override from base class diskcache.Disk.

            :param int mode: value mode raw, binary, text, or pickle
            :param str filename: filename of corresponding value
            :param value: database value
            :param bool read: when True, return an open file handle
            :return: corresponding Python value
        """
        value = super(dump_disk, self).fetch(mode, filename, value, read)
        if not read:
            value = zlib.decompress(value)
            value = pickle.loads(value)

        return value
    

def get_cache(scope_str):
    return diskcache.FanoutCache('data-cache/' + scope_str,
                       disk = dump_disk,
                       shards=100,
                       timeout=1,
                       size_limit=3e11,
                       )


my_cache = get_cache('Cache')


# Data set cration

@functools.lru_cache(3)
def get_paths_list(data_root_dir, dataset_return):
    """
    Getting list of all samples and returning list of the required dataset samples
    Every sample has: heave, pitch, roll, hs, tz, dp
    """   
    # Get file list
    _path = os.path.join(data_root_dir,"**", "*.json")
    _list_of_paths = glob.glob(_path,  recursive=True)
    
    # Shuffle list
    _list_of_paths.sort()
    random.seed(1985)
    random.shuffle(_list_of_paths)
    
    # Return list of subset    
    if dataset_return == 0:
        return _list_of_paths[0:int(0.8*len(_list_of_paths))]
    if dataset_return == 1:
        return _list_of_paths[int(0.8*len(_list_of_paths)):int(0.9*len(_list_of_paths))]
    
    return _list_of_paths[int(0.9*len(_list_of_paths)):]


@functools.lru_cache(maxsize=1, typed=True)
def get_data(instance):
    """
        Help function for cashing.
    """
    return Data(instance)

@my_cache.memoize(typed=True)
def get_data_sample(instance):
    """
        Help function for cashing.
    """
    _data = get_data(instance)
    _input, _output = _data.get_sample()
    return _input, _output

class Data:
    """
    Class for loading data from json
    """
    def __init__(self, instance_path):
        self.instance_path = instance_path
        
        # Open JSON file
        _file = open(instance_path)
        
        # Return JSON object as dir
        self.data = json.load(_file)
        
        _file.close()
    
    def get_sample(self):
        # Return data
        
        _heave = np.array(self.data['input']['heave']) / 20.0
        _pitch = np.array(self.data['input']['pitch']) / 20.0
        _roll = np.array(self.data['input']['roll']) / 20.0
                           
        _hs = np.array(self.data['output']['Hs']) / 15.0
        _tz = np.array(self.data['output']['Tz']) / 15.0
        _dp = np.array(self.data['output']['Dp']) / 360.0
                           
        return ((_heave, _pitch, _roll), (_hs, _tz, _dp))

class ship_dataset:
    
    def __init__(self, data_root_dir = None, dataset_return = 'train'):
        """
        Init: Creates data loader.
        
        Args:
        
            * data_root_dir, str, path to data root
            
            * dataset_return, number or enum [0,1,2]: 0 - train, 1 - valid, 2 - test, default: train
        """
        
        # Transform names data
        assert dataset_return in [0, 1, 2, 'train', 'valid', 'test'], f"Wrong dataset descriptor, got: {dataset_return}"
        _name_transform_dict = {'train': 0, 'valid': 1, 'test': 2}
              
        if isinstance(dataset_return, str): 
            self.dataset_return = _name_transform_dict[dataset_return]
        else:
            self.dataset_return = dataset_return
        
        # Obtain data list
        try:
            if data_root_dir != None:
                self.data_list = copy.copy(get_paths_list(data_root_dir, self.dataset_return))
            else:
                raise ValueError
        except ValueError:
            print("Data root dir missing, Can not load data!")
            sys.exit(1)
        
        # Number of samples
        self.samples_cnt = len(self.data_list)
        
    def shuffle_samples(self):
         # Shuffeling dataset
        random(2000)
        random.shuffle(self.data_list)
        
    def __len__(self):
        return self.samples_cnt
    
    def __getitem__(self, ndx):
        # Get sample id
        _sample_path = self.data_list[ndx]
                           
        _input, _output = get_data_sample(_sample_path)

        _input = torch.from_numpy(np.asarray(_input))

        _output = torch.from_numpy(np.asarray(_output))

        _input = _input.to(torch.float64)
        _input = _input.unsqueeze(0)

        

        # RESCALING DO HERE
        return (_input, _output, _sample_path)               
