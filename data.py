"""
@author: jpzxshi
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from learner.utils import map_elementwise

class Data:
    '''Standard data format. 
    '''
    def __init__(self, X_train=None, y_train=None, X_test=None, y_test=None, device='gpu'):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        self.__device = device
        self.__dtype = None
    
    def get_batch(self, batch_size):
        @map_elementwise
        def batch_mask(X, num):
            return np.random.choice(X.size(0), num, replace=False)
        @map_elementwise
        def batch(X, mask):
            if self.__device == 'cpu':
                return X[mask]
            elif self.__device == 'gpu':
                return X[mask].cuda()
        if batch_size == None:
            if self.__device == 'cpu':
                return self.X_train, self.y_train, range(self.X_train.shape[0])
            elif self.__device == 'gpu':
                return self.X_train.cuda(), self.y_train.cuda(), range(self.X_train.shape[0])
        else:
            mask = batch_mask(self.X_train, batch_size)


            return batch(self.X_train, mask), batch(self.y_train, mask), mask
        
    def get_batch_test(self, batch_size):
        @map_elementwise
        def batch_mask(X, num):
            return np.random.choice(X.size(0), num, replace=False)
        @map_elementwise
        def batch(X, mask):
            if self.__device == 'cpu':
                return X[mask]
            elif self.__device == 'gpu':
                return X[mask].cuda()
        if batch_size == None:
            if self.__device == 'cpu':
                return self.X_test, self.y_test, range(self.X_test.shape[0])
            elif self.__device == 'gpu':
                return self.X_test.cuda(), self.y_test.cuda(), range(self.X_test.shape[0])
        else:
            mask = batch_mask(self.X_test, batch_size)
            return batch(self.X_test, mask), batch(self.y_test, mask), mask
    
    @property
    def device(self):
        return self.__device
    
    @property
    def dtype(self):
        return self.__dtype
    
    @device.setter    
    def device(self, d):
        self.__to_cpu()
        self.__device = d
    
    @dtype.setter     
    def dtype(self, d):
        if d == 'float':
            self.__to_float()
            self.__dtype = torch.float32
        elif d == 'double':
            self.__to_double()
            self.__dtype = torch.float64
        else:
            raise ValueError
    
    @property
    def dim(self):
        if isinstance(self.X_train, np.ndarray):
            return self.X_train.shape[-1]
        elif isinstance(self.X_train, torch.Tensor):
            return self.X_train.size(-1)
    
    @property
    def K(self):
        if isinstance(self.y_train, np.ndarray):
            return self.y_train.shape[-1]
        elif isinstance(self.y_train, torch.Tensor):
            return self.y_train.size(-1)
    
    @property
    def X_train_np(self):
        return Data.to_np(self.X_train)
    
    @property
    def y_train_np(self):
        return Data.to_np(self.y_train)
    
    @property
    def X_test_np(self):
        return Data.to_np(self.X_test)
    
    @property
    def y_test_np(self):
        return Data.to_np(self.y_test)
    
    @staticmethod
    @map_elementwise
    def to_np(d):
        if isinstance(d, np.ndarray) or d is None:
            return d
        elif isinstance(d, torch.Tensor):
            return d.cpu().detach().numpy()
        else:
            raise ValueError
    
    def __to_cpu(self):
        @map_elementwise
        def trans(d):
            if isinstance(d, np.ndarray):
                return torch.DoubleTensor(d)
            elif isinstance(d, torch.Tensor):
                return d.cpu()
        for d in ['X_train', 'y_train', 'X_test', 'y_test']:
            setattr(self, d, trans(getattr(self, d)))
    
    def __to_gpu(self):
        @map_elementwise
        def trans(d):
            if isinstance(d, np.ndarray):
                return torch.cuda.DoubleTensor(d)
            elif isinstance(d, torch.Tensor):
                return d.cuda()
        for d in ['X_train', 'y_train', 'X_test', 'y_test']:
            setattr(self, d, trans(getattr(self, d)))
    
    def __to_float(self):
        if self.device is None: 
            raise RuntimeError('device is not set')
        @map_elementwise
        def trans(d):
            if isinstance(d, torch.Tensor):
                return d.float()
        for d in ['X_train', 'y_train', 'X_test', 'y_test']:
            setattr(self, d, trans(getattr(self, d)))
    
    def __to_double(self):
        if self.device is None: 
            raise RuntimeError('device is not set')
        @map_elementwise
        def trans(d):
            if isinstance(d, torch.Tensor):
                return d.double()
        for d in ['X_train', 'y_train', 'X_test', 'y_test']:
            setattr(self, d, trans(getattr(self, d)))
            
            
            
class SimpleDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def random_batch(data, labels, batch_size):
    dataset = SimpleDataset(data, labels)
    sampler = RandomSampler(dataset, replacement=True, num_samples=batch_size)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    for batch_data, batch_labels in loader:
        return batch_data, batch_labels
