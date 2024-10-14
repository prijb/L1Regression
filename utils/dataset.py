#This module gives the dataset definition
#Deep learning
import torch
from torch.utils.data import Dataset
#Storage
import pickle

class L1ObjectDataset(Dataset):

    def __init__(self, X, y, W=None):
        self.x = X
        self.y = y
        if W is None:
            self.W = torch.ones(self.y.shape[0])
        else:
            self.W = W
        
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

