# Takes the X, y, w data from the preprocessor and wraps it into a BDT dataset
import numpy as np
import awkward as ak    
import torch
from torch.utils.data import Dataset

# System
import os
import sys

# Plotting
import matplotlib.pyplot as plt

# Aesthetic
from tqdm import tqdm

class BDTDataset(Dataset):
    def __init__(self, X, y, w, train_features, target_features):
        self.train_features = train_features
        self.target_features = target_features

        X = X.loc[:, train_features]
        y = y.loc[:, target_features]

        self.x = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)
        self.W = torch.tensor(w.values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.W)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.W[idx]
    