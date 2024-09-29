#This is a module that contains the dataset definition used for the pT regression training
#Processing
import uproot   
import awkward as ak
import numpy as np
import pandas as pd
import os
#Deep learning
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
#Storage
import pickle

#Memory profiling
from memory_profiler import profile

class L1ScoutingDataset(Dataset):
    #@profile
    def __init__(self, dir_path, var_dict, file_format="root", transform=None, target_transform=None, object_type="jet"):
        self.dir_path = dir_path
        self.var_dict = var_dict
        self.object_type = object_type
        
        #Load the file
        file_list = os.listdir(dir_path)
        self.file_list = [f for f in file_list if file_format in f]
        
        self.transform = transform
        self.target_transform = target_transform

        #Load variables
        self.read_variables = self.var_dict["read"]
        #Object specific variables
        self.jet_variables = self.var_dict["jet"]
        self.muon_variables = self.var_dict["muon"]
        self.egamma_variables = self.var_dict["egamma"]
        self.aux_variables = self.var_dict["aux"]
        #Target variables
        self.jet_target = self.var_dict["jet_target"]
        self.jet_target_rel = self.var_dict["jet_target_rel"]
        self.muon_target = self.var_dict["muon_target"]
        self.muon_target_rel = self.var_dict["muon_target_rel"]
        self.egamma_target = self.var_dict["egamma_target"]
        self.egamma_target_rel = self.var_dict["egamma_target_rel"]

        #Set train and target variables based on object type
        if self.object_type == "jet":
            self.train_variables = self.jet_variables
            self.target_variables = self.jet_target_rel
        elif self.object_type == "muon":
            self.train_variables = self.muon_variables
            self.target_variables = self.muon_target_rel
        elif self.object_type == "egamma":
            self.train_variables = self.egamma_variables
            self.target_variables = self.egamma_target_rel

        self.num_train_variables = len(self.train_variables)
        self.num_target_variables = len(self.target_variables)

    #@profile
    def __len__(self):
        return len(self.file_list)
    
    #@profile
    def __getitem__(self, idx):
        #Open file
        file_path = os.path.join(self.dir_path, self.file_list[idx])
        f = uproot.open(file_path)
        t = f["Events"]
        events = t.arrays(self.read_variables, library="ak")

        #Global filters
        filter_jet_saturation = ak.sum(events["Jet_pt"] > 1000, axis=-1) == 0
        filter_egamma_saturation = ak.sum(events["EGamma_pt"] > 255, axis=-1) == 0
        filter_muon_saturation = ak.sum(events["Muon_pt"] > 245.5, axis=-1) == 0
        filter_saturation = filter_jet_saturation & filter_egamma_saturation & filter_muon_saturation
        events = events[filter_saturation]

        #Inputs based on training and target variables
        data = events[self.train_variables + self.target_variables]

        #Object specific filters
        if self.transform:
            data = self.transform(data)
        
        X = data[self.train_variables]
        y = data[self.target_variables]

        #Pandas and then tensor conversion
        X = ak.to_dataframe(X)
        y = ak.to_dataframe(y)

        X_tensor = torch.tensor(X.values)
        y_tensor = torch.tensor(y.values)   

        return X_tensor, y_tensor

    #Function to cache X_tensor and y_tensor
    #@profile
    def cache_file(self, file_path, output_path):
        #Store the data in a dict
        data_dict = {}

        #Basically does the same thing as __getitem__ but stores the data as a pickle file
        f = uproot.open(file_path)
        t = f["Events"]
        events = t.arrays(self.read_variables, library="ak")

        #Global filters
        filter_jet_saturation = ak.sum(events["Jet_pt"] > 1000, axis=-1) == 0
        filter_egamma_saturation = ak.sum(events["EGamma_pt"] > 255, axis=-1) == 0
        filter_muon_saturation = ak.sum(events["Muon_pt"] > 245.5, axis=-1) == 0
        filter_saturation = filter_jet_saturation & filter_egamma_saturation & filter_muon_saturation
        events = events[filter_saturation]

        #Inputs based on training and target variables
        data = events[self.train_variables + self.target_variables]

        #Object specific filters
        if self.transform:
            data = self.transform(data)
        
        X = data[self.train_variables]
        y = data[self.target_variables]

        #Pandas and then tensor conversion
        X = ak.to_dataframe(X)
        y = ak.to_dataframe(y)

        X_tensor = torch.tensor(X.values)
        y_tensor = torch.tensor(y.values)

        #Store the data in a dict
        data_dict["X"] = X_tensor
        data_dict["y"] = y_tensor

        #Save the data
        with open(output_path, "wb") as f:
            pickle.dump(data_dict, f)
        
        return None

#Padding due to variable number of events
from torch.nn.utils.rnn import pad_sequence
def collate_fn(batch):
    # Separate X and y from the batch
    X_batch = [item[0] for item in batch]
    y_batch = [item[1] for item in batch]
    
    # Pad sequences for X and y
    X_batch_padded = pad_sequence(X_batch, batch_first=True, padding_value=0.0)
    y_batch_padded = pad_sequence(y_batch, batch_first=True, padding_value=0.0)
    
    return X_batch_padded, y_batch_padded

