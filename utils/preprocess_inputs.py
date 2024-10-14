#This module has an object which takes the list of files, preprocesses and caches them
#Processing
import uproot   
import awkward as ak
import numpy as np
import pandas as pd
import os
#Deep learning
import torch
#Storage
import pickle

#Aesthetic
from tqdm import tqdm

#Memory profiling
from memory_profiler import profile

class Preprocessor():
    #@profile
    def __init__(self, file_list, var_dict, transform=None, target_transform=None, object_type="jet", cache_dir=None, use_existing_cache=False):
        self.file_list = file_list
        self.var_dict = var_dict
        self.object_type = object_type
        self.cache_dir = cache_dir
        self.use_existing_cache = use_existing_cache
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

    def __len__(self):
        return len(self.file_list)

    #@profile
    #This function caches one file
    def cache_file(self, input_path, output_path):
        #Store the data in a dict
        data_dict = {}

        f = uproot.open(input_path)
        t = f["Events"]
        events = t.arrays(self.read_variables, library="ak")

        #Global filters
        filter_jet_saturation = ak.sum(events["Jet_pt"] > 1000, axis=-1) == 0
        filter_egamma_saturation = ak.sum(events["EGamma_pt"] > 255, axis=-1) == 0
        filter_muon_saturation = ak.sum(events["Muon_pt"] > 245.5, axis=-1) == 0
        filter_saturation = filter_jet_saturation & filter_egamma_saturation & filter_muon_saturation
        events = events[filter_saturation]

        #Inputs based on training and target variables
        #Split train variables based on whether they're in aux or not
        train_variables_obj = [var for var in self.train_variables if var not in self.aux_variables]
        train_variables_aux = [var for var in self.train_variables if var in self.aux_variables]

        data = events[self.train_variables + self.target_variables]
        #Broadcast the aux variables
        for var in train_variables_aux:
            data[var] = ak.broadcast_arrays(data[var], data[train_variables_obj[0]])[0]

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

    #@profile
    #This function loads and caches all the files
    def cache_files(self):
        if self.use_existing_cache:
            print("Using existing cache at {}".format(self.cache_dir))
            return None
        else:
            print("Caching files to ", self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
            #Clear cache directory
            for file in os.listdir(self.cache_dir):
                os.remove(os.path.join(self.cache_dir, file))
            #for i, file in enumerate(self.file_list):
            #    input_path = file
            #    output_path = os.path.join(self.cache_dir, "file_{}.pkl".format(i))
            #    self.cache_file(input_path, output_path)
            #    print(f"File {i+1}/{len(self.file_list)} cached")
            #Rewrite using tqdm
            for i, file in enumerate(tqdm(self.file_list, total=len(self.file_list))):
                input_path = file
                output_path = os.path.join(self.cache_dir, "file_{}.pkl".format(i))
                self.cache_file(input_path, output_path)
            

    #@profile
    #This function returns the combined X and y tensors from the cache
    def get_X_y(self):
        X = None
        y = None
        print("Loading cache from ", self.cache_dir)
        for i, file in enumerate(tqdm(os.listdir(self.cache_dir), total=len(os.listdir(self.cache_dir)))):
            cache_file = os.path.join(self.cache_dir, file)
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
                X_i = data["X"]
                y_i = data["y"]
                if i==0:
                    X = X_i
                    y = y_i
                else:
                    X = torch.cat((X, X_i), dim=0)
                    y = torch.cat((y, y_i), dim=0)
        return X, y
            
    
