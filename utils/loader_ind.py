#This dataset definition obtains singular objects from the L1ScoutingDataset using cached files
import awkward as ak
import numpy as np
import pandas as pd
import os
import bisect
#Deep learning
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
#Storage
import pickle
#Garbage collection
import gc

#Memory profiling
from memory_profiler import profile

#Inherits from the L1ScoutingDataset class
from utils.loader import L1ScoutingDataset

class L1ScoutingObject(L1ScoutingDataset):
    #@profile
    def __init__(self, dir_path, var_dict, file_format="root", transform=None, target_transform=None, object_type="jet", use_cache=False, cache_path=None):
        super(L1ScoutingObject, self).__init__(dir_path, var_dict, file_format, transform, target_transform, object_type)
        self.cache_path = cache_path
        self.use_cache = use_cache
        self.counts_per_file = [] #Number of objects in each file
        self.cumulative_counts = []  #Cumulative counts in each file
        total_counts = 0

        #Cache all files 
        if not self.use_cache:
            print("Caching files to ", cache_path)
            os.makedirs(cache_path, exist_ok=True)
            for i, file in enumerate(self.file_list):
                input_path = os.path.join(self.dir_path, file)
                output_path = os.path.join(cache_path, file.replace(".root", ".pkl"))
                #print("File: ", i+1)
                #print("Loading file: ", input_path)
                #print("Writing to file: ", output_path)
                self.cache_file(input_path, output_path)
                print(f"File {i+1}/{len(self.file_list)} cached")
        
        #Save the object counts (cumulative) and file indices by reading from the cache
        print("Loading cached files from ", cache_path)
        self.cache_file_list = os.listdir(cache_path)
        for i, file in enumerate(self.cache_file_list):
            with open(os.path.join(cache_path, file), "rb") as f:
                data = pickle.load(f)
                y = data["y"]
                total_counts += len(y)
                self.counts_per_file.append(len(y))
                self.cumulative_counts.append(total_counts)
                
            #Garbage collection    
            del data
            del y
            gc.collect()
        
        self.total_counts = total_counts
        print("Total number of objects in dataset: ", total_counts)

    #@profile
    def __len__(self):
        return self.total_counts
    
    #@profile
    #Uses the index position with respect to the cumulative counts and loads the correct file
    def __getitem__(self, idx):
        #Get the file index and the index of the object in that file
        file_index = bisect.bisect_right(self.cumulative_counts, idx)
        object_index = idx
        if file_index != 0:
            object_index = idx - self.cumulative_counts[file_index-1]

        #Load the file
        file_path = os.path.join(self.cache_path, self.cache_file_list[file_index])
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            X = data["X"]
            y = data["y"]
            X_i = X[object_index]
            y_i = y[object_index]
        
        #Garbage collection
        del data
        del X
        del y
        gc.collect()
        
        return X_i, y_i
    


        
