# Preprocessing for per-object (Flattening involved)
import importlib
import yaml
import uproot   
import awkward as ak
import numpy as np
import pandas as pd
import os
#Storage
import pickle

#Aesthetic
from tqdm import tqdm

class Preprocessor():
    def __init__(self, file_list, config_name, tree_name="Events", label=0, transform=None, cache_dir="cachedir", use_existing_cache=False, batch_size=10000):
        self.file_list = file_list
        self.config_name = config_name
        self.config = None
        self.tree_name = tree_name
        self.label = label
        self.transform = transform
        self.cache_dir = cache_dir
        self.use_existing_cache = use_existing_cache
        self.batch_size = batch_size

        # Variable lists
        self.read_vars = None
        self.train_vars = None
        self.plot_vars = None
        self.all_vars = None

        # Training object and target variable(s)
        self.train_object = None
        self.target_vars = None

        # Cutflow
        self.cutflow = {
            "total_events": 0,
            "passed_events": 0,
        }

        # Load config
        config = yaml.load(open(self.config_name, "r"), Loader=yaml.FullLoader)
        self.config = config

        # Read variables
        read_vars = []
        for var in config["read_vars"]["scalar_vars"]:
            read_vars.append(var)
        for obj in config["read_vars"]["jagged_vars"].keys():
            for var in config["read_vars"]["jagged_vars"][obj]:
                read_vars.append(f"{obj}_{var}")
        
        # Train variables
        train_vars = []
        for var in config["train_vars"]["scalar_vars"]:
            train_vars.append(var)
        for obj in config["train_vars"]["jagged_vars"].keys():
            for var in config["train_vars"]["jagged_vars"][obj]:
                train_vars.append(f"{var}")            

        # Plot variables
        plot_vars = []
        for var in config["plot_vars"]["scalar_vars"]:
            plot_vars.append(var)
        for obj in config["plot_vars"]["jagged_vars"].keys():
            for var in config["plot_vars"]["jagged_vars"][obj]:
                plot_vars.append(f"{var}")
        
        # All variables
        all_vars = []

        self.read_vars = read_vars
        self.train_vars = train_vars
        self.plot_vars = plot_vars  
        self.all_vars = all_vars 

        # Infer training collection
        if len(config["train_vars"]["jagged_vars"].keys()) > 1:
            raise ValueError(f"More than one jagged collection trained on: {config['train_vars']['jagged_vars'].keys()}. \nUse only one for training in per-object MVAs")
        self.train_object = (list(config["train_vars"]["jagged_vars"].keys()))[0]

        # Target variables
        self.target_vars = config["target_vars"]
             
        # Debug
        print(f"All read variables:", read_vars)
        print(f"All train variables:", train_vars)
        print(f"All plot variables:", plot_vars)
        print(f"Inclusive set of variables:", all_vars)
        print(f"Target variables:", self.target_vars)
    
    def __len__(self):
        return len(self.file_list)
    
    # Load the data from an existing cache
    def get_X_y_w(self):
        X = None
        y = None
        w = None
        print("Loading cache from", self.cache_dir)
        for i, file in enumerate(tqdm(os.listdir(self.cache_dir), total=len(os.listdir(self.cache_dir)))):
            cache_file = os.path.join(self.cache_dir, file)

            with open(cache_file, "rb") as f:
                data = pickle.load(f)
                X_i = data["X"]
                y_i = data["y"]
                w_i = data["w"]
                self.cutflow["total_events"] += data["total_events"]
                self.cutflow["passed_events"] += data["passed_events"]
                if i == 0:
                    X = X_i
                    y = y_i
                    w = w_i
                else:
                    X = pd.concat([X, X_i])
                    y = pd.concat([y, y_i])
                    w = pd.concat([w, w_i])
        
        return X, y, w
    
    # File caching (does the bulk of preprocessing)
    def cache_file(self, input_path, output_path):

        # Open the file
        f = uproot.open(input_path)
        t = f[self.tree_name]
        # Get total number of events
        n_entries_file = t.num_entries
        n_batches = n_entries_file//self.batch_size + 1
        print(f"Splitting file into {n_batches} batches")

        for i_batch in tqdm(range(n_batches)):
            data_dict = {}

            events = t.arrays(self.read_vars, library="ak", entry_start=i_batch*self.batch_size, entry_stop=(i_batch+1)*self.batch_size, how="zip")

            # Count total events
            total_events = len(events)

            ############### Helper functions ####################
            helper_funcs = self.config["helper_funcs"]
            for module_name in helper_funcs.keys():
                module = importlib.import_module(module_name)
                for helper_func_name in helper_funcs[module_name]:
                    helper_func = getattr(module, helper_func_name)
                    events = helper_func(events)

            ######### Jagged and scalar filters ##################
            filters = self.config["filters"]
            if filters["scalar_filters"] is not None:
                for scalar_filter in filters["scalar_filters"]:
                    mask = eval(f"events.{scalar_filter}")
                    events = events[mask]
            if filters["jagged_filters"] is not None:
                collections_filtered = []
                for jagged_filter in filters["jagged_filters"]:
                    collection = jagged_filter.split(".")[0]
                    mask = eval(f"events.{jagged_filter}")
                    events[collection] = events[collection][mask]
                    if collection not in collections_filtered:
                        collections_filtered.append(collection)
                # Recount multiplicities after filtering
                #for collection in collections_filtered:
                #    events[f"n{collection}"] = ak.num(events[collection])

            # Count passed events
            passed_events = len(events)

            # Reorder the jagged arrays
            sort_vars = self.config["sort_vars"]
            for obj in sort_vars:
                sort_var = sort_vars[obj]["var"]
                #print(f"Sorting {obj} by {sort_var}")
                sort_mask = ak.argsort(events[obj][sort_var], axis=-1, ascending=sort_vars[obj]["ascending"])
                events[obj] = events[obj][sort_mask]

            ############ Broadcast and flatten object collections ############
            # Train
            train_object_dict = {}
            for var in self.config["train_vars"]["jagged_vars"][self.train_object]:
                train_object_dict[var] = events[self.train_object][var]       
            train_objects = ak.zip(train_object_dict)
            for var in self.config["train_vars"]["scalar_vars"]:
                train_objects = ak.with_field(
                    train_objects,
                    ak.broadcast_arrays(
                        events[var][:, None],
                        train_objects[(train_object_dict.keys())[0]]
                    )[0],
                    var
                )
            train_objects = ak.flatten(train_objects, axis=-1)
            # Revert to dict
            train_object_dict = {}
            for key in train_objects.fields:
                train_object_dict[key] = train_objects[key]

            # Target
            target_object_dict = {}
            for var in self.target_vars:
                target_object_dict[var] = events[self.train_object][var]
            target_objects = ak.zip(target_object_dict)
            target_objects = ak.flatten(target_objects, axis=-1)
            # Revert to dict
            target_object_dict = {}
            for key in target_objects.fields:
                target_object_dict[key] = target_objects[key]

            X_df = pd.DataFrame(train_object_dict, dtype=np.float64)
            y_df = pd.DataFrame(target_object_dict, dtype=np.float64)
            if self.config["weight_var"] is not None:
                w_df = pd.DataFrame({self.config["weight_var"]: events[self.config["weight_var"]]})
            else:
                w_df = pd.DataFrame({"weight": np.ones((X_df.shape[0],))})

            # Store the data
            data_dict["X"] = X_df
            data_dict["y"] = y_df 
            data_dict["w"] = w_df
            # Save cutflow efficiencies
            data_dict["total_events"] = total_events
            data_dict["passed_events"] = passed_events

            # Replace the file name with a batch number: file_0.pkl -> file_0_0.pkl
            output_path_batch = output_path.replace(".pkl", f"_{i_batch}.pkl")
            with open(output_path_batch, "wb") as f:
                pickle.dump(data_dict, f)

        return None


    # Cache all files
    def cache_files(self):
        if self.use_existing_cache:
            print(f"Using existing cache at {self.cache_dir}")
            return None
        
        else:
            print("Caching files to ", self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
            #Clear cache directory
            for file in os.listdir(self.cache_dir):
                os.remove(os.path.join(self.cache_dir, file))

            for i, file in enumerate(tqdm(self.file_list)):
                input_path = file
                output_path = os.path.join(self.cache_dir, f"file_{i}.pkl")
                self.cache_file(input_path, output_path)

# Example usage
if __name__ == "__main__":
    import sys
    import glob
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    file_list = glob.glob(f"test/files/*.root")
    print(f"Number of  files: {len(file_list)}")
    print(file_list)

    # Preprocess files
    preprocessor = Preprocessor(file_list, "config/l1regression.yml", tree_name="Events", label=0, cache_dir=f"cache/test", use_existing_cache=False, batch_size=100000)

    preprocessor.cache_files()

    X,y,w = preprocessor.get_X_y_w()

    print(f"X: {X}")
    print(f"y: {y}")
    print(f"w: {w}")