# Preprocesses files 
import yaml
import uproot   
import awkward as ak
import numpy as np
import pandas as pd
import os
import sys
# Deep learning
import torch
import torch_geometric
from torch_geometric.data import Data
# Storage
import pickle
# Memory management
import psutil
import gc

# Aesthetic
from tqdm import tqdm

"""
Returns dataset with following structure:
X:
        feature_0 feature_1 ... feature_n
Jet_0
Jet_1
...

Y:
        recojet_pT recojet_eta recojet_phi
Jet_0
Jet_1
...

W: Jet weights (two jets per event)
"""

def phi_phasewrap(phi):
    """
    Used for example when phi is deltaphi = phi1 - phi2
    """
    return (phi + np.pi) % (2 * np.pi) - np.pi

class Preprocessor():
    def __init__(self, file_list, tree_name="scNtuplizer/Events", cache_dir="cachedir", use_existing_cache=False, batch_size=10000, class_label=0):
        self.file_list = file_list
        self.tree_name = tree_name
        self.cache_dir = cache_dir
        self.use_existing_cache = use_existing_cache  
        self.batch_size = batch_size
        self.class_label = class_label  

    def __len__(self):
        return len(self.file_list)
    
    # Preprocess and cache each file
    def cache_file(self, input_path, output_path):

        # Open the file
        f = uproot.open(input_path)
        t = f[self.tree_name]

        # Stores X, y, w, u
        data_dict = {}

        var_dict = {
            "GenJet": ["pt", "eta", "phi"],
            "Jet": ["pt", "eta", "phi"],
            "Muon": ["pt", "eta", "phi", "etaAtVtx", "phiAtVtx", "hwCharge"],
            "EGamma": ["pt", "eta", "phi", "e"],
            "RecoJet": ["pt", "eta", "phi", "e", "mass", "btagPNetB"]
        }     

        var_list = []
        for key in var_dict:
            for var in var_dict[key]:
                var_list.append(f"{key}_{var}")

        genjet_vars = [f"GenJet_{var}" for var in var_dict["GenJet"]]
        jet_vars = [f"Jet_{var}" for var in var_dict["Jet"]]
        muon_vars = [f"Muon_{var}" for var in var_dict["Muon"]]
        egamma_vars = [f"EGamma_{var}" for var in var_dict["EGamma"]]
        recojet_vars = [f"RecoJet_{var}" for var in var_dict["RecoJet"]]

        # Add global vars
        var_list += ["nGenJet", "nJet", "nMuon", "nEGamma", "nRecoJet", "etSum", "htSum", "etMiss", "etMissPhi", "htMiss", "htMissPhi", "towerCount"]

        # Read the data
        #print(f"Reading variables: {var_list}")

        total_events_file = t.num_entries
        if total_events_file < self.batch_size:
            n_batches = 1
        else:
            n_batches = total_events_file//self.batch_size + 1
        #print(f"Splitting file into {n_batches} batches")

        # Batch reading (TBD)
        for i_batch in tqdm(range(n_batches), desc=f"Processing {input_path}"):
            events = t.arrays(var_list, library="ak", entry_start=i_batch*self.batch_size, entry_stop=(i_batch+1)*self.batch_size)
            total_events = len(events)

            # Apply basic filers
            events = events[(events["nJet"] > 1) & (events["nRecoJet"] > 1)]
            jet_pt_filter = (events["Jet_pt"][:, 0] > 30) & (events["Jet_pt"][:, 1] > 30)
            jet_eta_filter = (np.abs(events["Jet_eta"][:, 0]) < 2.5) & (np.abs(events["Jet_eta"][:, 1]) < 2.5)
            saturated_jet_veto  = ak.sum(events["Jet_pt"] > 1023, axis=1) == 0
            events = events[jet_pt_filter & jet_eta_filter & saturated_jet_veto]

            # Zip the collections
            l1jets = ak.zip({
                "pt": events["Jet_pt"],
                "eta": events["Jet_eta"],
                "phi": events["Jet_phi"]
            })
            recojets = ak.zip({
                "pt": events["RecoJet_pt"],
                "eta": events["RecoJet_eta"],
                "phi": events["RecoJet_phi"],
                "e": events["RecoJet_e"],
                "mass": events["RecoJet_mass"],
                "btagPNetB": events["RecoJet_btagPNetB"]
            })
            genjets = ak.zip({
                "pt": events["GenJet_pt"],
                "eta": events["GenJet_eta"],
                "phi": events["GenJet_phi"]
            })
            muons = ak.zip({
                "pt": events["Muon_pt"],
                "eta": events["Muon_eta"],
                "phi": events["Muon_phi"],
                "etaAtVtx": events["Muon_etaAtVtx"],
                "phiAtVtx": events["Muon_phiAtVtx"],
                "hwCharge": events["Muon_hwCharge"]
            })
            egammas = ak.zip({
                "pt": events["EGamma_pt"],
                "eta": events["EGamma_eta"],
                "phi": events["EGamma_phi"],
                "e": events["EGamma_e"]
            })

            # Take only the first two l1 jets
            l1jets = l1jets[:, :2]

            # Match to recojets
            l1_reco_pair = ak.cartesian({"l1": l1jets, "reco": recojets}, nested=True)
            l1_reco_pair_args = ak.argcartesian({"l1": l1jets, "reco": recojets}, nested=True)
            l1_reco_dR = np.sqrt(
                (l1_reco_pair["l1"]["eta"] - l1_reco_pair["reco"]["eta"])**2 +
                (phi_phasewrap(l1_reco_pair["l1"]["phi"] - l1_reco_pair["reco"]["phi"]))**2
            )
            l1_reco_dR_order = ak.argsort(l1_reco_dR, ascending=True, axis=2)
            l1_reco_pair = l1_reco_pair[l1_reco_dR_order]
            l1_reco_pair_args = l1_reco_pair_args[l1_reco_dR_order]
            l1_reco_dR = l1_reco_dR[l1_reco_dR_order]
            # Match within 0.3
            l1_reco_match_cut = l1_reco_dR < 0.3
            l1_reco_pair = l1_reco_pair[l1_reco_match_cut]
            l1_reco_pair_args = l1_reco_pair_args[l1_reco_match_cut]
            l1_reco_dR = l1_reco_dR[l1_reco_match_cut]
            # Perform arbitration
            l1jet_i, recojet_i = ak.unzip(l1_reco_pair)
            l1jet_arg_i, recojet_arg_i = ak.unzip(l1_reco_pair_args)
            match_per_l1jet = ak.num(l1_reco_pair, axis=-1)
            l1jet_arg_i = l1jet_arg_i[match_per_l1jet == 1]
            l1jet_arg_i = ak.firsts(l1jet_arg_i, axis=-1)
            l1jet_arg_i = l1jet_arg_i[~ak.is_none(l1jet_arg_i, axis=-1)]
            recojet_arg_i = recojet_arg_i[match_per_l1jet == 1]
            recojet_arg_i = ak.firsts(recojet_arg_i, axis=-1)
            recojet_arg_i = recojet_arg_i[~ak.is_none(recojet_arg_i, axis=-1)]  
            l1jets_matched = l1jets[l1jet_arg_i]
            recojets_matched = recojets[recojet_arg_i]

            # Feature extraction
            # Match to muons
            l1_muon_pair = ak.cartesian({"l1": l1jets_matched, "muon": muons}, nested=True)
            l1_muon_pair_args = ak.argcartesian({"l1": l1jets_matched, "muon": muons}, nested=True)
            l1_muon_dR = np.sqrt(
                (l1_muon_pair["l1"]["eta"] - l1_muon_pair["muon"]["etaAtVtx"])**2 +
                (phi_phasewrap(l1_muon_pair["l1"]["phi"] - l1_muon_pair["muon"]["phiAtVtx"]))**2
            )
            l1_muon_dR_order = ak.argsort(l1_muon_dR, ascending=True, axis=2)
            l1_muon_pair = l1_muon_pair[l1_muon_dR_order]
            l1_muon_pair_args = l1_muon_pair_args[l1_muon_dR_order]
            l1_muon_dR = l1_muon_dR[l1_muon_dR_order]
            # Match within 0.4
            l1_muon_match_cut = l1_muon_dR < 0.4
            l1_muon_pair = l1_muon_pair[l1_muon_match_cut]
            l1_muon_pair_args = l1_muon_pair_args[l1_muon_match_cut]
            l1_muon_dR = l1_muon_dR[l1_muon_match_cut]
            # Sum up muon pt for all matched muons
            l1jet_i, muon_i = ak.unzip(l1_muon_pair)
            l1jet_muon_iso = ak.sum(muon_i["pt"], axis=-1)
            l1jets_matched = ak.with_field(l1jets_matched, l1jet_muon_iso, "muonIso")
            l1jets_matched = ak.with_field(l1jets_matched, l1jet_muon_iso/l1jets_matched.pt, "muonRelIso")
            
            # Match to egammas
            l1_egamma_pair = ak.cartesian({"l1": l1jets_matched, "egamma": egammas}, nested=True)
            l1_egamma_pair_args = ak.argcartesian({"l1": l1jets_matched, "egamma": egammas}, nested=True)
            l1_egamma_dR = np.sqrt(
                (l1_egamma_pair["l1"]["eta"] - l1_egamma_pair["egamma"]["eta"])**2 +
                (phi_phasewrap(l1_egamma_pair["l1"]["phi"] - l1_egamma_pair["egamma"]["phi"]))**2
            )
            l1_egamma_dR_order = ak.argsort(l1_egamma_dR, ascending=True, axis=2)
            l1_egamma_pair = l1_egamma_pair[l1_egamma_dR_order]
            l1_egamma_pair_args = l1_egamma_pair_args[l1_egamma_dR_order]
            l1_egamma_dR = l1_egamma_dR[l1_egamma_dR_order]
            # Match within 0.4
            l1_egamma_match_cut = l1_egamma_dR < 0.4
            l1_egamma_pair = l1_egamma_pair[l1_egamma_match_cut]
            l1_egamma_pair_args = l1_egamma_pair_args[l1_egamma_match_cut]
            l1_egamma_dR = l1_egamma_dR[l1_egamma_match_cut]
            # Sum up egamma pt
            l1jet_i, egamma_i = ak.unzip(l1_egamma_pair)
            l1jet_egamma_iso = ak.sum(egamma_i["pt"], axis=-1)
            l1jets_matched = ak.with_field(l1jets_matched, l1jet_egamma_iso, "egammaIso")
            l1jets_matched = ak.with_field(l1jets_matched, l1jet_egamma_iso/l1jets_matched.pt, "egammaRelIso")

            # Broadcast global features to l1jet dimensionality
            features_to_broadcast = ["etSum", "htSum", "etMiss", "etMissPhi", "htMiss", "htMissPhi", "towerCount"]
            for feature in features_to_broadcast:
                l1jets_matched = ak.with_field(
                    l1jets_matched,
                    ak.broadcast_arrays(
                        events[feature][:, None],
                        l1jets_matched.pt
                    )[0],
                    feature
                )

            # Jet dataframe creation
            l1jets_matched = ak.flatten(l1jets_matched, axis=-1)
            l1jets_fields = l1jets_matched.fields
            l1jets_dict = {f"{field}": l1jets_matched[field] for field in l1jets_fields}
            l1jets_df = pd.DataFrame(l1jets_dict, dtype=np.float64)

            # Target dataframe creation
            recojets_matched = ak.flatten(recojets_matched, axis=-1)
            l1jet_recoptdiff = recojets_matched.pt - l1jets_matched.pt
            l1jet_recoptratio = recojets_matched.pt / l1jets_matched.pt
            l1jet_recoetadiff = recojets_matched.eta - l1jets_matched.eta
            l1jet_recophidiff = phi_phasewrap(recojets_matched.phi - l1jets_matched.phi)
            targets_dict = {
                "recojet_pt": recojets_matched.pt,
                "recojet_ptdiff": l1jet_recoptdiff,
                "recojet_ptratio": l1jet_recoptratio,
                "recojet_eta": recojets_matched.eta,
                "recojet_etadiff": l1jet_recoetadiff,
                "recojet_phi": recojets_matched.phi,
                "recojet_phidiff": l1jet_recophidiff,
                "recojet_btag": recojets_matched.btagPNetB
            }
            targets_df = pd.DataFrame(targets_dict, dtype=np.float64)
            
            # Weights dataframe creation
            weights_df = pd.DataFrame({
                "weight": np.ones(len(l1jets_df), dtype=np.float32)
            })

            # Store the data
            data_dict["X"] = l1jets_df
            data_dict["y"] = targets_df
            data_dict["w"] = weights_df
            output_path_batch = output_path.replace(".pkl", f"_{i_batch}.pkl")
            with open(output_path_batch, "wb") as f:
                pickle.dump(data_dict, f)
        


        return None
    
    # Cache all files in file list
    def cache_files(self):
        if self.use_existing_cache:
            print(f"Using existing cache at {self.cache_dir}")
            return None
        else:
            print(f"Caching files to {self.cache_dir}")
            os.makedirs(self.cache_dir, exist_ok=True)
            # Clear cache directory
            for file in os.listdir(self.cache_dir):
                os.remove(os.path.join(self.cache_dir, file))
            for i, file in enumerate(tqdm(self.file_list)):
                input_path = file
                output_path = os.path.join(self.cache_dir, f"file_{i}.pkl")
                self.cache_file(input_path, output_path)

    # Load the concatenated data from the cache
    def get_data_dict(self):
        X_chunks, y_chunks, w_chunks= [], [], []
        next_entry_id = 0

        print(f"Loading data from {self.cache_dir}")
        for i, file in enumerate(tqdm(os.listdir(self.cache_dir), total=len(os.listdir(self.cache_dir)))):
            
            cache_file = os.path.join(self.cache_dir, file)
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            
            X_i = data["X"]
            y_i = data["y"]
            w_i = data["w"]

            #n_jets = y_i.shape[0]

            #old_entries = X_i.index.levels[0]
            #new_entries = old_entries + next_entry_id
            #X_i = X_i.copy()
            #X_i.index = X_i.index.set_levels(
            #    [new_entries, X_i.index.levels[1]],  # [new_entry_ids, same subentry level]
            #    level=[0, 1]
            #)
            #y_i = y_i.copy(); y_i.index = y_i.index + next_entry_id
            #w_i = w_i.copy(); w_i.index = w_i.index + next_entry_id
            #next_entry_id += n_jets
            
            X_chunks.append(X_i)
            y_chunks.append(y_i)
            w_chunks.append(w_i)

        X = pd.concat(X_chunks)
        y = pd.concat(y_chunks)
        w = pd.concat(w_chunks)
        return X, y, w


    

# Example usage
if __name__ == "__main__":
    file_list = os.listdir("test")
    file_list = [os.path.join("test", file) for file in file_list if file.endswith(".root")]
    print(f"Found {len(file_list)} files to process")
    preprocessor = Preprocessor(file_list, batch_size=100000)

    #preprocessor.cache_file(file_list[0], "cachedir/output_1.pkl")
    preprocessor.cache_files()
    X, y, w = preprocessor.get_data_dict()