# Train a BDT classifier 
import os
import sys
import numpy as np

# Data loaders
import torch
import torch.utils
from torch.utils.data import DataLoader, Subset, ConcatDataset

# Stats
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Additional imports
# Add the project path
cwd = os.getcwd()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Make plotdir if it doesn't exist
os.makedirs(f"{cwd}/plots/training", exist_ok=True)
os.makedirs(f"{cwd}/plots/features", exist_ok=True)

from utils.preprocess import Preprocessor
from utils.bdt_dataset import BDTDataset

input_files = os.listdir("test")
input_files = [os.path.join("test", file) for file in input_files if file.endswith(".root")]

preprocessor = Preprocessor(input_files, batch_size=100000, use_existing_cache=True)
preprocessor.cache_files()
X, y, w = preprocessor.get_data_dict()

############# Data exploration #################
print(f"Loaded data with {len(X)} jets")
import mplhep as hep
plt.style.use(hep.style.CMS)

jet_features_to_draw = ["pt", "eta", "phi", "muonRelIso", "egammaRelIso"]
global_features_to_draw = ["etSum", "htSum", "etMiss", "etMissPhi", "htMiss", "htMissPhi", "towerCount"]
target_features_to_draw = ["recojet_pt", "recojet_eta", "recojet_phi", "recojet_ptdiff", "recojet_ptratio", "recojet_etadiff", "recojet_phidiff", "recojet_btag"]

for feature in jet_features_to_draw:
    fig, ax = plt.subplots()
    plt.hist(X[feature], bins=100, histtype='step', label=f"L1Jet_{feature}")
    plt.xlabel(f"L1Jet_{feature}")
    plt.ylabel("Number of jets")
    ax.set_yscale('log')
    plt.savefig(f"plots/features/l1jet_{feature}.png")
    plt.close()

for feature in global_features_to_draw:
    fig, ax = plt.subplots()
    plt.hist(X[feature], bins=100, histtype='step', label=f"{feature}")
    plt.xlabel(f"{feature}")
    plt.ylabel("Number of jets")
    ax.set_yscale('log')
    plt.savefig(f"plots/features/{feature}.png")
    plt.close()

for feature in target_features_to_draw:
    fig, ax = plt.subplots()
    plt.hist(y[feature], bins=100, histtype='step', label=f"{feature}")
    plt.xlabel(f"RecoJet_{feature}")
    plt.ylabel("Number of jets")
    ax.set_yscale('log')
    plt.savefig(f"plots/features/recojet_{feature}.png")
    plt.close()

############# BDT training #################
train_features = ["pt", "eta", "phi", "muonRelIso", "egammaRelIso", "etSum", "htSum", "etMiss", "etMissPhi", "htMiss", "htMissPhi", "towerCount"]
target_features = ["recojet_pt"]

# Create the BDT dataset
bdt_dataset = BDTDataset(X, y, w, train_features, target_features)
print(f"BDT dataset created with {len(bdt_dataset)} samples")
