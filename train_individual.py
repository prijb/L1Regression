#This code loads batches at a per object level as opposed to a per file level
#This is the master script for training the mva
import uproot   
import awkward as ak
import numpy as np
import pandas as pd
import os
import pickle
#Deep learning
import torch
import torch.nn as nn
#Plotting 
import matplotlib.pyplot as plt
import mplhep as hep
import hist
plt.style.use(hep.style.CMS)
plt.rcParams["figure.figsize"] = (12.5, 10)

#Memory profiling
from memory_profiler import profile

os.getcwd()

#Import the modules
from utils.loader import L1ScoutingDataset, collate_fn
from utils.loader_ind import L1ScoutingObject
from utils.transforms import transform_muon, transform_jet, transform_egamma
from mva.dnn import ObjectNN
from mva.bdt import ObjectBDT
from torch.utils.data import DataLoader

plot_dir = "/vols/cms/pb4918/L1Scouting/Sep24/PtRegression/CMSSW_14_1_0_pre4/src/plots"
os.makedirs(plot_dir, exist_ok=True)

#Set the mode
mva_mode = "dnn"
object_type = "muon"

######################## Variable definition ###################
from mva.vars import *

var_dict = {
    "read": READ_VARS,
    "jet": JET_VARS,
    "muon": MUON_VARS,
    "egamma": EGAMMA_VARS,
    "aux": AUX_VARS,
    "jet_target": JET_TARGET,
    "jet_target_rel": JET_TARGET_REL,
    "muon_target": MUON_TARGET,
    "muon_target_rel": MUON_TARGET_REL,
    "egamma_target": EGAMMA_TARGET,
    "egamma_target_rel": EGAMMA_TARGET_REL  
}

object_transform = None
if object_type == "jet":
    object_transform = transform_jet
elif object_type == "muon":
    object_transform = transform_muon
elif object_type == "egamma":
    object_transform = transform_egamma

################# Dataset loading #################
object_dataset = L1ScoutingObject(dir_path="/vols/cms/pb4918/L1Scouting/Sep24/PtRegression/CMSSW_14_1_0_pre4/src/outputs/sample", var_dict=var_dict, transform=object_transform, object_type=object_type, use_cache=True, cache_path=f"/vols/cms/pb4918/L1Scouting/Sep24/PtRegression/CMSSW_14_1_0_pre4/src/scripts/training/cache_{object_type}")
num_objects = len(object_dataset)

#Split into train and test
print("Loading train and test datasets")
num_train = int(0.7*num_objects)
num_test = num_objects - num_train
train_dataset, test_dataset = torch.utils.data.random_split(object_dataset, [num_train, num_test])

#Data loader
#train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
#test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)

print("Setting train and test loaders")
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)

#Statistics (mean, std) requires full dataset which is loaded in chunks of size 10000
print("Setting full dataset loader")

"""
full_loader_batch_size = 1000
full_loader = DataLoader(object_dataset, batch_size=full_loader_batch_size, shuffle=False)

print("Computing mean and std from full dataset")

total_sum_X = 0.0
total_sum_y = 0.0
total_sum_X2 = 0.0
total_sum_y2 = 0.0
total_count = 0

num_iterations = num_objects//full_loader_batch_size
num_iterations = 10
for i in range(num_iterations):
    print("Estimating statistics from iteration {}/{}".format(i+1, num_iterations))
    X_batch, y_batch = next(iter(full_loader))

    total_count += X_batch.size(0)
    total_sum_X += X_batch.sum(dim=0)
    total_sum_y += y_batch.sum(dim=0)
    total_sum_X2 += (X_batch**2).sum(dim=0)
    total_sum_y2 += (y_batch**2).sum(dim=0)

X_mean = total_sum_X/total_count
y_mean = total_sum_y/total_count
X_std = torch.sqrt(total_sum_X2/total_count - X_mean**2)
y_std = torch.sqrt(total_sum_y2/total_count - y_mean**2)
"""

#print("Getting full dataset iteration")
#X_full, y_full = next(iter(full_loader))
#Get the mean and std
#X_mean = X_full.mean(dim=0)
#X_std = X_full.std(dim=0)
#y_mean = y_full.mean(dim=0)
#y_std = y_full.std(dim=0)

#Using the dataset loader and not the object level one
object_dataset_full = L1ScoutingDataset(dir_path="/vols/cms/pb4918/L1Scouting/Sep24/PtRegression/CMSSW_14_1_0_pre4/src/outputs/sample", var_dict=var_dict, transform=object_transform, object_type=object_type)
X_full, y_full = next(iter(DataLoader(object_dataset_full, batch_size=10, shuffle=False, collate_fn=collate_fn))) 
X_full = X_full.flatten(start_dim=0, end_dim=1)
y_full = y_full.flatten(start_dim=0, end_dim=1)

#Filter out zero rows
print("Computing mean and std from full dataset")
mask = X_full[:, 0] > 0
X_full = X_full[mask]
y_full = y_full[mask]

X_mean = X_full.mean(dim=0)
X_std = X_full.std(dim=0)
y_mean = y_full.mean(dim=0)
y_std = y_full.std(dim=0)


print("\n Dataset statistics")
print(f"X mean: {X_mean}, X std: {X_std}")
print(f"y mean: {y_mean}, y std: {y_std}")

############ Model specific code ############

#Define the model parameters
bdt_model_params = {
    "subsample": 1.0,
    "max_depth": 10,
    "n_estimators": 500,
    "eta": 0.01,
    "reg_lambda": 0,
    "reg_alpha": 0,
    "multi_strategy": "multi_output_tree",
}

dnn_model_params = {
    "input_size": object_dataset.num_train_variables,
    "hidden_size": 128,
    "output_size": object_dataset.num_target_variables,
    "learning_rate": 1e-5,
    "num_epochs": 100
}


################### DNN training ##################
if mva_mode == "dnn":
    print("Training DNN")
    model = ObjectNN(input_size=dnn_model_params["input_size"], hidden_size=dnn_model_params["hidden_size"], output_size=dnn_model_params["output_size"])

    #Define the loss function
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=dnn_model_params["learning_rate"])
    num_epochs = dnn_model_params["num_epochs"]

    epochs = []
    train_losses = []
    test_losses = []
    
    #Training loop  
    for epoch in range(num_epochs):
        X_train, y_train = next(iter(train_loader))

        #Normalise using full dataset mean and std
        X_train = (X_train - X_mean)/X_std
        y_train = (y_train - y_mean)/y_std

        #Forward pass
        optimizer.zero_grad()
        y_pred = model(X_train.float())
        train_loss = criterion(y_pred, y_train.float())

        #Backward pass
        train_loss.backward()
        optimizer.step()

        #Test eval
        test_loss = 0
        with torch.no_grad():
            X_test, y_test = next(iter(test_loader))
            
            #Normalise using full dataset mean and std
            X_test = (X_test - X_mean)/X_std
            y_test = (y_test - y_mean)/y_std

            #Evaluate
            y_pred_test = model(X_test.float())
            test_loss = criterion(y_pred_test, y_test.float())
        
        #Store the losses
        epochs.append(epoch)
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())

        print(f"Epoch {epoch}, Train loss: {train_loss.item()}, Test loss: {test_loss.item()}")

    #Plot the loss
    print("Plotting model evaluation")
    fig, ax = plt.subplots()
    ax.plot(epochs, train_losses, label="Train")
    ax.plot(epochs, test_losses, label="Test")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    hep.cms.label(llabel="Private Work", rlabel="Level-1 Scouting 2024", ax=ax)
    plt.savefig(f"{plot_dir}/{object_type}_dnn_loss.png")

    #Get the prediction
    with torch.no_grad():
        #Use the full dataset for evaluation
        eval_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        X_test, y_test = next(iter(eval_loader))

        #Normalise using full dataset mean and std
        X_test = (X_test - X_mean)/X_std
        y_test = (y_test - y_mean)/y_std

        #Evaluate
        y_pred = model(X_test.float())

        #Denormalise
        X_test = X_test*X_std + X_mean  
        y_pred = y_pred*y_std + y_mean
        y_test = y_test*y_std + y_mean

    
################ BDT training ###################
if mva_mode == "bdt":
    print("Training BDT")
    model = ObjectBDT(train_params=bdt_model_params, do_eval=True)

    #Using only one "iteration"
    X_train, y_train = next(iter(train_loader))
    X_test, y_test = next(iter(test_loader))

    #Detach for BDT
    X_train = X_train.detach().numpy()
    y_train = y_train.detach().numpy()
    X_test = X_test.detach().numpy()
    y_test = y_test.detach().numpy()

    #Train the model
    model.train(X_train, y_train, X_test, y_test)

    #Evaluate the model
    epochs, results = model.evaluate(X_test, y_test)

    #Get the prediction
    y_pred = model.predict(X_test)

    #Plot the evaluation loss
    if model.do_eval:
        print("Plotting model evaluation")
        x_axis = range(0, epochs)

        #RMSE
        fig, ax = plt.subplots()
        ax.plot(x_axis, results["validation_0"]["rmse"], label="Train")
        ax.plot(x_axis, results["validation_1"]["rmse"], label="Test")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("RMSE")
        ax.legend()
        hep.cms.label(llabel="Private Work", rlabel="Level-1 Scouting 2024", ax=ax)
        plt.savefig(f"{plot_dir}/{object_type}_bdt_rmse.png")

################### Common plotting code (eval) ###################
#Plot the model performance
print("Plotting model performance")
obj_reco_pt = X_test[:, 0]
obj_reco_eta = X_test[:, 1]
obj_reco_phi = X_test[:, 2]

obj_true_pt = y_test[:, 0]*obj_reco_pt
obj_true_eta = y_test[:, 1] + obj_reco_eta
obj_true_phi = y_test[:, 2] + obj_reco_phi

obj_pred_pt = y_pred[:, 0]*obj_reco_pt
obj_pred_eta = y_pred[:, 1] + obj_reco_eta
obj_pred_phi = y_pred[:, 2] + obj_reco_phi

#pT
h_obj_pt_reco_reldiff = hist.Hist(hist.axis.Regular(50, -1, 1, name="dpT/pT"), storage=hist.storage.Weight())
h_obj_pt_pred_reldiff = hist.Hist(hist.axis.Regular(50, -1, 1, name="dpT/pT"), storage=hist.storage.Weight())
#eta
h_obj_eta_reco_diff = hist.Hist(hist.axis.Regular(50, -0.4, 0.4, name="dEta"), storage=hist.storage.Weight())
h_obj_eta_pred_diff = hist.Hist(hist.axis.Regular(50, -0.4, 0.4, name="dEta"), storage=hist.storage.Weight())
#phi
h_obj_phi_reco_diff = hist.Hist(hist.axis.Regular(50, -0.5, 0.5, name="dPhi"), storage=hist.storage.Weight())
h_obj_phi_pred_diff = hist.Hist(hist.axis.Regular(50, -0.5, 0.5, name="dPhi"), storage=hist.storage.Weight())

#Fill for objects with a reco pT in the 3.5-45 GeV range
pt_mask = (obj_reco_pt > 3.5) & (obj_reco_pt < 45)
h_obj_pt_reco_reldiff.fill((obj_reco_pt[pt_mask] - obj_true_pt[pt_mask])/obj_true_pt[pt_mask])
h_obj_pt_pred_reldiff.fill((obj_pred_pt[pt_mask] - obj_true_pt[pt_mask])/obj_true_pt[pt_mask])
h_obj_eta_reco_diff.fill(obj_reco_eta[pt_mask] - obj_true_eta[pt_mask])
h_obj_eta_pred_diff.fill(obj_pred_eta[pt_mask] - obj_true_eta[pt_mask])
h_obj_phi_reco_diff.fill(obj_reco_phi[pt_mask] - obj_true_phi[pt_mask])
h_obj_phi_pred_diff.fill(obj_pred_phi[pt_mask] - obj_true_phi[pt_mask])

#Plot
#pT
fig, ax = plt.subplots()
hep.histplot(h_obj_pt_reco_reldiff, ax=ax, histtype="step", flow=None, color='red', label='Reco', density=True)
hep.histplot(h_obj_pt_pred_reldiff, ax=ax, histtype="step", flow=None, color='blue', label='Pred', density=True)
ax.set_xlabel("dpT/pT")
ax.set_ylabel("Events")
ax.legend()
hep.cms.label(llabel="Private Work", rlabel="Level-1 Scouting 2024", ax=ax)
plt.savefig(f"{plot_dir}/{object_type}_{mva_mode}_pt_reldiff.png")

#eta
fig, ax = plt.subplots()
hep.histplot(h_obj_eta_reco_diff, ax=ax, histtype="step", flow=None, color='red', label='Reco', density=True)
hep.histplot(h_obj_eta_pred_diff, ax=ax, histtype="step", flow=None, color='blue', label='Pred', density=True)
ax.set_xlabel("dEta")
ax.set_ylabel("Events")
ax.legend()
hep.cms.label(llabel="Private Work", rlabel="Level-1 Scouting 2024", ax=ax)
plt.savefig(f"{plot_dir}/{object_type}_{mva_mode}_eta_diff.png")

#phi
fig, ax = plt.subplots()
hep.histplot(h_obj_phi_reco_diff, ax=ax, histtype="step", flow=None, color='red', label='Reco', density=True)
hep.histplot(h_obj_phi_pred_diff, ax=ax, histtype="step", flow=None, color='blue', label='Pred', density=True)
ax.set_xlabel("dPhi")
ax.set_ylabel("Events")
ax.legend()
hep.cms.label(llabel="Private Work", rlabel="Level-1 Scouting 2024", ax=ax)
plt.savefig(f"{plot_dir}/{object_type}_{mva_mode}_phi_diff.png")