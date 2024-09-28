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

os.getcwd()

#Import the modules
from utils.loader import L1ScoutingDataset, collate_fn
from utils.transforms import transform_muon, transform_jet, transform_egamma
from mva.dnn import ObjectNN
from mva.bdt import ObjectBDT
from torch.utils.data import DataLoader

plot_dir = "/vols/cms/pb4918/L1Scouting/Sep24/PtRegression/CMSSW_14_1_0_pre4/src/plots"
os.makedirs(plot_dir, exist_ok=True)

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


################# Dataset loading #################

object_dataset = L1ScoutingDataset(dir_path="/vols/cms/pb4918/L1Scouting/Sep24/PtRegression/CMSSW_14_1_0_pre4/src/outputs/sample", var_dict=var_dict, transform=transform_muon, object_type="muon")

#Split into train and test
train_dataset, test_dataset = torch.utils.data.random_split(object_dataset, [0.5, 0.5])

#Data loader
train_loader = DataLoader(train_dataset, batch_size=40, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=40, shuffle=True, collate_fn=collate_fn)

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
    "input_size": 5,
    "hidden_size": 128,
    "output_size": 3
}

model = ObjectBDT(train_params=bdt_model_params, do_eval=True)

################ BDT training ###################
#Using only one "iteration"
X_train, y_train = next(iter(train_loader))
X_test, y_test = next(iter(test_loader))
#Flatten
X_train = X_train.flatten(start_dim=0, end_dim=1)
y_train = y_train.flatten(start_dim=0, end_dim=1)
X_test = X_test.flatten(start_dim=0, end_dim=1)
y_test = y_test.flatten(start_dim=0, end_dim=1)
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
    plt.savefig(f"{plot_dir}/muon_bdt_rmse.png")

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
hep.histplot(h_obj_pt_reco_reldiff, ax=ax, histtype="step", flow=None, color='blue', label='Reco')
hep.histplot(h_obj_pt_pred_reldiff, ax=ax, histtype="step", flow=None, color='red', label='Pred')
ax.set_xlabel("dpT/pT")
ax.set_ylabel("Events")
ax.legend()
hep.cms.label(llabel="Private Work", rlabel="Level-1 Scouting 2024", ax=ax)
plt.savefig(f"{plot_dir}/muon_bdt_pt_reldiff.png")

#eta
fig, ax = plt.subplots()
hep.histplot(h_obj_eta_reco_diff, ax=ax, histtype="step", flow=None, color='blue', label='Reco')
hep.histplot(h_obj_eta_pred_diff, ax=ax, histtype="step", flow=None, color='red', label='Pred')
ax.set_xlabel("dEta")
ax.set_ylabel("Events")
ax.legend()
hep.cms.label(llabel="Private Work", rlabel="Level-1 Scouting 2024", ax=ax)
plt.savefig(f"{plot_dir}/muon_bdt_eta_diff.png")

#phi
fig, ax = plt.subplots()
hep.histplot(h_obj_phi_reco_diff, ax=ax, histtype="step", flow=None, color='blue', label='Reco')
hep.histplot(h_obj_phi_pred_diff, ax=ax, histtype="step", flow=None, color='red', label='Pred')
ax.set_xlabel("dPhi")
ax.set_ylabel("Events")
ax.legend()
hep.cms.label(llabel="Private Work", rlabel="Level-1 Scouting 2024", ax=ax)
plt.savefig(f"{plot_dir}/muon_bdt_phi_diff.png")