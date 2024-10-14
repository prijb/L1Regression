#This code loads batches at a per object level as opposed to a per file level
#Faster than train_individual
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
from torch.utils.data import DataLoader
from torch.utils.data import BatchSampler, RandomSampler
#Plotting 
import matplotlib.pyplot as plt
import mplhep as hep
import hist
plt.style.use(hep.style.CMS)
plt.rcParams["figure.figsize"] = (12.5, 10)

os.getcwd()

#Import the modules
from utils.preprocess_inputs import Preprocessor
from utils.transforms import transform_muon, transform_jet, transform_egamma
from utils.dataset import L1ObjectDataset
from mva.dnn import ObjectNN
from mva.bdt import ObjectBDT

plot_dir = "/vols/cms/pb4918/L1Scouting/Sep24/PtRegression/CMSSW_14_1_0_pre4/src/plots"
os.makedirs(plot_dir, exist_ok=True)
mva_mode = "dnn"
object_type = "jet"

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
#Extract a train test split proportion of files
train_test_split = 0.8
input_dir = "/vols/cms/pb4918/L1Scouting/Sep24/PtRegression/CMSSW_14_1_0_pre4/src/outputs/sample"
file_list = os.listdir(input_dir)
file_list = [f"{input_dir}/{f}" for f in file_list if "root" in f]
train_files = file_list[:int(train_test_split*len(file_list))]
test_files = file_list[int(train_test_split*len(file_list)):]

#Preprocess the data
preprocessor_train = Preprocessor(file_list=train_files, var_dict=var_dict, transform=object_transform, object_type=object_type, cache_dir=f"/vols/cms/pb4918/L1Scouting/Sep24/PtRegression/CMSSW_14_1_0_pre4/src/scripts/training/cache_{object_type}_train", use_existing_cache=False)
preprocessor_test = Preprocessor(file_list=test_files, var_dict=var_dict, transform=object_transform, object_type=object_type, cache_dir=f"/vols/cms/pb4918/L1Scouting/Sep24/PtRegression/CMSSW_14_1_0_pre4/src/scripts/training/cache_{object_type}_test", use_existing_cache=False)

print("\nPreprocessing training data")
preprocessor_train.cache_files()
print("\nPreprocessing testing data")
preprocessor_test.cache_files()

#Get the training and test datasets
X_train, y_train = preprocessor_train.get_X_y()
X_test, y_test = preprocessor_test.get_X_y()
train_dataset = L1ObjectDataset(X_train, y_train)
test_dataset = L1ObjectDataset(X_test, y_test)

n_train_samples = len(train_dataset)
n_test_samples = len(test_dataset)

print("\nDataset loaded with")
print("Train variables: ", preprocessor_train.train_variables)
print("Target variables: ", preprocessor_train.target_variables)

#Get statistics from training data
X_mean = X_train.mean(dim=0)
X_std = X_train.std(dim=0)
y_mean = y_train.mean(dim=0)
y_std = y_train.std(dim=0)
print("\n Dataset statistics")
print(f"X mean: {X_mean}, X std: {X_std}")
print(f"y mean: {y_mean}, y std: {y_std}")

#Dataloader
#BDT
#batch_size_train = 100000
#batch_size_test = 100000
#DNN
batch_size_train = 256
batch_size_test = 256
batch_size_eval = 500000
#If batch size is bigger than the dataset size, cap it to the dataset size
if batch_size_train > n_train_samples:
    batch_size_train = n_train_samples
    print(f"Train batch size capped to {n_train_samples}")
if batch_size_test > n_test_samples:
    batch_size_test = n_test_samples
    print(f"Test batch size capped to {n_test_samples}")
if batch_size_eval > n_test_samples:
    batch_size_eval = n_test_samples
    print(f"Eval batch size capped to {n_test_samples}")


train_sampler = BatchSampler(RandomSampler(train_dataset), batch_size=batch_size_train, drop_last=True)
test_sampler = BatchSampler(RandomSampler(test_dataset), batch_size=batch_size_test, drop_last=True)

train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
test_loader = DataLoader(test_dataset, batch_sampler=test_sampler)

############ Model specific code ############

#Define the model parameters
bdt_model_params = {
    "subsample": 0.1,
    "max_depth": 5,
    "n_estimators": 500,
    "eta": 0.01,
    "reg_lambda": 0,
    "reg_alpha": 0,
    "multi_strategy": "multi_output_tree",
}

dnn_model_params = {
    "input_size": preprocessor_train.num_train_variables,
    "hidden_size": 128,
    "output_size": preprocessor_train.num_target_variables,
    "learning_rate": 1e-3,
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

    train_loss_sum = 0
    test_loss_sum = 0
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
        train_loss_sum += train_loss.item() 

        #Backward pass
        train_loss.backward()
        optimizer.step()

        #Test loss
        with torch.no_grad():
            X_test, y_test = next(iter(test_loader))

            #Normalise using full dataset mean and std
            X_test = (X_test - X_mean)/X_std
            y_test = (y_test - y_mean)/y_std

            #Evaluate
            y_pred_test = model(X_test.float())
            test_loss = criterion(y_pred_test, y_test.float())
            test_loss_sum += test_loss.item()
        
        #Store the losses
        epochs.append(epoch)
        train_losses.append(train_loss_sum/(epoch + 1))
        test_losses.append(test_loss_sum/(epoch + 1))

        print(f"Epoch {epoch}, Train loss (mean): {train_loss_sum/(epoch + 1)}, Test loss (mean): {test_loss_sum/(epoch + 1)}")

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
        eval_sampler = BatchSampler(RandomSampler(test_dataset), batch_size=batch_size_eval, drop_last=True)
        eval_loader = DataLoader(test_dataset, batch_sampler=eval_sampler)
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

#Relative errors
#pT
h_obj_pt_reco_reldiff = hist.Hist(hist.axis.Regular(50, -1, 1, name="dpT/pT"), storage=hist.storage.Weight())
h_obj_pt_pred_reldiff = hist.Hist(hist.axis.Regular(50, -1, 1, name="dpT/pT"), storage=hist.storage.Weight())
#eta
h_obj_eta_reco_reldiff = hist.Hist(hist.axis.Regular(50, -1, 1, name="dEta"), storage=hist.storage.Weight())
h_obj_eta_pred_reldiff = hist.Hist(hist.axis.Regular(50, -1, 1, name="dEta"), storage=hist.storage.Weight())
#phi
h_obj_phi_reco_reldiff = hist.Hist(hist.axis.Regular(50, -1, 1, name="dPhi"), storage=hist.storage.Weight())
h_obj_phi_pred_reldiff = hist.Hist(hist.axis.Regular(50, -1, 1, name="dPhi"), storage=hist.storage.Weight())


#Absolute errors
#pT
h_obj_pt_reco_diff = hist.Hist(hist.axis.Regular(50, -100, 100, name="dpT"), storage=hist.storage.Weight())
h_obj_pt_pred_diff = hist.Hist(hist.axis.Regular(50, -100, 100, name="dpT"), storage=hist.storage.Weight())
#eta
h_obj_eta_reco_diff = hist.Hist(hist.axis.Regular(50, -0.5, 0.5, name="dEta"), storage=hist.storage.Weight())
h_obj_eta_pred_diff = hist.Hist(hist.axis.Regular(50, -0.5, 0.5, name="dEta"), storage=hist.storage.Weight())
#phi
h_obj_phi_reco_diff = hist.Hist(hist.axis.Regular(50, -0.5, 0.5, name="dPhi"), storage=hist.storage.Weight())
h_obj_phi_pred_diff = hist.Hist(hist.axis.Regular(50, -0.5, 0.5, name="dPhi"), storage=hist.storage.Weight())

#Add a pT mask for filling
#pt_mask = (obj_reco_pt > 3.5) & (obj_reco_pt < 45)
pt_mask = (obj_reco_pt > 0.)

#Fill relative errors
h_obj_pt_reco_reldiff.fill((obj_reco_pt[pt_mask] - obj_true_pt[pt_mask])/obj_true_pt[pt_mask])
h_obj_pt_pred_reldiff.fill((obj_pred_pt[pt_mask] - obj_true_pt[pt_mask])/obj_true_pt[pt_mask])

h_obj_eta_reco_reldiff.fill((obj_reco_eta[pt_mask] - obj_true_eta[pt_mask])/obj_true_eta[pt_mask])
h_obj_eta_pred_reldiff.fill((obj_pred_eta[pt_mask] - obj_true_eta[pt_mask])/obj_true_eta[pt_mask])

h_obj_phi_reco_reldiff.fill((obj_reco_phi[pt_mask] - obj_true_phi[pt_mask])/obj_true_phi[pt_mask])
h_obj_phi_pred_reldiff.fill((obj_pred_phi[pt_mask] - obj_true_phi[pt_mask])/obj_true_phi[pt_mask])

#Fill absolute errors
h_obj_pt_reco_diff.fill(obj_reco_pt[pt_mask] - obj_true_pt[pt_mask])
h_obj_pt_pred_diff.fill(obj_pred_pt[pt_mask] - obj_true_pt[pt_mask])

h_obj_eta_reco_diff.fill(obj_reco_eta[pt_mask] - obj_true_eta[pt_mask])
h_obj_eta_pred_diff.fill(obj_pred_eta[pt_mask] - obj_true_eta[pt_mask])

h_obj_phi_reco_diff.fill(obj_reco_phi[pt_mask] - obj_true_phi[pt_mask])
h_obj_phi_pred_diff.fill(obj_pred_phi[pt_mask] - obj_true_phi[pt_mask])

#Plot the kinematics
pt_lim = [0, 50]
eta_lim = [-5, 5]
phi_lim = [-3.14, 3.14]

if object_type == "muon":
    pt_lim = [3, 255]
    eta_lim = [-2.5, 2.5]
elif object_type == "egamma":
    pt_lim = [30, 255]
else:
    pt_lim = [30, 300]

h_obj_pt_truth = hist.Hist(hist.axis.Regular(50, pt_lim[0], pt_lim[1], name="pT [GeV]"), storage=hist.storage.Weight())
h_obj_pt_reco = hist.Hist(hist.axis.Regular(50, pt_lim[0], pt_lim[1], name="pT [GeV]"), storage=hist.storage.Weight())
h_obj_pt_pred = hist.Hist(hist.axis.Regular(50, pt_lim[0], pt_lim[1], name="pT [GeV]"), storage=hist.storage.Weight())

h_obj_eta_truth = hist.Hist(hist.axis.Regular(50, eta_lim[0], eta_lim[1], name="Eta"), storage=hist.storage.Weight())
h_obj_eta_reco = hist.Hist(hist.axis.Regular(50, eta_lim[0], eta_lim[1], name="Eta"), storage=hist.storage.Weight())
h_obj_eta_pred = hist.Hist(hist.axis.Regular(50, eta_lim[0], eta_lim[1], name="Eta"), storage=hist.storage.Weight())

h_obj_phi_truth = hist.Hist(hist.axis.Regular(50, phi_lim[0], phi_lim[1], name="Phi"), storage=hist.storage.Weight())
h_obj_phi_reco = hist.Hist(hist.axis.Regular(50, phi_lim[0], phi_lim[1], name="Phi"), storage=hist.storage.Weight())
h_obj_phi_pred = hist.Hist(hist.axis.Regular(50, phi_lim[0], phi_lim[1], name="Phi"), storage=hist.storage.Weight())

h_obj_pt_truth.fill(obj_true_pt[pt_mask])
h_obj_pt_reco.fill(obj_reco_pt[pt_mask])
h_obj_pt_pred.fill(obj_pred_pt[pt_mask])

h_obj_eta_truth.fill(obj_true_eta[pt_mask])
h_obj_eta_reco.fill(obj_reco_eta[pt_mask])
h_obj_eta_pred.fill(obj_pred_eta[pt_mask])

h_obj_phi_truth.fill(obj_true_phi[pt_mask])
h_obj_phi_reco.fill(obj_reco_phi[pt_mask])
h_obj_phi_pred.fill(obj_pred_phi[pt_mask])


#Plot absolute and relative errors on the same plot
#pT 
fig, axs = plt.subplots(1, 2, figsize=(25, 10))
#Absolute
hep.histplot(h_obj_pt_reco_diff, ax=axs[0], histtype="step", flow=None, color='red', label='L1', density=True)
hep.histplot(h_obj_pt_pred_diff, ax=axs[0], histtype="step", flow=None, color='blue', label='Pred', density=True)
axs[0].set_xlabel(r"$\mathrm{ p_{T}^{L1} - p_{T}^{Gen} }$ [GeV]")
axs[0].set_ylabel("Counts")
axs[0].set_yscale("log")
axs[0].legend()
#Relative
hep.histplot(h_obj_pt_reco_reldiff, ax=axs[1], histtype="step", flow=None, color='red', label='L1', density=True)
hep.histplot(h_obj_pt_pred_reldiff, ax=axs[1], histtype="step", flow=None, color='blue', label='Pred', density=True)
#axs[1].set_xlabel("dpT/pT")
axs[1].set_xlabel(r"$\mathrm{ \frac{ p_{T}^{L1} - p_{T}^{Gen} }{ p_{T}^{Gen} } }$")
axs[1].set_ylabel("Counts")
axs[1].legend()
#hep.cms.label(llabel="Private Work", rlabel="Level-1 Scouting 2024", ax=axs[0])
hep.cms.text("Private Work", ax=axs[0])
hep.cms.lumitext("Level-1 Scouting 2024", ax=axs[1])
plt.tight_layout()
plt.savefig(f"{plot_dir}/{object_type}_{mva_mode}_pt_err.png")

#eta
fig, axs = plt.subplots(1, 2, figsize=(25, 10))
#Absolute
hep.histplot(h_obj_eta_reco_diff, ax=axs[0], histtype="step", flow=None, color='red', label='L1', density=True)
hep.histplot(h_obj_eta_pred_diff, ax=axs[0], histtype="step", flow=None, color='blue', label='Pred', density=True)
#axs[0].set_xlabel("dEta")
axs[0].set_xlabel(r"$\mathrm{ \eta^{L1} - \eta^{Gen} }$")
axs[0].set_ylabel("Counts")
#axs[0].set_yscale("log")
axs[0].legend()
#Relative
hep.histplot(h_obj_eta_reco_reldiff, ax=axs[1], histtype="step", flow=None, color='red', label='L1', density=True)
hep.histplot(h_obj_eta_pred_reldiff, ax=axs[1], histtype="step", flow=None, color='blue', label='Pred', density=True)
#axs[1].set_xlabel("dEta/Eta")
axs[1].set_xlabel(r"$\mathrm{ \frac{ \eta^{L1} - \eta^{Gen} }{ \eta^{Gen} } }$")
axs[1].set_ylabel("Counts")
axs[1].legend()
hep.cms.text("Private Work", ax=axs[0])
hep.cms.lumitext("Level-1 Scouting 2024", ax=axs[1])
plt.tight_layout()
plt.savefig(f"{plot_dir}/{object_type}_{mva_mode}_eta_err.png")

#phi
fig, axs = plt.subplots(1, 2, figsize=(25, 10))
#Absolute
hep.histplot(h_obj_phi_reco_diff, ax=axs[0], histtype="step", flow=None, color='red', label='L1', density=True)
hep.histplot(h_obj_phi_pred_diff, ax=axs[0], histtype="step", flow=None, color='blue', label='Pred', density=True)
#axs[0].set_xlabel("dPhi")
axs[0].set_xlabel(r"$\mathrm{ \phi^{L1} - \phi^{Gen} }$")
axs[0].set_ylabel("Counts")
#axs[0].set_yscale("log")
axs[0].legend()
#Relative
hep.histplot(h_obj_phi_reco_reldiff, ax=axs[1], histtype="step", flow=None, color='red', label='L1', density=True)
hep.histplot(h_obj_phi_pred_reldiff, ax=axs[1], histtype="step", flow=None, color='blue', label='Pred', density=True)
#axs[1].set_xlabel("dPhi/Phi")
axs[1].set_xlabel(r"$\mathrm{ \frac{ \phi^{L1} - \phi^{Gen} }{ \phi^{Gen} } }$")
axs[1].set_ylabel("Counts")
axs[1].legend()
hep.cms.text("Private Work", ax=axs[0])
hep.cms.lumitext("Level-1 Scouting 2024", ax=axs[1])
plt.tight_layout()
plt.savefig(f"{plot_dir}/{object_type}_{mva_mode}_phi_err.png")

#Plot the kinematics
#pT
fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
#Upper axis
hep.histplot(h_obj_pt_truth, ax=axs[0], histtype="step", flow=None, color='black', label='Truth', density=True)
hep.histplot(h_obj_pt_reco, ax=axs[0], histtype="step", flow=None, color='red', label='L1', density=True)
hep.histplot(h_obj_pt_pred, ax=axs[0], histtype="step", flow=None, color='blue', label='Pred', density=True)
axs[0].set_ylabel("Counts")
axs[0].set_xlabel("")
axs[0].set_yscale("log")
axs[0].legend()
#Lower axis
axs[1].axhline(1, color='black', linestyle='--')
ratio_l1 = h_obj_pt_reco.values()/h_obj_pt_truth.values()
ratio_l1_err = ratio_l1*np.sqrt(1/h_obj_pt_reco.values() + 1/h_obj_pt_truth.values())
ratio_pred = h_obj_pt_pred.values()/h_obj_pt_truth.values()
ratio_pred_err = ratio_pred*np.sqrt(1/h_obj_pt_pred.values() + 1/h_obj_pt_truth.values())

hep.histplot(ratio_l1, h_obj_pt_truth.axes[0].edges, ax=axs[1], histtype="step", yerr=False, flow=None, color='red', label='L1', density=False)
hep.histplot(ratio_pred, h_obj_pt_truth.axes[0].edges, ax=axs[1], histtype="step", yerr=False, flow=None, color='blue', label='Pred', density=False)
axs[1].fill_between(h_obj_pt_truth.axes[0].edges[:-1], ratio_l1 - ratio_l1_err, ratio_l1 + ratio_l1_err, 
                    step="post", color='red', alpha=0.2)
axs[1].fill_between(h_obj_pt_truth.axes[0].edges[:-1], ratio_pred - ratio_pred_err, ratio_pred + ratio_pred_err, 
                    step="post", color='blue', alpha=0.2)
#axs[1].set_xlabel("pT [GeV]")
axs[1].set_xlabel(r"$\mathrm{ p_{T} }$ [GeV]")
axs[1].set_ylabel("Ratio")
axs[1].set_ylim(0.5, 1.5)
axs[1].set_xlim(pt_lim)
hep.cms.label(llabel="Private Work", rlabel="Level-1 Scouting 2024", ax=axs[0])
plt.subplots_adjust(wspace=0, hspace=0.04)
plt.savefig(f"{plot_dir}/{object_type}_{mva_mode}_pt.png")

#eta
fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
#Upper axis
hep.histplot(h_obj_eta_truth, ax=axs[0], histtype="step", flow=None, color='black', label='Truth', density=True)
hep.histplot(h_obj_eta_reco, ax=axs[0], histtype="step", flow=None, color='red', label='L1', density=True)
hep.histplot(h_obj_eta_pred, ax=axs[0], histtype="step", flow=None, color='blue', label='Pred', density=True)
axs[0].set_ylabel("Counts")
axs[0].set_xlabel("")
axs[0].legend()
#Lower axis
axs[1].axhline(1, color='black', linestyle='--')
ratio_l1 = h_obj_eta_reco.values()/h_obj_eta_truth.values()
ratio_l1_err = ratio_l1*np.sqrt(1/h_obj_eta_reco.values() + 1/h_obj_eta_truth.values())
ratio_pred = h_obj_eta_pred.values()/h_obj_eta_truth.values()
ratio_pred_err = ratio_pred*np.sqrt(1/h_obj_eta_pred.values() + 1/h_obj_eta_truth.values())

hep.histplot(ratio_l1, h_obj_eta_truth.axes[0].edges, ax=axs[1], histtype="step", yerr=False, flow=None, color='red', label='L1', density=False)
hep.histplot(ratio_pred, h_obj_eta_truth.axes[0].edges, ax=axs[1], histtype="step", yerr=False, flow=None, color='blue', label='Pred', density=False)
axs[1].fill_between(h_obj_eta_truth.axes[0].edges[:-1], ratio_l1 - ratio_l1_err, ratio_l1 + ratio_l1_err,
                    step="post", color='red', alpha=0.2)
axs[1].fill_between(h_obj_eta_truth.axes[0].edges[:-1], ratio_pred - ratio_pred_err, ratio_pred + ratio_pred_err,
                    step="post", color='blue', alpha=0.2)
#axs[1].set_xlabel("Eta")
axs[1].set_xlabel(r"$\mathrm{ \eta }$")
axs[1].set_ylabel("Ratio")
axs[1].set_ylim(0.5, 1.5)
axs[1].set_xlim(eta_lim)
hep.cms.label(llabel="Private Work", rlabel="Level-1 Scouting 2024", ax=axs[0])
plt.subplots_adjust(wspace=0, hspace=0.04)
plt.savefig(f"{plot_dir}/{object_type}_{mva_mode}_eta.png")

#phi
fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
#Upper axis
hep.histplot(h_obj_phi_truth, ax=axs[0], histtype="step", flow=None, color='black', label='Truth', density=True)
hep.histplot(h_obj_phi_reco, ax=axs[0], histtype="step", flow=None, color='red', label='L1', density=True)
hep.histplot(h_obj_phi_pred, ax=axs[0], histtype="step", flow=None, color='blue', label='Pred', density=True)
axs[0].set_ylabel("Counts")
axs[0].set_xlabel("")
axs[0].legend()
#Lower axis
axs[1].axhline(1, color='black', linestyle='--')
ratio_l1 = h_obj_phi_reco.values()/h_obj_phi_truth.values()
ratio_l1_err = ratio_l1*np.sqrt(1/h_obj_phi_reco.values() + 1/h_obj_phi_truth.values())
ratio_pred = h_obj_phi_pred.values()/h_obj_phi_truth.values()
ratio_pred_err = ratio_pred*np.sqrt(1/h_obj_phi_pred.values() + 1/h_obj_phi_truth.values())

hep.histplot(ratio_l1, h_obj_phi_truth.axes[0].edges, ax=axs[1], histtype="step", yerr=False, flow=None, color='red', label='L1', density=False)
hep.histplot(ratio_pred, h_obj_phi_truth.axes[0].edges, ax=axs[1], histtype="step", yerr=False, flow=None, color='blue', label='Pred', density=False)
axs[1].fill_between(h_obj_phi_truth.axes[0].edges[:-1], ratio_l1 - ratio_l1_err, ratio_l1 + ratio_l1_err,
                    step="post", color='red', alpha=0.2)
axs[1].fill_between(h_obj_phi_truth.axes[0].edges[:-1], ratio_pred - ratio_pred_err, ratio_pred + ratio_pred_err,
                    step="post", color='blue', alpha=0.2)
#axs[1].set_xlabel("Phi")
axs[1].set_xlabel(r"$\mathrm{ \phi }$")
axs[1].set_ylabel("Ratio")
axs[1].set_ylim(0.5, 1.5)
axs[1].set_xlim(phi_lim)
hep.cms.label(llabel="Private Work", rlabel="Level-1 Scouting 2024", ax=axs[0])
plt.subplots_adjust(wspace=0, hspace=0.04)
plt.savefig(f"{plot_dir}/{object_type}_{mva_mode}_phi.png")




