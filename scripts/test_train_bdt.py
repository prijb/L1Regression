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

#sample = "ttbar"
sample = "qcd_15to7000"

# Plot the training features
plot_features = False

#train_cachedir = f"{cwd}/cachedir/{sample}/train"
#valid_cachedir = f"{cwd}/cachedir/{sample}/validation"
#test_cachedir = f"{cwd}/cachedir/{sample}/test"

#train_cachedir = f"{cwd}/cachedir/ttbar/train"
#valid_cachedir = f"{cwd}/cachedir/ttbar/validation"
#test_cachedir = f"{cwd}/cachedir/qcd_15to7000/train"

train_cachedir = f"{cwd}/cachedir/qcd_15to7000/train"
valid_cachedir = f"{cwd}/cachedir/qcd_15to7000/validation"
test_cachedir = f"{cwd}/cachedir/ttbar/train"

from utils.preprocess import Preprocessor
from utils.bdt_dataset import BDTDataset

input_files = ["dummy_file.root"]

preprocessor_train = Preprocessor(input_files, batch_size=100000, use_existing_cache=True, cache_dir=train_cachedir)
preprocessor_valid = Preprocessor(input_files, batch_size=100000, use_existing_cache=True, cache_dir=valid_cachedir)
preprocessor_test = Preprocessor(input_files, batch_size=100000, use_existing_cache=True, cache_dir=test_cachedir)

X_train, y_train, w_train = preprocessor_train.get_data_dict()
X_valid, y_valid, w_valid = preprocessor_valid.get_data_dict()
X_test, y_test, w_test = preprocessor_test.get_data_dict()

############# Data exploration #################
print(f"Loaded training data with {len(X_train)} jets")
import mplhep as hep
plt.style.use(hep.style.CMS)

if plot_features:
    print("Plotting training features...")
    jet_features_to_draw = ["pt", "eta", "phi", "muonRelIso", "egammaRelIso"]
    global_features_to_draw = ["etSum", "htSum", "etMiss", "etMissPhi", "htMiss", "htMissPhi", "towerCount"]
    target_features_to_draw = ["recojet_pt", "recojet_eta", "recojet_phi", "recojet_ptdiff", "recojet_ptratio", "recojet_etadiff", "recojet_phidiff", "recojet_btag"]

    for feature in jet_features_to_draw:
        fig, ax = plt.subplots()
        plt.hist(X_train[feature], bins=100, histtype='step', label=f"L1Jet_{feature}")
        plt.xlabel(f"L1Jet_{feature}")
        plt.ylabel("Number of jets")
        ax.set_yscale('log')
        plt.savefig(f"plots/features/l1jet_{feature}.png")
        plt.close()

    for feature in global_features_to_draw:
        fig, ax = plt.subplots()
        plt.hist(X_train[feature], bins=100, histtype='step', label=f"{feature}")
        plt.xlabel(f"{feature}")
        plt.ylabel("Number of jets")
        ax.set_yscale('log')
        plt.savefig(f"plots/features/{feature}.png")
        plt.close()

    for feature in target_features_to_draw:
        fig, ax = plt.subplots()
        plt.hist(y_train[feature], bins=100, histtype='step', label=f"{feature}")
        plt.xlabel(f"RecoJet_{feature}")
        plt.ylabel("Number of jets")
        ax.set_yscale('log')
        plt.savefig(f"plots/features/recojet_{feature}.png")
        plt.close()


############# BDT training #################
#train_features = ["pt", "eta", "phi", "muonRelIso", "egammaRelIso", "etSum", "htSum", "etMiss", "etMissPhi", "htMiss", "htMissPhi", "towerCount", "muonIso", "egammaIso"]
#train_features = ["pt", "eta", "phi", "egammaRelIso", "muonRelIso", "htSum", "htMiss", "towerCount"]
train_features = ["pt", "eta", "phi", "egammaRelIso", "muonRelIso"]
#train_features = ["pt", "eta", "phi"]
#target_features = ["recojet_pt"]
#target_features = ["recojet_ptdiff"]
target_features = ["recojet_ptratio"]
#target_features = ["recojet_pt", "recojet_eta", "recojet_phi"]
#target_features = ["recojet_ptratio", "recojet_eta", "recojet_phi"]

# Create the BDT datasets
train_dataset = BDTDataset(X_train, y_train, w_train, train_features, target_features)
val_dataset = BDTDataset(X_valid, y_valid, w_valid, train_features, target_features)
test_dataset = BDTDataset(X_test, y_test, w_test, train_features, target_features)

n_jets = len(train_dataset) + len(val_dataset) + len(test_dataset)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# BDT parameters

from mva.bdt_regressor import BDTRegressor
model = BDTRegressor(do_eval=True)

X_train, y_train, w_train = train_dataset[:]
X_val, y_val, w_val = val_dataset[:]
X_test, y_test, w_test = test_dataset[:]


print(f"\nTraining BDT")
print(f"Training features: {train_features}")
print(f"Target features: {target_features}")

X_train = X_train.detach().numpy()
y_train = y_train.detach().numpy()
w_train = w_train.detach().numpy()
X_val = X_val.detach().numpy()
y_val = y_val.detach().numpy()
w_val = w_val.detach().numpy()

model.train([X_train, y_train, w_train], [X_val, y_val, w_val])
epochs, results = model.evaluate(X_val, y_val)
print(f"BDT training finished after {epochs} epochs")

# Save the model
model.save_model(f"models/bdt_model_{sample}.json")

# Plot evaluation results
if model.do_eval:
    print("Plotting evaluation results...")
    x_axis = np.arange(epochs)

    # RMSE 
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
    ax.plot(x_axis, results['validation_1']['rmse'], label='Validation')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('RMSE')
    ax.legend()
    plt.savefig(f"plots/training/bdt_rmse.png")

    # MAE
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['mae'], label='Train')
    ax.plot(x_axis, results['validation_1']['mae'], label='Validation')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('MAE')
    ax.legend()
    plt.savefig(f"plots/training/bdt_mae.png")

    # Plot the feature importances if there's only one target feature
    if len(target_features) == 1:
        from xgboost import plot_importance
        feature_names_orig = model.model.get_booster().feature_names

        # Total gain
        model.model.get_booster().feature_names = train_features
        fig, ax = plt.subplots()
        plot_importance(model.model, ax=ax, importance_type='total_gain', show_values=False)
        ax.set_title('Feature Importance (Total Gain)')
        plt.tight_layout()
        plt.savefig(f"plots/training/bdt_feature_importance_total_gain.png")

        # Total cover
        fig, ax = plt.subplots()
        model.model.get_booster().feature_names = train_features
        plot_importance(model.model, ax=ax, importance_type='total_cover', show_values=False)
        ax.set_title('Feature Importance (Total Cover)')
        plt.tight_layout()
        plt.savefig(f"plots/training/bdt_feature_importance_total_cover.png")

        # Total weight
        fig, ax = plt.subplots()
        model.model.get_booster().feature_names = train_features
        plot_importance(model.model, ax=ax, importance_type='weight', show_values=False)
        ax.set_title('Feature Importance (Weight)')
        plt.tight_layout()
        plt.savefig(f"plots/training/bdt_feature_importance_weight.png")

        model.model.get_booster().feature_names = feature_names_orig 

############# Evaluation #################
from scipy.optimize import curve_fit
# Gaussian function to fit any distributions
def gaussian(x, mu, sigma, A=1.0):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

y_pred = model.predict(X_test)

# Define the predicted and true reco values (This depends on your target feature)
# If regressing to recojet pt
#l1jet_pt = X_test[:, 0].detach().numpy() 
#if y_pred.ndim == 1: l1jet_regressed_pt = y_pred
#else: l1jet_regressed_pt = y_pred[:, 0]
#recojet_pt = y_test[:, 0].detach().numpy() 

# If regressing to recojet ptdiff
#l1jet_pt = X_test[:, 0].detach().numpy()
#if y_pred.ndim == 1: l1jet_regressed_pt = l1jet_pt + y_pred
#else: l1jet_regressed_pt = l1jet_pt + y_pred[:, 0]
#recojet_pt = l1jet_pt + y_test[:, 0].detach().numpy()

# If regressing to recojet ptratio
l1jet_pt = X_test[:, 0].detach().numpy()
if y_pred.ndim == 1: l1jet_regressed_pt = l1jet_pt * y_pred
else: l1jet_regressed_pt = l1jet_pt * y_pred[:, 0]
recojet_pt = l1jet_pt * y_test[:, 0].detach().numpy()

# Plotting the results
# pT distribution
y_vals, x_edges = np.histogram(l1jet_pt, bins=100, range=(0, 1000))
y_vals_regressed, x_edges_regressed = np.histogram(l1jet_regressed_pt, bins=100, range=(0, 1000))
y_vals_reco, x_edges_reco = np.histogram(recojet_pt, bins=100, range=(0, 1000))
x_centers = 0.5 * (x_edges[1:] +  x_edges[:-1])
fig, axs = plt.subplots(2, 1, gridspec_kw=dict(height_ratios=[3, 1], hspace=0.1), sharex=True)
axs[0].hist(l1jet_pt, bins=100, range=(0, 1000), histtype='step', label='L1 Jet pT', color='red')
axs[0].hist(l1jet_regressed_pt, bins=100, range=(0, 1000), histtype='step', label='Regressed L1 Jet pT', color='blue')
axs[0].hist(recojet_pt, bins=100, range=(0, 1000), histtype='step', label='Reco Jet pT', color='black')
axs[0].set_xlabel("")
axs[0].set_ylabel("Number of jets")
axs[0].legend()
axs[0].set_xlim(0, 1000)
axs[0].set_yscale('log')
axs[0].axvline(30, color='black', linestyle='--', label='30 GeV threshold')
# Ratio plot
ratio_l1_err = (y_vals / y_vals_reco) * np.sqrt((1 / y_vals_reco) + (1 / y_vals))
ratio_l1_regressed_err = (y_vals_regressed / y_vals_reco) * np.sqrt((1 / y_vals_reco) + (1 / y_vals_regressed))
axs[1].errorbar(x_centers, y_vals / y_vals_reco, yerr=ratio_l1_err, fmt='o', label='L1 Jet pT / Reco Jet pT', color='red')
axs[1].errorbar(x_centers, y_vals_regressed / y_vals_reco, yerr=ratio_l1_regressed_err, fmt='o', label='Regressed L1 Jet pT / Reco Jet pT', color='blue')
axs[1].axhline(1, color='black', linestyle='--')
axs[1].axvline(30, color='black', linestyle='--')
axs[1].set_xlabel("Jet pT [GeV]")
axs[1].set_ylabel("L1/Reco")
axs[1].set_ylim(0, 2)
plt.savefig(f"plots/training/jet_pt_distribution.png")

# pT diff
pt_diff = l1jet_pt - recojet_pt
pt_diff_regressed = l1jet_regressed_pt - recojet_pt
# Fit a Gaussian near peak
y_vals, x_edges = np.histogram(pt_diff, bins=100, range=(-150, 150))
y_vals_regressed, x_edges_regressed = np.histogram(pt_diff_regressed, bins=100, range=(-150, 150))
x_centers = 0.5 * (x_edges[1:] + x_edges[:-1])
y_vals_to_fit = y_vals[min(0, np.argmax(y_vals) - 10):max(len(y_vals), np.argmax(y_vals) + 10)]
x_centers_to_fit = x_centers[min(0, np.argmax(y_vals) - 10):max(len(y_vals), np.argmax(y_vals) + 10)]
y_vals_regressed_to_fit = y_vals_regressed[min(0, np.argmax(y_vals_regressed) - 10):max(len(y_vals_regressed), np.argmax(y_vals_regressed) + 10)]
x_centers_regressed_to_fit = x_centers[min(0, np.argmax(y_vals_regressed) - 10):max(len(y_vals_regressed), np.argmax(y_vals_regressed) + 10)]
popt, _ = curve_fit(gaussian, x_centers_to_fit, y_vals_to_fit, p0=[0, np.std(pt_diff), np.sum(y_vals_to_fit)])
popt_regressed, _ = curve_fit(gaussian, x_centers_regressed_to_fit, y_vals_regressed_to_fit, p0=[0, np.std(pt_diff_regressed), np.sum(y_vals_regressed_to_fit)])
x_vals_to_plot = np.linspace(x_centers_to_fit[0], x_centers_to_fit[-1], 100)
y_vals_to_plot = gaussian(x_vals_to_plot, *popt)
y_vals_regressed_to_plot = gaussian(x_vals_to_plot, *popt_regressed)
# Get the mean and std of the differences
mean_diff = popt[0]
std_diff = popt[1]
mean_diff_regressed = popt_regressed[0]
std_diff_regressed = popt_regressed[1]
fig, ax = plt.subplots()
ax.hist(pt_diff, bins=100, range=(-150, 150), histtype='step', label='Original', color='red')
ax.hist(pt_diff_regressed, bins=100, range=(-150, 150), histtype='step', label='Regressed', color='blue')
ax.plot(x_vals_to_plot, y_vals_to_plot, color='red', linestyle='--')
ax.plot(x_vals_to_plot, y_vals_regressed_to_plot, color='blue', linestyle='--')
ax.text(0.60, 0.75, f'Mean: {mean_diff:.2f} $\pm$ {std_diff:.2f}', transform=ax.transAxes, fontsize=16)
ax.text(0.60, 0.70, f'Mean (Reg): {mean_diff_regressed:.2f} $\pm$ {std_diff_regressed:.2f}', transform=ax.transAxes, fontsize=16)
ax.axvline(0, color='black', linestyle='--')
ax.set_xlabel(r"$p_{T}^{L1} \, -  \, p_{T}^{Reco}$ [GeV]")
ax.set_ylabel("Number of jets")
ax.legend()
#ax.set_yscale('log')
plt.savefig(f"plots/training/pt_diff.png")

# pT ratio
#pt_ratio = l1jet_pt / recojet_pt
#pt_ratio_regressed = l1jet_regressed_pt / recojet_pt
pt_ratio = recojet_pt / l1jet_pt    
pt_ratio_regressed = recojet_pt / l1jet_regressed_pt
# Fit a Gaussian near peak
y_vals_ratio, x_edges_ratio = np.histogram(pt_ratio, bins=100, range=(0, 3))
y_vals_regressed_ratio, x_edges_regressed_ratio = np.histogram(pt_ratio_regressed, bins=100, range=(0, 3))
x_centers_ratio = 0.5 * (x_edges_ratio[1:] + x_edges_ratio[:-1])
y_vals_ratio_to_fit = y_vals_ratio[min(0, np.argmax(y_vals_ratio) - 10):max(len(y_vals_ratio), np.argmax(y_vals_ratio) + 10)]
x_centers_ratio_to_fit = x_centers_ratio[min(0, np.argmax(y_vals_ratio) - 10):max(len(y_vals_ratio), np.argmax(y_vals_ratio) + 10)]
y_vals_regressed_ratio_to_fit = y_vals_regressed_ratio[min(0, np.argmax(y_vals_regressed_ratio) - 10):max(len(y_vals_regressed_ratio), np.argmax(y_vals_regressed_ratio) + 10)]
x_centers_regressed_ratio_to_fit = x_centers_ratio[min(0, np.argmax(y_vals_regressed_ratio) - 10):max(len(y_vals_regressed_ratio), np.argmax(y_vals_regressed_ratio) + 10)]
popt_ratio, _ = curve_fit(gaussian, x_centers_ratio_to_fit, y_vals_ratio_to_fit, p0=[1, 0.1, np.sum(y_vals_ratio_to_fit)])
popt_regressed_ratio, _ = curve_fit(gaussian, x_centers_regressed_ratio_to_fit, y_vals_regressed_ratio_to_fit, p0=[1, 0.1, np.sum(y_vals_regressed_ratio_to_fit)])
x_vals_ratio_to_plot = np.linspace(x_centers_ratio_to_fit[0], x_centers_ratio_to_fit[-1], 100)
y_vals_ratio_to_plot = gaussian(x_vals_ratio_to_plot, *popt_ratio)
y_vals_regressed_ratio_to_plot = gaussian(x_vals_ratio_to_plot, *popt_regressed_ratio)
# Get the mean and std of the ratios
mean_ratio = popt_ratio[0]
std_ratio = popt_ratio[1]
mean_ratio_regressed = popt_regressed_ratio[0]
std_ratio_regressed = popt_regressed_ratio[1]
fig, ax = plt.subplots()
ax.hist(pt_ratio, bins=100, range=(0, 3), histtype='step', label='Original', color='red')
ax.hist(pt_ratio_regressed, bins=100, range=(0, 3), histtype='step', label='Regressed', color='blue')
ax.plot(x_vals_ratio_to_plot, y_vals_ratio_to_plot, color='red', linestyle='--')
ax.plot(x_vals_ratio_to_plot, y_vals_regressed_ratio_to_plot, color='blue', linestyle='--')
ax.text(0.60, 0.75, f'Mean: {mean_ratio:.2f} $\pm$ {std_ratio:.2f}', transform=ax.transAxes, fontsize=16)
ax.text(0.60, 0.70, f'Mean (Reg): {mean_ratio_regressed:.2f} $\pm$ {std_ratio_regressed:.2f}', transform=ax.transAxes, fontsize=16)
ax.axvline(1, color='black', linestyle='--')
ax.set_xlabel(r"$p_{T}^{Reco}/p_{T}^{L1}$")
ax.set_ylabel("Number of jets")
ax.legend()
#ax.set_yscale('log')
plt.savefig(f"plots/training/pt_ratio.png")

