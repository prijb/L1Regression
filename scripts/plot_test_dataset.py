# Plots the test dataset
import os
import sys
import numpy as np

# Data loaders
import torch
import torch.utils

# Stats
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

from scipy.optimize import curve_fit
# Gaussian function to fit any distributions
def gaussian(x, mu, sigma, A=1.0):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Additional imports
# Add the project path
cwd = os.getcwd()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Make plotdir if it doesn't exist
os.makedirs(f"{cwd}/plots/testing", exist_ok=True)
os.makedirs(f"{cwd}/plots/features", exist_ok=True)

sample = "qcd_15to7000"
#sample = "qcd_15to7000"

#test_cachedir = f"{cwd}/cachedir/{sample}/train"
test_cachedir = f"{cwd}/cachedir/{sample}/test"

from utils.preprocess import Preprocessor
from utils.bdt_dataset import BDTDataset

input_files = ["dummy_file.root"]
preprocessor_test = Preprocessor(input_files, batch_size=100000, use_existing_cache=True, cache_dir=test_cachedir)

X_test, y_test, w_test = preprocessor_test.get_data_dict()

############# BDT evaluation #################
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
test_dataset = BDTDataset(X_test, y_test, w_test, train_features, target_features)
X_test, y_test, w_test = test_dataset[:]

# Load the BDT
from mva.bdt_regressor import BDTRegressor
model = BDTRegressor()
model.load_model(f"models/bdt_model_qcd_15to7000.json")

############# Evaluation #################
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

# Eta and phi
l1jet_eta = X_test[:, 1].detach().numpy()
l1jet_phi = X_test[:, 2].detach().numpy()


#################### Inclusive pT distributions #####################
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
plt.savefig(f"plots/testing/jet_pt_distribution.png")
#plt.close()

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
plt.savefig(f"plots/testing/pt_diff.png")
#plt.close()

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
plt.savefig(f"plots/testing/pt_ratio.png")
#plt.close()

#################### Eta binned pT distributions #####################
os.makedirs(f"{cwd}/plots/testing/eta_binned", exist_ok=True)
eta_bins = np.array([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])

for i_bin in range(len(eta_bins) - 1):
    print(f"Plotting eta bin {i_bin} [{eta_bins[i_bin]} - {eta_bins[i_bin + 1]}]")
    mask = np.logical_and(l1jet_eta >= eta_bins[i_bin], l1jet_eta < eta_bins[i_bin + 1])
    l1jet_pt_bin = l1jet_pt[mask]
    l1jet_regressed_pt_bin = l1jet_regressed_pt[mask]
    recojet_pt_bin = recojet_pt[mask]

    # pT distribution
    y_vals, x_edges = np.histogram(l1jet_pt_bin, bins=100, range=(0, 1000))
    y_vals_regressed, x_edges_regressed = np.histogram(l1jet_regressed_pt_bin, bins=100, range=(0, 1000))
    y_vals_reco, x_edges_reco = np.histogram(recojet_pt_bin, bins=100, range=(0, 1000))
    x_centers = 0.5 * (x_edges[1:] + x_edges[:-1])
    fig, axs = plt.subplots(2, 1, gridspec_kw=dict(height_ratios=[3, 1], hspace=0.1), sharex=True)
    axs[0].hist(l1jet_pt_bin, bins=100, range=(0, 1000), histtype='step', label='L1 Jet pT', color='red')
    axs[0].hist(l1jet_regressed_pt_bin, bins=100, range=(0, 1000), histtype='step', label='Regressed L1 Jet pT', color='blue')
    axs[0].hist(recojet_pt_bin, bins=100, range=(0, 1000), histtype='step', label='Reco Jet pT', color='black')
    axs[0].set_xlabel("")
    axs[0].set_ylabel("Number of jets")
    axs[0].legend()
    axs[0].text(0.60, 0.70, f"$\eta$ [{eta_bins[i_bin]:.1f} , {eta_bins[i_bin + 1]:.1f}]", transform=axs[0].transAxes, fontsize=16)
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
    plt.savefig(f"plots/testing/eta_binned/jet_pt_distribution_eta_{i_bin}.png")
    #plt.close()

    # pT diff
    pt_diff_bin = l1jet_pt_bin - recojet_pt_bin
    pt_diff_regressed_bin = l1jet_regressed_pt_bin - recojet_pt_bin
    # Fit a Gaussian near peak
    y_vals, x_edges = np.histogram(pt_diff_bin, bins=100, range=(-150, 150))
    y_vals_regressed, x_edges_regressed = np.histogram(pt_diff_regressed_bin, bins=100, range=(-150, 150))
    x_centers = 0.5 * (x_edges[1:] + x_edges[:-1])
    y_vals_to_fit = y_vals[min(0, np.argmax(y_vals) - 10):max(len(y_vals), np.argmax(y_vals) + 10)]
    x_centers_to_fit = x_centers[min(0, np.argmax(y_vals) - 10):max(len(y_vals), np.argmax(y_vals) + 10)]
    y_vals_regressed_to_fit = y_vals_regressed[min(0, np.argmax(y_vals_regressed) - 10):max(len(y_vals_regressed), np.argmax(y_vals_regressed) + 10)]
    x_centers_regressed_to_fit = x_centers[min(0, np.argmax(y_vals_regressed) - 10):max(len(y_vals_regressed), np.argmax(y_vals_regressed) + 10)]
    popt, _ = curve_fit(gaussian, x_centers_to_fit, y_vals_to_fit, p0=[0, np.std(pt_diff_bin), np.sum(y_vals_to_fit)])
    popt_regressed, _ = curve_fit(gaussian, x_centers_regressed_to_fit, y_vals_regressed_to_fit, p0=[0, np.std(pt_diff_regressed_bin), np.sum(y_vals_regressed_to_fit)])
    x_vals_to_plot = np.linspace(x_centers_to_fit[0], x_centers_to_fit[-1], 100)
    y_vals_to_plot = gaussian(x_vals_to_plot, *popt)
    y_vals_regressed_to_plot = gaussian(x_vals_to_plot, *popt_regressed)
    # Get the mean and std of the differences
    mean_diff = popt[0]
    std_diff = popt[1]
    mean_diff_regressed = popt_regressed[0]
    std_diff_regressed = popt_regressed[1]
    fig, ax = plt.subplots()
    ax.hist(pt_diff_bin, bins=100, range=(-150, 150), histtype='step', label='Original', color='red')
    ax.hist(pt_diff_regressed_bin, bins=100, range=(-150, 150), histtype='step', label='Regressed', color='blue')
    ax.plot(x_vals_to_plot, y_vals_to_plot, color='red', linestyle='--')
    ax.plot(x_vals_to_plot, y_vals_regressed_to_plot, color='blue', linestyle='--')
    ax.text(0.60, 0.75, f'Mean: {mean_diff:.2f} $\pm$ {std_diff:.2f}', transform=ax.transAxes, fontsize=16)
    ax.text(0.60, 0.70, f'Mean (Reg): {mean_diff_regressed:.2f} $\pm$ {std_diff_regressed:.2f}', transform=ax.transAxes, fontsize=16)
    ax.text(0.60, 0.65, f"$\eta$ [{eta_bins[i_bin]:.1f} , {eta_bins[i_bin + 1]:.1f}]", transform=ax.transAxes, fontsize=16)
    ax.axvline(0, color='black', linestyle='--')
    ax.set_xlabel(r"$p_{T}^{L1} \, -  \, p_{T}^{Reco}$ [GeV]")
    ax.set_ylabel("Number of jets")
    ax.legend()
    #ax.set_yscale('log')Eta binned pT distributions
    plt.savefig(f"plots/testing/eta_binned/pt_diff_eta_{i_bin}.png")
    #plt.close()

    # pT ratio
    pt_ratio_bin = recojet_pt_bin / l1jet_pt_bin
    pt_ratio_regressed_bin = recojet_pt_bin / l1jet_regressed_pt_bin
    # Fit a Gaussian near peak
    y_vals_ratio, x_edges_ratio = np.histogram(pt_ratio_bin, bins=100, range=(0, 3))
    y_vals_regressed_ratio, x_edges_regressed_ratio = np.histogram(pt_ratio_regressed_bin, bins=100, range=(0, 3))
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
    ax.hist(pt_ratio_bin, bins=100, range=(0, 3), histtype='step', label='Original', color='red')
    ax.hist(pt_ratio_regressed_bin, bins=100, range=(0, 3), histtype='step', label='Regressed', color='blue')
    ax.plot(x_vals_ratio_to_plot, y_vals_ratio_to_plot, color='red', linestyle='--')
    ax.plot(x_vals_ratio_to_plot, y_vals_regressed_ratio_to_plot, color='blue', linestyle='--')
    ax.text(0.60, 0.75, f'Mean: {mean_ratio:.2f} $\pm$ {std_ratio:.2f}', transform=ax.transAxes, fontsize=16)
    ax.text(0.60, 0.70, f'Mean (Reg): {mean_ratio_regressed:.2f} $\pm$ {std_ratio_regressed:.2f}', transform=ax.transAxes, fontsize=16)
    ax.text(0.60, 0.65, f"$\eta$ [{eta_bins[i_bin]:.1f} , {eta_bins[i_bin + 1]:.1f}]", transform=ax.transAxes, fontsize=16)
    ax.axvline(1, color='black', linestyle='--')
    ax.set_xlabel(r"$p_{T}^{Reco}/p_{T}^{L1}$")
    ax.set_ylabel("Number of jets")
    ax.legend()
    #ax.set_yscale('log')
    plt.savefig(f"plots/testing/eta_binned/pt_ratio_eta_{i_bin}.png")
    #plt.close()

#################### pT binned pT scale distributions #####################
os.makedirs(f"{cwd}/plots/testing/pt_binned", exist_ok=True)
pt_bins = np.array([30, 50, 100, 300, 1000])

for i_bin in range(len(pt_bins) - 1):
    print(f"Plotting pT bin {i_bin} [{pt_bins[i_bin]} - {pt_bins[i_bin + 1]}]")
    mask = np.logical_and(l1jet_pt >= pt_bins[i_bin], l1jet_pt < pt_bins[i_bin + 1])
    l1jet_pt_bin = l1jet_pt[mask]
    l1jet_regressed_pt_bin = l1jet_regressed_pt[mask]
    recojet_pt_bin = recojet_pt[mask]

    # pT diff
    pt_diff_bin = l1jet_pt_bin - recojet_pt_bin
    pt_diff_regressed_bin = l1jet_regressed_pt_bin - recojet_pt_bin
    # Fit a Gaussian near peak
    y_vals, x_edges = np.histogram(pt_diff_bin, bins=100, range=(-150, 150))
    y_vals_regressed, x_edges_regressed = np.histogram(pt_diff_regressed_bin, bins=100, range=(-150, 150))
    x_centers = 0.5 * (x_edges[1:] + x_edges[:-1])
    y_vals_to_fit = y_vals[min(0, np.argmax(y_vals) - 10):max(len(y_vals), np.argmax(y_vals) + 10)]
    x_centers_to_fit = x_centers[min(0, np.argmax(y_vals) - 10):max(len(y_vals), np.argmax(y_vals) + 10)]
    y_vals_regressed_to_fit = y_vals_regressed[min(0, np.argmax(y_vals_regressed) - 10):max(len(y_vals_regressed), np.argmax(y_vals_regressed) + 10)]
    x_centers_regressed_to_fit = x_centers[min(0, np.argmax(y_vals_regressed) - 10):max(len(y_vals_regressed), np.argmax(y_vals_regressed) + 10)]
    popt, _ = curve_fit(gaussian, x_centers_to_fit, y_vals_to_fit, p0=[0, np.std(pt_diff_bin), np.sum(y_vals_to_fit)])
    popt_regressed, _ = curve_fit(gaussian, x_centers_regressed_to_fit, y_vals_regressed_to_fit, p0=[0, np.std(pt_diff_regressed_bin), np.sum(y_vals_regressed_to_fit)])
    x_vals_to_plot = np.linspace(x_centers_to_fit[0], x_centers_to_fit[-1], 100)
    y_vals_to_plot = gaussian(x_vals_to_plot, *popt)
    y_vals_regressed_to_plot = gaussian(x_vals_to_plot, *popt_regressed)
    # Get the mean and std of the differences
    mean_diff = popt[0]
    std_diff = popt[1]
    mean_diff_regressed = popt_regressed[0]
    std_diff_regressed = popt_regressed[1]
    fig, ax = plt.subplots()
    ax.hist(pt_diff_bin, bins=100, range=(-150, 150), histtype='step', label='Original', color='red')
    ax.hist(pt_diff_regressed_bin, bins=100, range=(-150, 150), histtype='step', label='Regressed', color='blue')
    ax.plot(x_vals_to_plot, y_vals_to_plot, color='red', linestyle='--')
    ax.plot(x_vals_to_plot, y_vals_regressed_to_plot, color='blue', linestyle='--')
    ax.text(0.60, 0.75, f'Mean: {mean_diff:.2f} $\pm$ {std_diff:.2f}', transform=ax.transAxes, fontsize=16)
    ax.text(0.60, 0.70, f'Mean (Reg): {mean_diff_regressed:.2f} $\pm$ {std_diff_regressed:.2f}', transform=ax.transAxes, fontsize=16)
    ax.text(0.60, 0.65, f"$pT$ [{pt_bins[i_bin]:.1f} , {pt_bins[i_bin + 1]:.1f}]", transform=ax.transAxes, fontsize=16)
    ax.axvline(0, color='black', linestyle='--')
    ax.set_xlabel(r"$p_{T}^{L1} \, -  \, p_{T}^{Reco}$ [GeV]")
    ax.set_ylabel("Number of jets")
    ax.legend()
    #ax.set_yscale('log')Eta binned pT distributions
    plt.savefig(f"plots/testing/pt_binned/pt_diff_pt_{i_bin}.png")
    #plt.close()

    # pT ratio
    pt_ratio_bin = recojet_pt_bin / l1jet_pt_bin
    pt_ratio_regressed_bin = recojet_pt_bin / l1jet_regressed_pt_bin
    # Fit a Gaussian near peak
    y_vals_ratio, x_edges_ratio = np.histogram(pt_ratio_bin, bins=100, range=(0, 3))
    y_vals_regressed_ratio, x_edges_regressed_ratio = np.histogram(pt_ratio_regressed_bin, bins=100, range=(0, 3))
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
    ax.hist(pt_ratio_bin, bins=100, range=(0, 3), histtype='step', label='Original', color='red')
    ax.hist(pt_ratio_regressed_bin, bins=100, range=(0, 3), histtype='step', label='Regressed', color='blue')
    ax.plot(x_vals_ratio_to_plot, y_vals_ratio_to_plot, color='red', linestyle='--')
    ax.plot(x_vals_ratio_to_plot, y_vals_regressed_ratio_to_plot, color='blue', linestyle='--')
    ax.text(0.60, 0.75, f'Mean: {mean_ratio:.2f} $\pm$ {std_ratio:.2f}', transform=ax.transAxes, fontsize=16)
    ax.text(0.60, 0.70, f'Mean (Reg): {mean_ratio_regressed:.2f} $\pm$ {std_ratio_regressed:.2f}', transform=ax.transAxes, fontsize=16)
    ax.text(0.60, 0.65, f"$pT$ [{pt_bins[i_bin]:.1f} , {pt_bins[i_bin + 1]:.1f}]", transform=ax.transAxes, fontsize=16)
    ax.axvline(1, color='black', linestyle='--')
    ax.set_xlabel(r"$p_{T}^{Reco}/p_{T}^{L1}$")
    ax.set_ylabel("Number of jets")
    ax.legend()
    #ax.set_yscale('log')
    plt.savefig(f"plots/testing/pt_binned/pt_ratio_pt_{i_bin}.png")
    #plt.close()

# pT-eta binned pT scale distributions
os.makedirs(f"{cwd}/plots/testing/pt_eta_binned", exist_ok=True)

for i_pt_bin in range(len(pt_bins) - 1):
    for i_eta_bin in range(len(eta_bins) - 1):
        print(f"Plotting pT-eta bin {i_pt_bin} [{pt_bins[i_pt_bin]} - {pt_bins[i_pt_bin + 1]}] and eta bin {i_eta_bin} [{eta_bins[i_eta_bin]} - {eta_bins[i_eta_bin + 1]}]")
        mask_pt = np.logical_and(l1jet_pt >= pt_bins[i_pt_bin], l1jet_pt < pt_bins[i_pt_bin + 1])
        mask_eta = np.logical_and(l1jet_eta >= eta_bins[i_eta_bin], l1jet_eta < eta_bins[i_eta_bin + 1])
        mask = np.logical_and(mask_pt, mask_eta)

        l1jet_pt_bin = l1jet_pt[mask]
        l1jet_regressed_pt_bin = l1jet_regressed_pt[mask]
        recojet_pt_bin = recojet_pt[mask]

        # pT distribution
        y_vals, x_edges = np.histogram(l1jet_pt_bin, bins=50, range=(pt_bins[i_pt_bin], pt_bins[i_pt_bin + 1]))
        y_vals_regressed, x_edges_regressed = np.histogram(l1jet_regressed_pt_bin, bins=50, range=(pt_bins[i_pt_bin], pt_bins[i_pt_bin + 1]))
        y_vals_reco, x_edges_reco = np.histogram(recojet_pt_bin, bins=50, range=(pt_bins[i_pt_bin], pt_bins[i_pt_bin + 1]))
        x_centers = 0.5 * (x_edges[1:] + x_edges[:-1])
        fig, axs = plt.subplots(2, 1, gridspec_kw=dict(height_ratios=[3, 1], hspace=0.1), sharex=True)
        axs[0].hist(l1jet_pt_bin, bins=50, range=(pt_bins[i_pt_bin], pt_bins[i_pt_bin + 1]), histtype='step', label='L1 Jet pT', color='red')
        axs[0].hist(l1jet_regressed_pt_bin, bins=50, range=(pt_bins[i_pt_bin], pt_bins[i_pt_bin + 1]), histtype='step', label='Regressed L1 Jet pT', color='blue')
        axs[0].hist(recojet_pt_bin, bins=50, range=(pt_bins[i_pt_bin], pt_bins[i_pt_bin + 1]), histtype='step', label='Reco Jet pT', color='black')
        axs[0].set_xlabel("")
        axs[0].set_ylabel("Number of jets")
        axs[0].legend()
        axs[0].text(0.60, 0.70, f"$\eta$ [{eta_bins[i_eta_bin]:.1f} , {eta_bins[i_eta_bin + 1]:.1f}]", transform=axs[0].transAxes, fontsize=16)
        axs[0].text(0.60, 0.65, f"$pT$ [{pt_bins[i_pt_bin]:.1f} , {pt_bins[i_pt_bin + 1]:.1f}]", transform=axs[0].transAxes, fontsize=16)
        axs[0].set_xlim(pt_bins[i_pt_bin], pt_bins[i_pt_bin + 1])
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
        plt.savefig(f"plots/testing/pt_eta_binned/jet_pt_distribution_pt_{i_pt_bin}_eta_{i_eta_bin}.png")
        #plt.close()

        # pT diff
        pt_diff_bin = l1jet_pt_bin - recojet_pt_bin
        pt_diff_regressed_bin = l1jet_regressed_pt_bin - recojet_pt_bin
        # Fit a Gaussian near peak
        y_vals, x_edges = np.histogram(pt_diff_bin, bins=100, range=(-150, 150))
        y_vals_regressed, x_edges_regressed = np.histogram(pt_diff_regressed_bin, bins=100, range=(-150, 150))
        x_centers = 0.5 * (x_edges[1:] + x_edges[:-1])
        y_vals_to_fit = y_vals[min(0, np.argmax(y_vals) - 10):max(len(y_vals), np.argmax(y_vals) + 10)]
        x_centers_to_fit = x_centers[min(0, np.argmax(y_vals) - 10):max(len(y_vals), np.argmax(y_vals) + 10)]
        y_vals_regressed_to_fit = y_vals_regressed[min(0, np.argmax(y_vals_regressed) - 10):max(len(y_vals_regressed), np.argmax(y_vals_regressed) + 10)]
        x_centers_regressed_to_fit = x_centers[min(0, np.argmax(y_vals_regressed) - 10):max(len(y_vals_regressed), np.argmax(y_vals_regressed) + 10)]
        popt, _ = curve_fit(gaussian, x_centers_to_fit, y_vals_to_fit, p0=[0, np.std(pt_diff_bin), np.sum(y_vals_to_fit)])
        popt_regressed, _ = curve_fit(gaussian, x_centers_regressed_to_fit, y_vals_regressed_to_fit, p0=[0, np.std(pt_diff_regressed_bin), np.sum(y_vals_regressed_to_fit)])
        x_vals_to_plot = np.linspace(x_centers_to_fit[0], x_centers_to_fit[-1], 100)
        y_vals_to_plot = gaussian(x_vals_to_plot, *popt)
        y_vals_regressed_to_plot = gaussian(x_vals_to_plot, *popt_regressed)
        # Get the mean and std of the differences
        mean_diff = popt[0]
        std_diff = popt[1]
        mean_diff_regressed = popt_regressed[0]
        std_diff_regressed = popt_regressed[1]
        fig, ax = plt.subplots()
        ax.hist(pt_diff_bin, bins=100, range=(-150, 150), histtype='step', label='Original', color='red')
        ax.hist(pt_diff_regressed_bin, bins=100, range=(-150, 150), histtype='step', label='Regressed', color='blue')
        ax.plot(x_vals_to_plot, y_vals_to_plot, color='red', linestyle='--')
        ax.plot(x_vals_to_plot, y_vals_regressed_to_plot, color='blue', linestyle='--')
        ax.text(0.60, 0.75, f'Mean: {mean_diff:.2f} $\pm$ {std_diff:.2f}', transform=ax.transAxes, fontsize=16)
        ax.text(0.60, 0.70, f'Mean (Reg): {mean_diff_regressed:.2f} $\pm$ {std_diff_regressed:.2f}', transform=ax.transAxes, fontsize=16)
        ax.text(0.60, 0.65, f"$\eta$ [{eta_bins[i_eta_bin]:.1f} , {eta_bins[i_eta_bin + 1]:.1f}]", transform=ax.transAxes, fontsize=16)
        ax.text(0.60, 0.60, f"$pT$ [{pt_bins[i_pt_bin]:.1f} , {pt_bins[i_pt_bin + 1]:.1f}]", transform=ax.transAxes, fontsize=16)
        ax.axvline(0, color='black', linestyle='--')
        ax.set_xlabel(r"$p_{T}^{L1} \, -  \, p_{T}^{Reco}$ [GeV]")
        ax.set_ylabel("Number of jets")
        ax.legend()
        #ax.set_yscale('log')
        plt.savefig(f"plots/testing/pt_eta_binned/pt_diff_pt_{i_pt_bin}_eta_{i_eta_bin}.png")
        #plt.close()
        
        # pT ratio
        pt_ratio_bin = recojet_pt_bin / l1jet_pt_bin
        pt_ratio_regressed_bin = recojet_pt_bin / l1jet_regressed_pt_bin
        # Fit a Gaussian near peak
        y_vals_ratio, x_edges_ratio = np.histogram(pt_ratio_bin, bins=100, range=(0, 3))
        y_vals_regressed_ratio, x_edges_regressed_ratio = np.histogram(pt_ratio_regressed_bin, bins=100, range=(0, 3))
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
        ax.hist(pt_ratio_bin, bins=100, range=(0, 3), histtype='step', label='Original', color='red')
        ax.hist(pt_ratio_regressed_bin, bins=100, range=(0, 3), histtype='step', label='Regressed', color='blue')
        ax.plot(x_vals_ratio_to_plot, y_vals_ratio_to_plot, color='red', linestyle='--')
        ax.plot(x_vals_ratio_to_plot, y_vals_regressed_ratio_to_plot, color='blue', linestyle='--')
        ax.text(0.60, 0.75, f'Mean: {mean_ratio:.2f} $\pm$ {std_ratio:.2f}', transform=ax.transAxes, fontsize=16)
        ax.text(0.60, 0.70, f'Mean (Reg): {mean_ratio_regressed:.2f} $\pm$ {std_ratio_regressed:.2f}', transform=ax.transAxes, fontsize=16)
        ax.text(0.60, 0.65, f"$\eta$ [{eta_bins[i_eta_bin]:.1f} , {eta_bins[i_eta_bin + 1]:.1f}]", transform=ax.transAxes, fontsize=16)
        ax.text(0.60, 0.60, f"$pT$ [{pt_bins[i_pt_bin]:.1f} , {pt_bins[i_pt_bin + 1]:.1f}]", transform=ax.transAxes, fontsize=16)
        ax.axvline(1, color='black', linestyle='--')
        ax.set_xlabel(r"$p_{T}^{Reco}/p_{T}^{L1}$")
        ax.set_ylabel("Number of jets")
        ax.legend()
        plt.savefig(f"plots/testing/pt_eta_binned/pt_ratio_pt_{i_pt_bin}_eta_{i_eta_bin}.png")
        #plt.close()

        
    

# Skip the below code if the output is one dimensional
#################### Inclusive eta distributions #####################
if y_pred.ndim == 1:
    print("Skipping eta distributions for one-dimensional output.")

else:
    print("Plotting eta distributions for multi-dimensional output.")



