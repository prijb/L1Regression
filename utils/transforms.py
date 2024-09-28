#This module is used to perform transformations on data for the neural network 
import awkward as ak
import numpy as np
import pandas as pd

def transform_jet(events):
    pt_cut = events["Jet_pt"] > 30
    events = events[pt_cut]

    #Remove events with no jets passing cuts
    filter = ak.num(events["Jet_pt"]) > 0
    events = events[filter] 

    return events

def transform_muon(events):
    pt_cut = events["Muon_pt"] > 3
    dphi_cut = np.abs(events["Muon_genPhidiff"]) < 3.14
    events = events[np.logical_and(pt_cut, dphi_cut)]

    #Remove events with no muons passing cuts
    filter = ak.num(events["Muon_pt"]) > 0
    events = events[filter]

    return events

def transform_egamma(events):
    pt_cut = events["EGamma_pt"] > 30
    events = events[pt_cut]

    #Remove events with no egammas passing cuts
    filter = ak.num(events["EGamma_pt"]) > 0
    events = events[filter]

    return events



