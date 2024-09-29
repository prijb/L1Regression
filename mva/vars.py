#This module defines the input variables

JET_VARS = ["Jet_pt", "Jet_eta", "Jet_phi", "Jet_relMuonIso", "Jet_relEGammaIso"]
MUON_VARS = ["Muon_pt", "Muon_etaAtVtx", "Muon_phiAtVtx", "Muon_qual", "Muon_hwCharge"]
#MUON_VARS = ["Muon_pt", "Muon_etaAtVtx", "Muon_phiAtVtx", "Muon_qual", "Muon_hwCharge", "Muon_relJetIso", "Muon_relMuonIso", "Muon_relEGammaIso"]
EGAMMA_VARS = ["EGamma_pt", "EGamma_eta", "EGamma_phi", "EGamma_relJetIso", "EGamma_relMuonIso", "EGamma_relEGammaIso", "EGamma_Iso"]
AUX_VARS = ["nJet", "nMuon", "nEGamma", "etSum", "etMiss", "etMissPhi", "htSum", "htMiss", "htMissPhi", "towerCount"]

#Target variables
JET_TARGET = ["Jet_genPt", "Jet_genEta", "Jet_genPhi"]
JET_TARGET_REL = ["Jet_relgenPt", "Jet_genEtadiff", "Jet_genPhidiff"]
MUON_TARGET = ["Muon_genPt", "Muon_genEta", "Muon_genPhi"]
MUON_TARGET_REL = ["Muon_relgenPt", "Muon_genEtadiff", "Muon_genPhidiff"]
EGAMMA_TARGET = ["EGamma_genPt", "EGamma_genEta", "EGamma_genPhi"]
EGAMMA_TARGET_REL = ["EGamma_relgenPt", "EGamma_genEtadiff", "EGamma_genPhidiff"]

#Variables to read
JET_READ_VARS = JET_VARS + JET_TARGET + JET_TARGET_REL
MUON_READ_VARS = MUON_VARS + MUON_TARGET + MUON_TARGET_REL
EGAMMA_READ_VARS = EGAMMA_VARS + EGAMMA_TARGET + EGAMMA_TARGET_REL
READ_VARS = JET_READ_VARS + MUON_READ_VARS + EGAMMA_READ_VARS + AUX_VARS