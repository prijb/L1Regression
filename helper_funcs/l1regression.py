# Helper functions for L1 object regression
import uproot   
import awkward as ak
import numpy as np
import pandas as pd

def phi_phasewrap(phi):
    """
    Used for example when phi is deltaphi = phi1 - phi2
    """
    return (phi + np.pi) % (2 * np.pi) - np.pi

def muonjet_selection(events):

    #print(f"Number of events before event filters: {len(events)}")

    # Event filters
    # Trigger
    trigger_filter = events["HLT_IsoMu24"]
    # Multiplicity
    multiplicty_filter = (events["nL1Jet"] > 0) & (events["nJet"] > 0)
    events = events[trigger_filter & multiplicty_filter]

    #print(f"Number of events passing event filters: {len(events)}")

    # Object filters
    # Muon selection
    muons = events["Muon"]
    passtightid = (muons["pfIsoId"] > 3) & (muons["mediumPromptId"])
    goodmuon_pt25 = (muons["pt"] > 25) & (np.abs(muons["pdgId"]) == 13) & (passtightid)
    badmuon_pt10 = (muons["pt"] > 10) & (np.abs(muons["pdgId"]) == 13) & (~passtightid)
    goodmuon_sel = ak.sum(goodmuon_pt25, axis=1) > 0
    badmuon_veto = ak.sum(badmuon_pt10, axis=1) == 0
    events = events[goodmuon_sel & badmuon_veto]

    #print(f"Number of events passing muon selections: {len(events)}")

    # Jet cleaning (optional)

    return events

# Match to offline candidates and calculate isolation
def match_l1_to_offline(events, obj="Jet"):
    l1key = f"L1{obj}"
    offlinekey = f"{obj}"
    if obj == "Mu":
        offlinekey = "Muon"

    l1dict = {}
    for var in events[l1key].fields:
        l1dict[var] = events[l1key][var]
    offlinedict = {}
    for var in events[offlinekey].fields:
        offlinedict[var] = events[offlinekey][var]

    l1objs = ak.zip(l1dict)
    recoobjs = ak.zip(offlinedict)

    # Match
    l1_reco_pair = ak.cartesian({"l1": l1objs, "reco": recoobjs}, nested=True)
    l1_reco_pair_args = ak.argcartesian({"l1": l1objs, "reco": recoobjs}, nested=True)
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
    l1objs_i, recoobjs_i = ak.unzip(l1_reco_pair)
    l1objs_arg_i, recoobjs_arg_i = ak.unzip(l1_reco_pair_args)
    match_per_l1objs = ak.num(l1_reco_pair, axis=-1)
    l1objs_arg_i = l1objs_arg_i[match_per_l1objs == 1]
    l1objs_arg_i = ak.firsts(l1objs_arg_i, axis=-1)
    l1objs_arg_i = l1objs_arg_i[~ak.is_none(l1objs_arg_i, axis=-1)]
    recoobjs_arg_i = recoobjs_arg_i[match_per_l1objs == 1]
    recoobjs_arg_i = ak.firsts(recoobjs_arg_i, axis=-1)
    recoobjs_arg_i = recoobjs_arg_i[~ak.is_none(recoobjs_arg_i, axis=-1)]  
    l1objs_matched = l1objs[l1objs_arg_i]
    recoobjs_matched = recoobjs[recoobjs_arg_i]

    # Create target variables
    recopt = recoobjs_matched.pt
    ptratio = recoobjs_matched.pt/l1objs_matched.pt
    etadiff = recoobjs_matched.eta - l1objs_matched.eta
    phidiff = phi_phasewrap(recoobjs_matched.phi - l1objs_matched.phi)
    l1objs_matched = ak.with_field(l1objs_matched, recopt, "recopt")
    l1objs_matched = ak.with_field(l1objs_matched, ptratio, "ptratio")
    l1objs_matched = ak.with_field(l1objs_matched, etadiff, "etadiff")
    l1objs_matched = ak.with_field(l1objs_matched, phidiff, "phidiff")

    # Create isolation
    l1muons = ak.zip({
            "pt": events["L1Mu"]["pt"],
            "eta": events["L1Mu"]["etaAtVtx"],
            "phi": events["L1Mu"]["phiAtVtx"],
    })
    l1jets = ak.zip({
        "pt": events["L1Jet"]["pt"],
        "eta": events["L1Jet"]["eta"],
        "phi": events["L1Jet"]["phi"],
    })
    l1egs = ak.zip({
        "pt": events["L1EG"]["pt"],
        "eta": events["L1EG"]["eta"],
        "phi": events["L1EG"]["phi"],
    })
    
    # Match to muons
    if obj != "Mu":
        l1_muon_pair = ak.cartesian({"l1": l1objs_matched, "muon": l1muons}, nested=True)
        l1_muon_pair_args = ak.argcartesian({"l1": l1objs_matched, "muon": l1muons}, nested=True)
        l1_muon_dR = np.sqrt(
            (l1_muon_pair["l1"]["eta"] - l1_muon_pair["muon"]["eta"])**2 +
            (phi_phasewrap(l1_muon_pair["l1"]["phi"] - l1_muon_pair["muon"]["phi"]))**2
        )
        l1_muon_dR_order = ak.argsort(l1_muon_dR, ascending=True, axis=2)
        l1_muon_pair = l1_muon_pair[l1_muon_dR_order]
        l1_muon_pair_args = l1_muon_pair_args[l1_muon_dR_order]
        l1_muon_dR = l1_muon_dR[l1_muon_dR_order]
        # Match within 0.3
        l1_muon_match_cut = l1_muon_dR < 0.3
        l1_muon_pair = l1_muon_pair[l1_muon_match_cut]
        l1_muon_pair_args = l1_muon_pair_args[l1_muon_match_cut]
        l1_muon_dR = l1_muon_dR[l1_muon_match_cut]
        # Sum up muon pt for all matched muons
        l1objs_i, muon_i = ak.unzip(l1_muon_pair)
        l1objs_muon_iso = ak.sum(muon_i["pt"], axis=-1)
        l1objs_matched = ak.with_field(l1objs_matched, l1objs_muon_iso, "muiso")
        l1objs_matched = ak.with_field(l1objs_matched, l1objs_muon_iso/l1objs_matched.pt, "mureliso")

    # Match to jets
    if obj != "Jet":
        l1_jet_pair = ak.cartesian({"l1": l1objs_matched, "jet": l1jets}, nested=True)
        l1_jet_pair_args = ak.argcartesian({"l1": l1objs_matched, "jet": l1jets}, nested=True)
        l1_jet_dR = np.sqrt(
            (l1_jet_pair["l1"]["eta"] - l1_jet_pair["jet"]["eta"])**2 +
            (phi_phasewrap(l1_jet_pair["l1"]["phi"] - l1_jet_pair["jet"]["phi"]))**2
        )
        l1_jet_dR_order = ak.argsort(l1_jet_dR, ascending=True, axis=2)
        l1_jet_pair = l1_jet_pair[l1_jet_dR_order]
        l1_jet_pair_args = l1_jet_pair_args[l1_jet_dR_order]
        l1_jet_dR = l1_jet_dR[l1_jet_dR_order]
        # Match within 0.3
        l1_jet_match_cut = l1_jet_dR < 0.3
        l1_jet_pair = l1_jet_pair[l1_jet_match_cut]
        l1_jet_pair_args = l1_jet_pair_args[l1_jet_match_cut]
        l1_jet_dR = l1_jet_dR[l1_jet_match_cut]
        # Sum up jet pt
        l1objs_i, jet_i = ak.unzip(l1_jet_pair)
        l1objs_jet_iso = ak.sum(jet_i["pt"], axis=-1)
        l1objs_matched = ak.with_field(l1objs_matched, l1objs_jet_iso, "jetiso")
        l1objs_matched = ak.with_field(l1objs_matched, l1objs_jet_iso/l1objs_matched.pt, "jetreliso")
        
    # Match to egammas
    if obj != "EG":
        l1_eg_pair = ak.cartesian({"l1": l1objs_matched, "eg": l1egs}, nested=True)
        l1_eg_pair_args = ak.argcartesian({"l1": l1objs_matched, "eg": l1egs}, nested=True)
        l1_eg_dR = np.sqrt(
            (l1_eg_pair["l1"]["eta"] - l1_eg_pair["eg"]["eta"])**2 +
            (phi_phasewrap(l1_eg_pair["l1"]["phi"] - l1_eg_pair["eg"]["phi"]))**2
        )
        l1_eg_dR_order = ak.argsort(l1_eg_dR, ascending=True, axis=2)
        l1_eg_pair = l1_eg_pair[l1_eg_dR_order]
        l1_eg_pair_args = l1_eg_pair_args[l1_eg_dR_order]
        l1_eg_dR = l1_eg_dR[l1_eg_dR_order]
        # Match within 0.3
        l1_eg_match_cut = l1_eg_dR < 0.3
        l1_eg_pair = l1_eg_pair[l1_eg_match_cut]
        l1_eg_pair_args = l1_eg_pair_args[l1_eg_match_cut]
        l1_eg_dR = l1_eg_dR[l1_eg_match_cut]
        # Sum up eg pt
        l1objs_i, eg_i = ak.unzip(l1_eg_pair)
        l1objs_eg_iso = ak.sum(eg_i["pt"], axis=-1)
        l1objs_matched = ak.with_field(l1objs_matched, l1objs_eg_iso, "egiso")
        l1objs_matched = ak.with_field(l1objs_matched, l1objs_eg_iso/l1objs_matched.pt, "egreliso")

    events = ak.with_field(events, l1objs_matched, l1key)

    return events