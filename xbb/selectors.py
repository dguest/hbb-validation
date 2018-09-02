import numpy as np

def mass_window_higgs(ds):
    fat_mass = ds['mass']
    return (fat_mass > 76e3) & (fat_mass < 146e3)

def mass_window_top(ds):
    return ds['mass'] > 60e3

def window_pt_range(pt_range, mass_window):
    def selector(ds):
        window = mass_window(ds)
        fat_pt = ds['pt']
        pt_window = (fat_pt > pt_range[0]) & (fat_pt < pt_range[1])
        return window & pt_window
    return selector

def truth_match(truth_label):
    def selector(ds):
        return ds[truth_label] == 1
    return selector

def all_events(ds):
    return np.ones_like(ds, dtype=bool)

def window_pt_range_truth_match(pt_range, mass_window, truth_label):
    def selector(ds):
        window = mass_window(ds)
        truth = ds[truth_label] == 1
        fat_pt = ds['pt']
        pt_window = (fat_pt > pt_range[0]) & (fat_pt < pt_range[1])
        return window & pt_window & truth
    return selector

EVENT_LABELS = {
    'higgs': 'GhostHBosonsCount',
    'top': 'GhostTQuarksFinalCount'
}
