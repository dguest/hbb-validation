#!/usr/bin/env python3

"""
Make histograms of the jet pt spectra
"""

from argparse import ArgumentParser
from h5py import File
from glob import glob
import numpy as np
import json, os

from xbb.common import get_denom_dict, get_dsid
from xbb.common import is_dijet, is_ditop, is_dihiggs
from xbb.common import SELECTORS
from xbb_draw_roc_curves import draw_roc_curves
from xbb.cross_section import CrossSections

# default settings
PT_RANGE = (250e3, np.inf)

def get_args():
    parser = ArgumentParser(description=__doc__)
    d = 'default: %(default)s'
    parser.add_argument('datasets', nargs='+')
    parser.add_argument('-d', '--denominator', required=True)
    parser.add_argument('-x', '--cross-sections', required=True)
    parser.add_argument('-i', '--input-hist-dir', default='pt-hists')
    parser.add_argument('-p', '--pt-range',
                        nargs=2, type=float, default=PT_RANGE, help=d)
    parser.add_argument('-s', '--save-file')
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser.parse_args()

####################
# variable getters #
####################

# various constants
SJ = 'subjet_VR_{}'
SJ_GHOST = 'subjet_VRGhostTag_{}'

def get_mv2(h5file, discriminant='MV2c10_discriminant'):
    disc1 = h5file[SJ.format(1)][discriminant]
    disc2 = h5file[SJ.format(2)][discriminant]
    discrim_comb = np.stack([disc1, disc2], axis=1).min(axis=1)
    invalid = np.isnan(discrim_comb)
    discrim_comb[invalid] = -1.0
    return discrim_comb

def get_dnn(h5file):
    return np.asarray(h5file['fat_jet']['HbbScore'])

def make_xbb_getter(ftop=0.5):
    def get_xbb(h5file,ftop=ftop):
        fj = h5file['fat_jet']['XbbScoreHiggs', 'XbbScoreQCD', 'XbbScoreTop']
        num = fj['XbbScoreHiggs']
        denom_qcd = fj['XbbScoreQCD'] * (1 - ftop)
        denom_top = fj['XbbScoreTop'] * ftop
        denom = denom_qcd + denom_top
        valid = (denom != 0.0) & (num != 0.0) & (np.isfinite(num))
        ret_vals = np.empty_like(num)
        ret_vals[valid] = np.log(num[valid] / denom[valid])
        ret_vals[~valid] = -10.0
        return ret_vals
    return get_xbb


def get_top_vs_qcd(h5file):
    fj = h5file['fat_jet']['XbbScoreQCD', 'XbbScoreTop']
    num = fj['XbbScoreTop']
    denom = fj['XbbScoreQCD']
    valid = (denom != 0.0) & (num != 0.0) & (np.isfinite(num))
    ret_vals = np.empty_like(num)
    ret_vals[valid] = np.log(num[valid] / denom[valid])
    ret_vals[~valid] = -10.0
    return ret_vals

def get_tau32(h5file):
    vals = 1.0 - h5file['fat_jet']['Tau32_wta']
    vals[~np.isfinite(vals)] = 0.0
    return vals

def make_dl1_getter(subjet=SJ):
    def get_dl1(h5file, subjet=subjet):
        def dl1_sj(subjet, f=0.08):
            sj = h5file[subjet]['DL1_pb','DL1_pu','DL1_pc']
            return sj['DL1_pb'] / ( (1-f) * sj['DL1_pu'] + f * sj['DL1_pc'])

        disc1 = dl1_sj(subjet.format(1))
        disc2 = dl1_sj(subjet.format(2))
        discrim_comb = np.stack([disc1, disc2], axis=1).min(axis=1)
        invalid = np.isnan(discrim_comb) | np.isinf(discrim_comb)
        discrim_comb[invalid] = 1e-15
        return np.log(np.clip(discrim_comb, 1e-30, 1e30))
    return get_dl1

#############
# selectors #
#############
def mass_window_higgs(ds):
    fat_mass = ds['mass']
    return (fat_mass > 76e3) & (fat_mass < 146e3)

def mass_window_top(ds):
    return ds['mass'] > 60e3

def mass_window_truth_match(ds, mass_window=mass_window_higgs):
    truth_match = ds['GhostHBosonsCount'] == 1
    truth_match &= mass_window(ds)
    return truth_match

def window_pt_range(pt_range, mass_window=mass_window_higgs):
    def selector(ds):
        window = mass_window(ds)
        fat_pt = ds['pt']
        pt_window = (fat_pt > pt_range[0]) & (fat_pt < pt_range[1])
        return window & pt_window
    return selector

def window_pt_range_truth_match(pt_range, mass_window=mass_window_higgs):
    def selector(ds):
        window = mass_window_truth_match(ds, mass_window)
        fat_pt = ds['pt']
        pt_window = (fat_pt > pt_range[0]) & (fat_pt < pt_range[1])
        return window & pt_window
    return selector

#################################
# functions that do real things #
#################################
def get_hist(ds, edges, discriminant, selection=mass_window_higgs):
    hist = 0
    for fpath in glob(f'{ds}/*.h5'):
        with File(fpath,'r') as h5file:
            discrim = discriminant(h5file)
            weight = h5file['fat_jet']['mcEventWeight']
            sel = selection(h5file['fat_jet'])
            hist += np.histogram(
                discrim[sel], edges, weights=weight[sel])[0]
    return hist

def get_hist_reweighted(ds, edges, weights_hist, discriminant,
                        selection=mass_window_truth_match):
    with File(weights_hist, 'r') as h5file:
        num = h5file['higgs']['hist']
        denom = h5file['dijet']['hist']
        ratio_edges = np.asarray(h5file['higgs']['edges'])
        ratio = np.zeros_like(num)
        valid = np.asarray(denom) > 0.0
        ratio[valid] = num[valid] / denom[valid]

    hist = 0
    for fpath in glob(f'{ds}/*.h5'):
        with File(fpath,'r') as h5file:
            pt = h5file['fat_jet']['pt']
            indices = np.digitize(pt, ratio_edges) - 1
            weight = ratio[indices]
            disc = discriminant(h5file)
            sel = selection(h5file['fat_jet'])
            hist += np.histogram(disc[sel], edges, weights=weight[sel])[0]
    return hist


DISCRIMINANT_GETTERS = {
    'tau32_top_vs_qcd': get_tau32,
    'xbb_anti_qcd': make_xbb_getter(ftop=0),
    'xbb_mixed': make_xbb_getter(ftop=0.5),
    'xbb_anti_top': make_xbb_getter(ftop=1.0),
    'dl1': make_dl1_getter(SJ),
    'xbb_top_vs_qcd': get_top_vs_qcd,
    # 'dl1ghost': make_dl1_getter(SJ_GHOST),
    # 'mv2': get_mv2,
    # 'dnn': get_dnn,
}
DISCRIMINANT_EDGES = {
    'dl1': np.linspace(-10, 10, 1e3),
    'dl1ghost': np.linspace(-10, 10, 1e3),
    'mv2': np.linspace(-1, 1, 1e3),
    'dnn': np.linspace(0, 1, 1e3),
    'xbb_anti_qcd': np.linspace(-10, 10, 1e3),
    'xbb_anti_top': np.linspace(-10, 10, 1e3),
    'xbb_mixed': np.linspace(-10, 10, 1e3),
    'xbb_top_vs_qcd': np.linspace(-10, 10, 1e3),
    'tau32_top_vs_qcd': np.linspace(0, 1, 1e3),
}

def write_discriminants(discrims, output_file):
    args = dict(dtype=float)
    for discrim in discrims:
        grp = output_file.create_group(discrim)
        for proc in ['higgs', 'dijet', 'top']:
            grp.create_dataset(proc, data=discrims[discrim][proc], **args)
        grp.create_dataset('edges', data=DISCRIMINANT_EDGES[discrim], **args)

def run():
    args = get_args()

    discrims = {}
    for discrim_name, getter in DISCRIMINANT_GETTERS.items():
        if args.verbose:
            print(f'running {discrim_name}')
        if 'top_vs_qcd' in discrim_name:
            window = mass_window_top
        else:
            window = mass_window_higgs
        dijet_selector = window_pt_range(args.pt_range, window)
        top_selector = window_pt_range(args.pt_range, window)
        higgs_selector = window_pt_range_truth_match(args.pt_range, window)

        edges = DISCRIMINANT_EDGES[discrim_name]
        discrims[discrim_name] = {
            'dijet': get_dijet(edges, args, getter, dijet_selector),
            'higgs': get_process(edges, 'higgs', args, getter, higgs_selector),
            'top': get_process(edges, 'top', args, getter, top_selector)
        }

    if args.save_file:
        with File(args.save_file, 'w') as output_file:
            write_discriminants(discrims, output_file)

    draw_roc_curves(discrims, args.input_hist_dir)

def get_dijet(edges, args, discriminant=get_mv2,
              selection=mass_window_higgs):
    with open(args.denominator, 'r') as denom_file:
        denom = get_denom_dict(denom_file)
    with open(args.cross_sections, 'r') as xsec_file:
        xsecs = CrossSections(xsec_file, denom)
    input_hists = args.input_hist_dir

    hist = 0
    for ds in args.datasets:
        dsid = get_dsid(ds)
        if not is_dijet(dsid, restricted=True):
            continue
        if xsecs.datasets[dsid]['denominator'] == 0:
            continue
        if args.verbose:
            print(f'running on {ds}')
        weight = xsecs.get_weight(dsid)
        this_dsid = get_hist_reweighted(
            ds, edges, f'{input_hists}/jetpt.h5',
            discriminant, selection) * weight
        hist += this_dsid

    return hist

def get_process(edges, process, args, discriminant, selection):
    input_hists = args.input_hist_dir

    hist = 0
    for ds in args.datasets:
        dsid = get_dsid(ds)
        if not SELECTORS[process](dsid):
            continue
        if args.verbose:
            print(f'running on {ds} as {process}')

        this_dsid = get_hist_reweighted(
            ds, edges, f'{input_hists}/jetpt.h5',
            discriminant, selection)
        hist += this_dsid

    return hist

if __name__ == '__main__':
    run()
