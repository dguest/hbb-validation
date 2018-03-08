#!/usr/bin/env python3

"""
Make histograms of the jet pt spectra
"""

from argparse import ArgumentParser
from h5py import File
from glob import glob
import numpy as np
import json, os

from common import get_denom_dict, get_dsid
from common import is_dijet, is_ditop, is_dihiggs
from cross_section import CrossSections

def get_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('datasets', nargs='+')
    parser.add_argument('-d', '--denominator', required=True)
    parser.add_argument('-x', '--cross-sections', required=True)
    parser.add_argument('-o', '--out-dir', default='pt-hists')
    return parser.parse_args()

def get_mv2(h5file, discriminant='MV2c10_discriminant'):
    disc1 = h5file['subjet1'][discriminant]
    disc2 = h5file['subjet2'][discriminant]
    discrim_comb = np.stack([disc1, disc2], axis=1).min(axis=1)
    invalid = np.isnan(discrim_comb)
    discrim_comb[invalid] = -1.0
    return discrim_comb

def get_dl1(h5file):
    sj1 = h5file['subjet1']
    disc1 = sj1['DL1_pb'] / sj1['DL1_pu']
    sj2 = h5file['subjet1']
    disc2 = sj2['DL1_pb'] / sj2['DL1_pu']
    discrim_comb = np.stack([disc1, disc2], axis=1).min(axis=1)
    invalid = np.isnan(discrim_comb) | np.isinf(discrim_comb)
    discrim_comb[invalid] = 1e-15
    return np.log(np.clip(discrim_comb, 1e-30, 1e30))

def mass_window(ds):
    fat_mass = ds['fat_jet']['mass']
    return (fat_mass > 76e3) & (fat_mass < 146e3)

def mass_window_truth_match(ds):
    truth_match = ds['fat_jet']['GhostHBosonsCount'] == 1
    truth_match &= mass_window(ds)
    return truth_match

def get_hist(ds, edges, discriminant, selection=mass_window):
    hist = 0
    for fpath in glob(f'{ds}/*.h5'):
        with File(fpath,'r') as h5file:
            discrim = discriminant(h5file)
            weight = h5file['fat_jet']['mcEventWeight']
            sel = selection(h5file)
            hist += np.histogram(
                discrim[sel], edges, weights=weight[sel])[0]
    return hist

def get_hist_reweighted(ds, edges, weights_hist, discriminant,
                        selection=mass_window_truth_match):
    with File(weights_hist, 'r') as h5file:
        num = h5file['dijet']['hist']
        denom = h5file['higgs']['hist']
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
            sel = selection(h5file)
            hist += np.histogram(disc[sel], edges, weights=weight[sel])[0]
    return hist

def draw_roc(canvas, sig, bg, out_dir, label, min_eff=0.4):
    from mpl import Canvas
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    eff = np.cumsum(sig[::-1])[::-1]
    eff /= eff.max()
    bg_eff = np.cumsum(bg[::-1])[::-1]
    bg_eff /= bg_eff.max()
    rej = np.zeros_like(bg_eff)
    valid = bg_eff > 0.0
    rej[valid] = 1/bg_eff[valid]

    xbins = np.arange(sig.size)

    with Canvas(f'{out_dir}/{label}.pdf') as can:
        can.ax.step(xbins[1:], sig[1:], label='signal')
        can.ax.step(xbins[1:], bg[1:], label='bg')
        can.ax.legend()

    valid_eff = eff > min_eff
    canvas.ax.plot(eff[valid_eff], rej[valid_eff], label=label)

def run():
    args = get_args()
    edges = np.linspace(-1, 1, 1e3)
    edges_dl1 = np.linspace(-10, 10, 1e3)

    bg = get_dijet(edges, args, get_mv2)
    sig = get_higgs_reweighted(edges, args, get_mv2)
    bg_dl1 = get_dijet(edges_dl1, args, get_dl1)
    sig_dl1 = get_higgs_reweighted(edges_dl1, args, get_dl1)
    from mpl import Canvas
    with Canvas(f'{args.out_dir}/roc.pdf') as can:
        draw_roc(can, sig, bg, args.out_dir, label='mv2')
        draw_roc(can, sig_dl1, bg_dl1, args.out_dir, label='dl1')
        can.ax.set_yscale('log')
        can.ax.legend()

def get_dijet(edges, args, discriminant=get_mv2):
    with open(args.denominator, 'r') as denom_file:
        denom = get_denom_dict(denom_file)
    with open(args.cross_sections, 'r') as xsec_file:
        xsecs = CrossSections(xsec_file, denom)

    hist = 0
    for ds in args.datasets:
        dsid = get_dsid(ds)
        if not is_dijet(dsid):
            continue
        if xsecs.datasets[dsid]['denominator'] == 0:
            continue
        weight = xsecs.get_weight(dsid)
        this_dsid = get_hist(ds, edges, discriminant) * weight
        hist += this_dsid

    return hist

def get_higgs_reweighted(edges, args, discriminant=get_mv2):
    out_dir = args.out_dir

    hist = 0
    for ds in args.datasets:
        dsid = get_dsid(ds)
        if not is_dihiggs(dsid):
            continue

        this_dsid = get_hist_reweighted(ds, edges, f'{out_dir}/jetpt.h5',
                                        discriminant)
        hist += this_dsid

    return hist

if __name__ == '__main__':
    run()
