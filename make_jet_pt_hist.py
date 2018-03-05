#!/usr/bin/env python3

"""
Make histograms of the jet pt spectra
"""

from argparse import ArgumentParser
from h5py import File
from glob import glob
import numpy as np
import json, os

from get_weight_demominator import get_denom_dict, get_dsid
from cross_section import CrossSections

def get_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('datasets', nargs='+')
    parser.add_argument('-d', '--denominator', required=True)
    parser.add_argument('-x', '--cross-sections', required=True)
    parser.add_argument('-o', '--out-dir', default='plots')
    return parser.parse_args()

def get_hist(ds, edges):
    hist = 0
    for fpath in glob(f'{ds}/*.h5'):
        with File(fpath,'r') as h5file:
            pt = h5file['fat_jet']['pt']
            weight = h5file['fat_jet']['mcEventWeight']
            hist += np.histogram(pt, edges, weights=weight)[0]
    return hist

def draw_hist(hist, edges, out_dir, parts={}):
    from mpl import Canvas
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    centers = 1e-6 * (edges[1:] + edges[:-1]) / 2
    gev_per_bin = (centers[1] - centers[0]) * 1e3
    with Canvas(f'{out_dir}/dijet.pdf') as can:
        can.ax.plot(centers, hist)
        can.ax.set_yscale('log')
        can.ax.set_ylabel(f'jets * fb / {gev_per_bin:.0f} GeV')
        can.ax.set_xlabel(r'Fat Jet $p_{\rm T}$ [GeV]')
        for dsid, part in parts.items():
            can.ax.plot(centers, part, label=str(dsid))
        can.ax.legend()

def run():
    args = get_args()
    with open(args.denominator, 'r') as denom_file:
        denom = get_denom_dict(denom_file)
    with open(args.cross_sections, 'r') as xsec_file:
        xsecs = CrossSections(xsec_file, denom)

    parts = {}
    hist = 0
    edges = np.linspace(0, 7e6, 101)
    for ds in args.datasets:
        dsid = get_dsid(ds)
        if xsecs.datasets[dsid]['nevt'] == 0:
            continue
        weight = xsecs.get_weight(dsid)

        this_dsid = get_hist(ds, edges) * weight
        parts[dsid] = np.array(this_dsid)
        hist += this_dsid

    draw_hist(hist, edges, args.out_dir, parts)

if __name__ == '__main__':
    run()
