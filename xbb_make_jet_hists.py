#!/usr/bin/env python3

"""
Make histograms of the jet pt spectra
"""

from argparse import ArgumentParser
from h5py import File
import numpy as np
import sys, os
import mpl

def get_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('kinematics_file')
    parser.add_argument('-o', '--out-dir', default='pt-hists')
    return parser.parse_args()

def get_hist(ds, edges, group_name, variable_name):
    hist = 0
    for fpath in glob(f'{ds}/*.h5'):
        with File(fpath,'r') as h5file:
            pt = h5file['fat_jet']['pt']
            weight = h5file['fat_jet']['mcEventWeight']
            var = h5file[group_name][variable_name]
            hist += np.histogram(var, edges, weights=weight)[0]
    return hist

def get_hist_reweighted(ds, edges, ratio, group_name, variable_namd):
    hist = 0
    for fpath in glob(f'{ds}/*.h5'):
        with File(fpath,'r') as h5file:
            pt = h5file['fat_jet']['pt']
            indices = np.digitize(pt, edges) - 1
            weight = ratio[indices]
            var = h5file[group_name][variable_name]
            hist += np.histogram(var, edges, weights=weight)[0]
    return hist


def draw_hist(hist, edges, out_dir, parts={}, file_name='dijet.pdf'):
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    centers = 1e-6 * (edges[1:] + edges[:-1]) / 2
    gev_per_bin = (centers[2] - centers[1]) * 1e3
    with Canvas(f'{out_dir}/{file_name}') as can:
        can.ax.plot(centers, hist)
        can.ax.set_yscale('log')
        can.ax.set_ylabel(f'jets * fb / {gev_per_bin:.0f} TeV')
        can.ax.set_xlabel(r'Fat Jet $p_{\rm T}$ [TeV]')
        maxval = can.ax.get_ylim()[1]
        can.ax.set_ylim(0.1, maxval)
        for dsid, part in parts.items():
            can.ax.plot(centers, part, label=str(dsid))
        can.ax.legend(ncol=2)

def save_hist(hist, edges, out_dir, file_name, group_name):
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    with File(f'{out_dir}/{file_name}','a') as h5file:
        hist_group = h5file.create_group(group_name)
        hist_group.create_dataset('hist', data=hist)
        hist_group.create_dataset('edges', data=edges)

def run():
    args = get_args()


if __name__ == '__main__':
    run()
