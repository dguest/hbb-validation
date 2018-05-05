#!/usr/bin/env python3

"""
Make histograms of the jet pt spectra
"""

from argparse import ArgumentParser
from h5py import File
import numpy as np
import sys, os
from xbb import mpl

def get_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('kinematics_file')
    parser.add_argument('-o', '--out-file', default='discriminants.h5')
    return parser.parse_args()


def run():
    args = get_args()
    ds_opt = dict(compression='gzip')
    with File(args.kinematics_file, 'r') as in_file:
        with File(args.out_file,'w') as out_file:
            for procname, ds in in_file.items():
                hist, edges = get_score_hist(ds)
                out_grp = out_file.create_group(procname)
                out_grp.create_dataset('hist', data=hist, **ds_opt)
                out_grp.create_dataset('edges', data=edges, **ds_opt)

def get_score_hist(ds):
    edges = np.concatenate([[-np.inf], np.linspace(0, 1, 1e3), [np.inf]])
    score = ds['HbbScore']
    # we need to up the weight precision because some of them are
    # really small. This isn't normally a problem, but when they are
    # combined in a numpy histogram the lower weights get lost.
    weights = np.array(ds['weights'], dtype=np.float128)
    hist, edges = np.histogram(score, edges, weights=weights)
    return np.array(hist, dtype=float), edges

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


if __name__ == '__main__':
    run()
