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
    parser.add_argument('-o', '--out-file', default='histograms.h5')
    return parser.parse_args()

def run():
    args = get_args()
    with File(args.kinematics_file, 'r') as in_file:
        with File(args.out_file,'w') as out_file:
            for procname, ds in in_file.items():
                make_hist(ds, out_file, procname)
                make_hist(ds, out_file, procname, 'pt', (0, 4e6))
                make_hist(ds, out_file, procname, 'mass', (0, 500e3))

def make_hist(ds, out_grp, procname, variable='HbbScore', rng=(0,1)):
    edges = np.concatenate([[-np.inf], np.linspace(*rng, 1e3), [np.inf]])
    score = ds[variable]
    # we need to up the weight precision because some of them are
    # really small. This isn't normally a problem, but when they are
    # combined in a numpy histogram the lower weights get lost.
    weights = np.array(ds['weights'], dtype=np.float128)
    hist, edges = np.histogram(score, edges, weights=weights)
    var_grp = out_grp.require_group(variable)
    hist_grp = var_grp.create_group(procname)
    ds_opt = dict(compression='gzip')
    hist_grp.create_dataset('hist', data=hist, dtype=float, **ds_opt)
    hist_grp.create_dataset('edges', data=edges, **ds_opt)

if __name__ == '__main__':
    run()
