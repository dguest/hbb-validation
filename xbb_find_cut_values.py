#!/usr/bin/env python3

"""
Get cut values for h-tagging discriminants
"""

from argparse import ArgumentParser
from h5py import File
import numpy as np

def get_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('distributinos', help='h5 file')
    return parser.parse_args()

def run():
    args = get_args()
    samples = {}
    with File(args.distributinos, 'r') as dists:
        for sample, discrim_hist in dists.items():
            samples[sample] = get_cut_values(discrim_hist)
    print(samples)

def get_cut_values(tagger_dists, effs=[0.5, 0.1, 0.01]):
    hist = np.asarray(tagger_dists['hist'])[:1:-1]
    cuts = np.asarray(tagger_dists['edges'])[-2:1:-1]
    integral = hist.cumsum() / hist.sum()
    interp_cuts = np.interp(effs, integral, cuts)
    return interp_cuts

if __name__ == '__main__':
    run()
