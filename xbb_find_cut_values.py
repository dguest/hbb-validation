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
    taggers = {}
    with File(args.distributinos, 'r') as dists:
        for tagger_name, tagger_dists in dists.items():
            taggers[tagger_name] = get_cut_values(tagger_dists)
    print(taggers)

def get_cut_values(tagger_dists, bg_effs=[0.5, 0.1, 0.01]):
    bg = np.asarray(tagger_dists['bg'])
    cuts = np.asarray(tagger_dists['edges'])[-2::-1]
    bg_integral = bg[::-1].cumsum() / bg.sum()
    print(bg_integral.shape, cuts.shape)
    interp_cuts = np.interp(bg_effs, bg_integral, cuts)
    return interp_cuts

if __name__ == '__main__':
    run()
