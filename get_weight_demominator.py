#!/usr/bin/env python3

"""
Gets the number of events in each sample, for normalization stuffs
"""

from argparse import ArgumentParser
from h5py import File
from glob import glob
from collections import Counter

import os, sys
import json

def get_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('datasets', nargs='+')
    parser.add_argument('-o','--out-file')
    return parser.parse_args()

def get_dsid(fpath):
    return int(os.path.basename(fpath).split('.')[2])

def is_dijet(dsid):
    return 361020 <= dsid <= 361032

def is_ditop(dsid):
    return 301322 <= dsid <= 301335

def is_dihiggs(dsid):
    return 301488 <= dsid <= 301507

def get_denom_dict(denom_file):
    return {int(k): v for k, v in json.load(denom_file).items()}

def get_counts(fpath, is_dijet):
    key = 'nEventsProcessed' if is_dijet else 'sumOfWeights'
    with File(fpath, 'r') as h5file:
        return float(h5file['metadata'][key])

def run():
    args = get_args()
    counts = Counter()
    for ds in args.datasets:
        dsid = get_dsid(ds)
        is_dijet = is_dijet(dsid)
        for fpath in glob(f'{ds}/*.h5'):
            counts[dsid] += get_counts(fpath, is_dijet)

    if args.out_file:
        out = open(args.out_file, 'w')
    else:
        out = sys.stdout
    json.dump(dict(counts), sys.stdout, indent=2)

if __name__ == '__main__':
    run()
