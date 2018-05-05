#!/usr/bin/env python3

"""
Gets the number of events in each sample, for normalization stuffs
"""

from argparse import ArgumentParser
from h5py import File
from glob import glob
from collections import Counter
from common import get_dsid, is_dijet

import sys
import json

def get_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('datasets', nargs='+')
    parser.add_argument('-o','--out-file')
    return parser.parse_args()

def get_counts(fpath, is_dijet):
    key = 'nEventsProcessed' if is_dijet else 'sumOfWeights'
    with File(fpath, 'r') as h5file:
        return float(h5file['metadata'][key])

def run():
    args = get_args()
    counts = Counter()
    for ds in args.datasets:
        dsid = get_dsid(ds)
        for fpath in glob(f'{ds}/*.h5'):
            counts[dsid] += get_counts(fpath, is_dijet=is_dijet(dsid))

    if args.out_file:
        out = open(args.out_file, 'w')
    else:
        out = sys.stdout
    json.dump(dict(counts), out, indent=2)

if __name__ == '__main__':
    run()
