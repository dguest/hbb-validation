#!/usr/bin/env python3

"""
Add a weight dataset to reweight histograms by
"""

from argparse import ArgumentParser
from h5py import File

def get_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('datasets', nargs='+')
    parser.add_argument('-d', '--denominator', required=True)
    parser.add_argument('-x', '--cross-sections', required=True)
    return parser.parse_args()

def run():
    args = get_args()
    print(args)

if __name__ == '__main__':
    run()

