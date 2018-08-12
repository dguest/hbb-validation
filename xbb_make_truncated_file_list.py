#!/usr/bin/env python3

"""Limit the number of entries in each dataset

Make a new directory structure that puts a limit on the number of
events in each dataset. Do this stupid-like: after the specified limit
is reached stop adding files.
"""

_dataset_help=(
    "this should be a list of directories, each of which contains"
    " a bunch of files")

from argparse import ArgumentParser
from h5py import File
from glob import glob
from os import symlink, mkdir, makedirs
from pathlib import Path

def get_args():
    parser = ArgumentParser(description=__doc__)
    d = 'default: %(default)s'
    parser.add_argument('datasets', nargs='+', help=_dataset_help)
    parser.add_argument(
        '-o','--output-dir', help="build new file tree here", required=True)
    parser.add_argument('-m','--max_entries', type=int,
                        default=1_000_000, help=d)
    return parser.parse_args()

def run():
    args = get_args()
    makedirs(args.output_dir)
    for ds in args.datasets:
        ds_count = 0
        outdir_path = Path(args.output_dir, Path(ds).name)
        mkdir(outdir_path)
        for fpath in sorted(glob('f{ds}/*.h5')):
            if ds_count > args.max_entries:
                break
            old_file = Path(fpath).resolved()
            new_file = Path(outdir_path, old_file.name)
            

if __name__ == '__main__':
    run()

