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
from os.path import relpath

def get_args():
    parser = ArgumentParser(description=__doc__)
    d = 'default: %(default)s'
    parser.add_argument('datasets', nargs='+', help=_dataset_help, type=Path)
    parser.add_argument(
        '-o','--output-dir', help="build new file tree here", required=True,
        type=Path)
    parser.add_argument('-m','--max_entries', type=int,
                        default=1_000_000, help=d)
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser.parse_args()

def run():
    args = get_args()
    makedirs(args.output_dir)
    for ds in args.datasets:
        ds_count = 0
        outdir_path = Path(args.output_dir, ds.name)
        mkdir(outdir_path)
        if args.verbose:
            print(f'adding {ds.name}')
        for fpath in sorted(ds.glob('*.h5')):
            if ds_count > args.max_entries:
                if args.verbose:
                    print(f'added {ds_count:,} entries to {ds.name}, stopping')
                break
            old_file = Path(fpath).resolve()
            new_file = outdir_path.joinpath(old_file.name).resolve()
            with File(old_file, 'r') as h5file:
                ds_count += h5file['fat_jet'].shape[0]
            link_path = relpath(old_file, new_file.parent)
            symlink(link_path, new_file)

if __name__ == '__main__':
    run()

