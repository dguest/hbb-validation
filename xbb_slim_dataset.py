#!/usr/bin/env python3

"""
Make slim datasets
"""

from argparse import ArgumentParser
from h5py import File
import numpy as np
import json, os
from pathlib import Path

def get_args():
    parser = ArgumentParser(description=__doc__)
    d = 'default: %(default)s'
    parser.add_argument('dataset', type=Path)
    parser.add_argument('-o', '--out-dir', default='slimmed', type=Path)
    parser.add_argument('-f', '--force', action='store_true')
    return parser.parse_args()

SUBJET_VARS = [*[f'DL1_p{x}' for x in 'cub'], 'MV2c10_discriminant']

DATASETS_N_VARS = {
    'fat_jet': [
        'pt', 'eta', 'mass', 'mcEventWeight',
        *[f'XbbScore{x}' for x in ['Top', 'QCD', 'Higgs']],
        'Tau32_wta', 'JSSTopScore',
    ],
    'subjet_VR_1': SUBJET_VARS,
    'subjet_VR_2': SUBJET_VARS,
    'subjet_VR_3': SUBJET_VARS,
}

def build_slimmed_file_for(ds, out_ds):
    paths = list(ds.glob('*.h5'))
    out_datasets = {}
    with File(paths[0], 'r') as base:
        for name, dsv in DATASETS_N_VARS.items():
            out_datasets[name] = OutputDataset(base[name], out_ds, name, dsv)
        metadata = np.asarray(base['metadata'])
    for fpath in paths:
        with File(fpath,'r') as infile:
            for ds_name, ods in out_datasets.items():
                ods.add(infile[ds_name])
            md = infile['metadata']
            for key in metadata.dtype.names:
                metadata[key] += md[key]
    out_ds.create_dataset('metadata', data=metadata)

class OutputDataset:
    def __init__(self, base_ds, out_group, output_name, variables):
        types = [(x, base_ds.dtype[x]) for x in variables]
        self.dataset = out_group.create_dataset(
            output_name, (0,), maxshape=(None,),
            dtype=types, chunks=(1000,),
            compression='gzip', compression_opts=7)
    def add(self, ds):
        oldmark = self.dataset.shape[0]
        self.dataset.resize(oldmark + ds.shape[0], axis=0)
        fields = self.dataset.dtype.names
        self.dataset[oldmark:] = np.asarray(ds[fields])


def run():
    args = get_args()
    output_ds = args.out_dir.joinpath(args.dataset.name)
    output_ds.mkdir(parents=True, exist_ok=args.force)
    with File(output_ds.joinpath('all.h5'), 'w') as output:
        build_slimmed_file_for(args.dataset, output)

if __name__ == '__main__':
    run()
