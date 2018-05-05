#!/usr/bin/env python3

"""
Make histograms of the jet pt spectra
"""

from argparse import ArgumentParser
# this will be fixed after h5py 2.8
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore', FutureWarning)
    from h5py import File
from glob import glob
import numpy as np
import json, os
from collections import defaultdict, Counter

from common import get_denom_dict, get_dsid
from common import is_dijet, is_ditop, is_dihiggs
from cross_section import CrossSections

def get_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('datasets', nargs='+')
    parser.add_argument('-d', '--denominator', required=True)
    parser.add_argument('-x', '--cross-sections', required=True)
    parser.add_argument('-r', '--ratio-dir', default='pt-hists')
    parser.add_argument('-o', '--out-file', default='variables.json')
    parser.add_argument('-n', '--sample-size', default=10000, type=int)
    parser.add_argument('-p', '--plots-dir')
    return parser.parse_args()


def run():
    args = get_args()
    build_summary_dataset(args.datasets, args)

DEFAULT_DATASETS = ['fat_jet', 'subjet1', 'subjet2', 'subjet3']
def sample_slice(file_dir, requested_entries, dataset_names=DEFAULT_DATASETS):
    files = glob(f'{file_dir}/*.h5')
    datasets = defaultdict(list)
    total_entries = 0
    total_added = 0
    for file_name in files:
        with File(file_name, 'r') as h5file:
            n_entries = 0
            needed = max(requested_entries - total_entries, 0)
            for ds_name in dataset_names:
                ds = h5file[ds_name]
                assert n_entries == 0 or ds.shape[0] == n_entries
                n_entries = ds.shape[0]
                if needed:
                    datasets[ds_name].append(ds[0:needed])
            total_entries += ds.shape[0]
            total_added += min(ds.shape[0], needed)
    out_datasets = {n: np.concatenate(lst) for n, lst in datasets.items()}
    return out_datasets, total_entries

def get_ratio(out_dir):
    with File(f'{out_dir}/jetpt.h5', 'r') as h5file:
        num = h5file['dijet']['hist']
        denom = h5file['higgs']['hist']
        ratio = np.zeros_like(num)
        valid = np.asarray(denom) > 0.0
        ratio[valid] = num[valid] / denom[valid]

        edges = np.asarray(h5file['dijet']['edges'])

    return ratio, edges

def build_summary_dataset(datasets, args):
    with open(args.denominator, 'r') as denom_file:
        denom = get_denom_dict(denom_file)
    with open(args.cross_sections, 'r') as xsec_file:
        xsecs = CrossSections(xsec_file, denom)

    samples = defaultdict(lambda: defaultdict(list))
    for ds_name in datasets:
        dsid = get_dsid(ds_name)
        if is_dijet(dsid):
            if xsecs.datasets[dsid]['denominator'] == 0:
                continue
            sample, total_entries = sample_slice(ds_name, args.sample_size)
            upweight = total_entries / sample['fat_jet'].shape[0]
            weight = xsecs.get_weight(dsid) * upweight
            sample['fat_jet']['mcEventWeight'] *= weight
            for object_name, obj in sample.items():
                samples['dijet'][object_name].append(obj)
        elif is_dihiggs(dsid):
            sample, total_entries = sample_slice(ds_name, args.sample_size)
            upweight = total_entries / sample['fat_jet'].shape[0]
            pt = sample['fat_jet']['pt']
            ratio, edges = get_ratio(args.ratio_dir)
            indices = np.digitize(pt, edges) - 1
            weight = ratio[indices] * upweight
            sample['fat_jet']['mcEventWeight'] = weight
            for object_name, obj in sample.items():
                samples['higgs'][object_name].append(obj)
        else:
            print(f'skipping {dsid}!')
            continue

    # merge things
    merged_samples = defaultdict(dict)
    for phys_proc_name, obj_dict in samples.items():
        for obj_name, samp_list in obj_dict.items():
            merged = np.concatenate(samp_list, 0)
            merged_samples[phys_proc_name][obj_name] = merged

    if args.plots_dir:
        draw_hist(merged_samples, ['higgs', 'dijet'], args.plots_dir)

def draw_hist(merged_samples, procs, out_dir, batch_size=1000):
    from mpl import Canvas
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    obj_name, var_name = 'fat_jet', 'pt'
    edges = np.concatenate([[-np.inf], np.linspace(0, 3e6, 101), [np.inf]])
    centers = 1e-6 * (edges[1:] + edges[:-1]) / 2
    gev_per_bin = (centers[2] - centers[1]) * 1e3
    with Canvas(f'{out_dir}/jet-pt.pdf') as can:
        for proc in procs:
            var = merged_samples[proc][obj_name][var_name]
            weight = merged_samples[proc]['fat_jet']['mcEventWeight']
            mega_weights = np.array(weight, dtype=np.float128)
            hist = np.histogram(var, edges, weights=mega_weights)[0]
            can.ax.plot(centers, hist, label=proc)
        can.ax.set_yscale('log')
        can.ax.set_ylabel(f'jets * fb / {gev_per_bin:.0f} TeV')
        can.ax.set_xlabel(r'Fat Jet $p_{\rm T}$ [TeV]')
        maxval = can.ax.get_ylim()[1]
        can.ax.set_ylim(0.1, maxval)
        can.ax.legend(ncol=2)

if __name__ == '__main__':
    run()
