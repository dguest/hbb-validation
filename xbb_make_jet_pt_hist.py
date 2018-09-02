#!/usr/bin/env python3

"""
Make histograms of the jet pt spectra
"""

from argparse import ArgumentParser
from h5py import File
from glob import glob
import numpy as np
import json, os
from sys import stderr
from numpy.lib.recfunctions import append_fields

from xbb.common import get_denom_dict, get_dsid
from xbb.common import SELECTORS
from xbb.common import is_dijet, is_dihiggs
from xbb.cross_section import CrossSections
from xbb.selectors import truth_match, EVENT_LABELS, all_events

def get_args():
    parser = ArgumentParser(description=__doc__)
    d = 'default: %(default)s'
    parser.add_argument('datasets', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-d', '--denominator', required=True)
    parser.add_argument('-x', '--cross-sections', required=True)
    parser.add_argument('-o', '--out-dir', default='pt-hists')
    parser.add_argument('-c', '--discrim-cut', type=float)
    parser.add_argument('-f', '--output-fields', nargs='+',
                        default=['pt', 'mass', 'HbbScore'], help=d)
    out_format = parser.add_mutually_exclusive_group()
    out_format.add_argument('-m', '--kinematic-ntuple')
    return parser.parse_args()

def get_hist(ds, edges, selection=all_events, output_dataset=None, ds_wt=1.0):
    hist = 0
    for fpath in glob(f'{ds}/*.h5'):
        with File(fpath,'r') as h5file:
            pt = h5file['fat_jet']['pt']
            weight = h5file['fat_jet']['mcEventWeight']
            mega_weights = np.array(weight, dtype=np.longdouble) * ds_wt
            sel_index = selection(h5file['fat_jet'])
            hist += np.histogram(
                pt[sel_index], edges, weights=mega_weights[sel_index])[0]
            if output_dataset:
                output_dataset.add(h5file, mega_weights)
            if np.any(np.isnan(hist)):
                stderr.write(f'{fpath} has nans\n')
    return hist

def get_hist_reweighted(ds, edges, ratio, selection, output_dataset=None):
    hist = 0
    for fpath in glob(f'{ds}/*.h5'):
        with File(fpath,'r') as h5file:
            pt = h5file['fat_jet']['pt']
            indices = np.digitize(pt, edges) - 1
            weight = ratio[indices]
            mega_weights = np.array(weight, dtype=np.longdouble)
            sel_index = selection(h5file['fat_jet'])
            hist += np.histogram(pt[sel_index], edges,
                                 weights=mega_weights[sel_index])[0]
            if output_dataset:
                output_dataset.add(h5file, mega_weights)
    return hist

class OutputDataset:
    def __init__(self, h5file, old_file, ds_name, variables):
        self.variables = tuple(variables)
        types = [(x, old_file['fat_jet'].dtype[x]) for x in variables]
        types.append( ('weights', float) )
        self.dataset = h5file.create_dataset(
            ds_name, (0,), maxshape=(None,), dtype=types, chunks=(1000,),
            compression='gzip', compression_opts=7)
    def add(self, h5file, weights):
        fat = h5file['fat_jet']
        oldmark = self.dataset.shape[0]
        self.dataset.resize(oldmark + fat.shape[0], axis=0)
        slim_fat = fat[self.variables]
        self.dataset[oldmark:] = append_fields(
            slim_fat, 'weights', data=weights)

def draw_hist(hist, edges, out_dir, parts={}, file_name='dijet.pdf'):
    from xbb.mpl import Canvas
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    centers = 1e-6 * (edges[1:] + edges[:-1]) / 2
    gev_per_bin = (centers[2] - centers[1]) * 1e3
    with Canvas(f'{out_dir}/{file_name}') as can:
        can.ax.plot(centers, hist)
        can.ax.set_yscale('log')
        can.ax.set_ylabel(f'jets * fb / {gev_per_bin:.0f} TeV')
        can.ax.set_xlabel(r'Fat Jet $p_{\rm T}$ [TeV]')
        maxval = can.ax.get_ylim()[1]
        can.ax.set_ylim(0.1, maxval)
        for dsid, part in parts.items():
            can.ax.plot(centers, part, label=str(dsid))
        can.ax.legend(ncol=2)

def save_hist(hist, edges, out_dir, file_name, group_name):
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    with File(f'{out_dir}/{file_name}','a') as h5file:
        hist_group = h5file.create_group(group_name)
        # we need a bit of a hack here: float128 (longdouble) doesn't
        # get stored properly by h5py as of version 2.8
        hist_group.create_dataset('hist', data=hist, dtype=float)
        hist_group.create_dataset('edges', data=edges)

def run():
    args = get_args()
    edges = np.concatenate([[-np.inf], np.linspace(0, 3e6, 101), [np.inf]])
    out_hists = os.path.join(args.out_dir, 'jetpt.h5')
    if os.path.isfile(out_hists):
        os.remove(out_hists)
    if args.kinematic_ntuple:
        out_file = File(args.kinematic_ntuple, 'w')
    else:
        out_file = None
    run_dijet(edges, args, out_file)
    run_sample(edges, 'higgs', args)
    run_sample(edges, 'top', args)
    run_higgs_reweighted(edges, args, out_file)

    if out_file:
        out_file.close()

def run_dijet(edges, args, output_file):
    with open(args.denominator, 'r') as denom_file:
        denom = get_denom_dict(denom_file)
    with open(args.cross_sections, 'r') as xsec_file:
        xsecs = CrossSections(xsec_file, denom)

    if output_file:
        with File(glob(args.datasets[0] + '/*.h5')[0], 'r') as old_file:
            out_ds = OutputDataset(
                output_file, old_file, 'dijet', args.output_fields)
    else:
        out_ds = None

    parts = {}
    hist = 0
    for ds in args.datasets:
        dsid = get_dsid(ds)
        if not is_dijet(dsid, restricted=True):
            continue
        if args.verbose:
            print(f'processing {ds} as dijet')
        if xsecs.datasets[dsid]['denominator'] == 0:
            continue
        weight = xsecs.get_weight(dsid)

        this_dsid = get_hist(ds, edges, all_events, out_ds, weight)
        parts[dsid] = np.array(this_dsid)
        hist += this_dsid

    draw_hist(hist, edges, args.out_dir, parts, file_name='dijet.pdf')
    save_hist(hist, edges, args.out_dir, 'jetpt.h5', 'dijet')

def run_sample(edges, process, args):
    hist = 0
    parts = {}
    selector = truth_match(EVENT_LABELS[process])
    for ds in args.datasets:
        dsid = get_dsid(ds)
        if not SELECTORS[process](dsid, restricted=True):
            continue
        if args.verbose:
            print(f'processing {ds} as {process}')

        this_dsid = get_hist(ds, edges, selector)
        parts[dsid] = np.array(this_dsid)
        hist += this_dsid

    draw_hist(hist, edges, args.out_dir, parts,
              file_name=f'{process}.pdf')
    save_hist(hist, edges, args.out_dir, 'jetpt.h5', process)

def run_higgs_reweighted(edges, args, output_file):
    hist = 0
    parts = {}
    out_dir = args.out_dir
    with File(f'{out_dir}/jetpt.h5', 'r') as h5file:
        num = h5file['dijet']['hist']
        denom = h5file['higgs']['hist']
        ratio = np.zeros_like(num)
        valid = np.asarray(denom) > 0.0
        ratio[valid] = num[valid] / denom[valid]

    if output_file:
        with File(glob(args.datasets[0] + '/*.h5')[0], 'r') as old_file:
            out_ds = OutputDataset(
                output_file, old_file, 'higgs', args.output_fields)
    else:
        out_ds = None

    for ds in args.datasets:
        dsid = get_dsid(ds)
        if not is_dihiggs(dsid, restricted=True):
            continue
        if args.verbose:
            print(f'processing {ds} as higgs')

        this_dsid = get_hist_reweighted(ds, edges, ratio,all_events, out_ds)
        parts[dsid] = np.array(this_dsid)
        hist += this_dsid

    draw_hist(hist, edges, args.out_dir, parts, file_name='higgs_reweight.pdf')

if __name__ == '__main__':
    run()
