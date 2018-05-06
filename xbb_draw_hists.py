#!/usr/bin/env python3

"""
Draw Histograms
"""

from argparse import ArgumentParser
from h5py import File
import numpy as np
import os

def get_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('hists')
    parser.add_argument('-o', '--out-dir', default='plots')
    return parser.parse_args()

def run():
    args = get_args()
    discrims = {}
    with File(args.hists,'r') as h5file:
        for varname, var_group in h5file.items():
            draw_hist(var_group, varname, args.out_dir)

def draw_hist(var_group, varname, out_dir):
    from xbb.mpl import Canvas
    with Canvas(f'{out_dir}/{varname}.pdf') as can:
        for sampname, hist_group in var_group.items():
            edges = hist_group['edges'][1:-1]
            centers = (edges[:-1] + edges[1:]) / 2
            bins = hist_group['hist'][1:-1]
            can.ax.plot(centers, bins, label=sampname)
        can.ax.set_yscale('log')
        can.ax.legend()
        can.ax.set_ylabel('Things')
        can.ax.set_xlabel(varname)


if __name__ == '__main__':
    run()
