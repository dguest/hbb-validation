#!/usr/bin/env python3

"""
Draw Roc curves
"""

from argparse import ArgumentParser
from h5py import File
import numpy as np
import json, os

def get_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('h5_roc_curves')
    parser.add_argument('-d', '--discriminant-list', nargs='*')
    parser.add_argument('-o', '--out-dir', default='plots')
    return parser.parse_args()

def run():
    args = get_args()
    discrims = {}
    discrims_top = {}
    with File(args.h5_roc_curves,'r') as h5file:
        for discrim_name, group in h5file.items():
            ds_names = ['higgs', 'dijet', 'top', 'edges']
            if args.discriminant_list:
                if discrim_name not in args.discriminant_list:
                    continue
            roc_components = {
                x: np.asarray(group[x]) for x in ds_names
            }
            discrims[discrim_name] = roc_components
    draw_roc_curves(discrims, args.out_dir)

def draw_roc_curves(discrims, out_dir):
    discrims_higgs = {}
    discrims_top = {}
    for name, disc in discrims.items():
        if 'top_vs_qcd' in name:
            discrims_top[name] = disc
        else:
            discrims_higgs[name] = disc
    draw_roc_curves_multijet(discrims_higgs, out_dir)
    draw_roc_curves_top(discrims_higgs, out_dir)
    draw_top_tagging_roc_curves(discrims_top, out_dir)

def draw_roc_curves_multijet(discrims, out_dir):
    from xbb.mpl import Canvas
    with Canvas(f'{out_dir}/roc_multijet.pdf') as can:
        for dis_name, discrims in discrims.items():
            sig, bg = discrims['higgs'], discrims['dijet']
            draw_roc(can, sig, bg, out_dir, label=dis_name.upper())
        can.ax.set_yscale('log')
        can.ax.legend()
        can.ax.set_ylabel('Multijet Rejection')
        can.ax.set_xlabel('Higgs Efficiency')

def draw_roc_curves_top(discrims, out_dir):
    from xbb.mpl import Canvas
    with Canvas(f'{out_dir}/roc_top.pdf') as can:
        for dis_name, discrims in discrims.items():
            sig, bg = discrims['higgs'], discrims['top']
            draw_roc(can, sig, bg, out_dir, label=dis_name.upper())
        can.ax.set_yscale('log')
        can.ax.legend()
        can.ax.set_ylabel('Top Rejection')
        can.ax.set_xlabel('Higgs Efficiency')

def draw_top_tagging_roc_curves(discrims, out_dir):
    from xbb.mpl import Canvas
    with Canvas(f'{out_dir}/top_tagging_roc.pdf') as can:
        for dis_name, discrims in discrims.items():
            sig, bg = discrims['top'], discrims['dijet']
            draw_roc(can, sig, bg, out_dir, label=dis_name.upper())
        can.ax.set_yscale('log')
        can.ax.legend()
        can.ax.set_ylabel('Multijet Rejection')
        can.ax.set_xlabel('Top Efficiency')


def draw_roc(canvas, sig, bg, out_dir, label, min_eff=0.4):
    from xbb.mpl import Canvas
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    eff = np.cumsum(sig[::-1])[::-1]
    eff /= eff.max()
    bg_eff = np.cumsum(bg[::-1])[::-1]
    bg_eff /= bg_eff.max()
    rej = np.zeros_like(bg_eff)
    valid = bg_eff > 0.0
    rej[valid] = 1/bg_eff[valid]

    xbins = np.arange(sig.size)

    with Canvas(f'{out_dir}/{label}.pdf') as can:
        can.ax.step(xbins, eff, label='signal')
        can.ax.step(xbins, bg_eff, label='bg')
        can.ax.legend()

    valid_eff = eff > min_eff
    canvas.ax.plot(eff[valid_eff], rej[valid_eff], label=label)

if __name__ == '__main__':
    run()
