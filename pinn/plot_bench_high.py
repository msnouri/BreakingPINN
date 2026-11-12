"""Figure of the steady-wave benchmark near the breaking threshold: PINN against exact B.

    python plot_bench_high.py bench_high [--out fig_bench_high]
Reference cases (lidar-like gaps) are joined by a line; variants at B = 0.849 are labelled by marker.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import figstyle

VARIANTS = {'full': ('full coverage', 's'), 'sharp': (r'sharper $\eta$ features', 'D'),
            'tight': ('data tolerance 5 mm', '^')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('--out', default='fig_bench_high')
    a = ap.parse_args()
    figstyle.use()
    runs = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(os.path.join(a.folder, '*.json'))}
    ref = sorted((r['B_exact'], r['B_pinn_median'], r['u_crest_pinn_median'] / r['u_crest_exact'] - 1)
                 for n, r in runs.items() if n.endswith('_gaps'))
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
    x, y, eu = map(np.array, zip(*ref))
    ax[0].plot([0.4, 0.92], [0.4, 0.92], ':', color='0.5', lw=1.3, label='exact')
    ax[0].plot(x, y, 'o-', color='k', label='PINN, lidar-like gaps')
    ax[1].plot(x, y - x, 'o-', color='k')
    for key, (lab, m) in VARIANTS.items():
        for n, r in runs.items():
            if n.endswith('_' + key):
                ax[0].plot(r['B_exact'], r['B_pinn_median'], m, mfc='white', color='k', ms=9, mew=1.5, label=lab)
                ax[1].plot(r['B_exact'], r['B_pinn_median'] - r['B_exact'], m, mfc='white', color='k', ms=9, mew=1.5)
    ax[0].set_xlabel(r'exact $B$'); ax[0].set_ylabel(r'reconstructed $B$')
    ax[0].set_xlim(0.4, 0.92); ax[0].set_ylim(0.4, 0.92); ax[0].set_aspect('equal')
    ax[0].legend(loc='upper left', bbox_to_anchor=(0.0, 0.92), fontsize=14)
    ax[1].axhline(0, ls=':', color='0.5', lw=1.3)
    ax[1].set_xlabel(r'exact $B$'); ax[1].set_ylabel(r'error in $B$')
    ax[1].set_xlim(0.4, 0.92)
    ax[1].set_ylim(-0.1, 0.015)
    ax[1].set_aspect(0.52 / 0.115)
    figstyle.panel_labels(ax, x=0.04, y=0.96)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(a.out + '.pdf'); fig.savefig(a.out + '.png', dpi=150)
    for b, p, e in ref:
        print(f'B exact {b:.3f}  PINN {p:.3f}  error {p - b:+.3f}  u {100 * e:+.1f}%')


if __name__ == '__main__':
    main()
