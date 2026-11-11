"""Figure of the field results: B_max of the breakers (and of the non-breaking controls when given).

    python plot_breakers.py runs/batch_breakers/summary.csv [--controls runs/batch_controls/summary.csv] [--out fig_field_Bmax]

Quality check (QC) for each wave: the physics residuals near the crest within a few tolerances
(kinematic < 5, Laplace < 10) and a physical crest speed (3 < C < 9 m/s at B_max).
Writes <out>.pdf (for the paper) and <out>.png.
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figstyle


def qc(d, kin=5.0, lap=10.0, cmin=3.0, cmax=9.0):
    return (d.crest_kin < kin) & (d.crest_laplace < lap) & d.C_at_B_max.between(cmin, cmax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('breakers')
    ap.add_argument('--controls', default=None)
    ap.add_argument('--out', default='fig_field_Bmax')
    ap.add_argument('--b-th', type=float, default=0.85)
    a = ap.parse_args()
    figstyle.use()
    C = figstyle.COLORS
    d = pd.read_csv(a.breakers)
    d['qc'] = qc(d)
    ctrl = None
    if a.controls:
        ctrl = pd.read_csv(a.controls)
        ctrl['qc'] = qc(ctrl)

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.4), gridspec_kw={'width_ratios': [1.15, 1, 1]})
    # (a) B_max against gamma at onset
    for typ in ('spilling', 'plunging'):
        s = d.type == typ
        m = figstyle.MARKERS[typ]
        ax[0].scatter(d.gamma_onset[s & d.qc], d.B_max[s & d.qc], c=C[typ], marker=m, s=34, lw=0,
                      label=f'{typ}')
        ax[0].scatter(d.gamma_onset[s & ~d.qc], d.B_max[s & ~d.qc], facecolors='none', edgecolors=C[typ],
                      marker=m, s=34, lw=1.0, label=f'{typ}, QC fail')
    ax[0].axhline(a.b_th, ls='--', c='k', lw=1.4)
    ax[0].set_xlabel(r'$\gamma = H/h$ at onset')
    ax[0].set_ylabel(r'$B_{\max}$ along the transect')
    ax[0].legend(loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=2, columnspacing=0.8, handletextpad=0.3,
                 borderaxespad=0.0)

    # (b) distribution of B_max
    bins = np.arange(0.0, 1.1, 0.05)
    data, cols, labs = [], [], []
    for typ in ('spilling', 'plunging'):
        data.append(d.B_max[(d.type == typ) & d.qc]); cols.append(C[typ]); labs.append(typ)
    ax[1].hist(data, bins=bins, stacked=True, color=cols, label=labs, edgecolor='white', lw=0.6)
    if ctrl is not None:
        ax[1].hist(ctrl.B_max[ctrl.qc], bins=bins, histtype='step', color='k', lw=1.8, label='non-breaking')
    ax[1].axvline(a.b_th, ls='--', c='k', lw=1.4)
    ax[1].set_xlabel(r'$B_{\max}$')
    ax[1].set_ylabel('number of waves')
    ax[1].set_xlim(0.1, 1.0)
    ax[1].legend(loc='upper left', bbox_to_anchor=(0.0, 0.88))

    # (c) lead of B_max before onset
    ax[2].hist(d.lead_B_max_T[d.qc], bins=np.arange(0, 0.95, 0.05), color='0.55', edgecolor='white', lw=0.6)
    ax[2].set_xlabel(r'$\Delta t_{\max}/T$')
    ax[2].set_ylabel('number of breakers')
    ax[2].set_ylim(0, ax[2].get_ylim()[1] * 1.15)
    ax[1].set_ylim(0, ax[1].get_ylim()[1] * 1.1)
    figstyle.panel_labels(ax)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(a.out + '.pdf')
    fig.savefig(a.out + '.png', dpi=150)
    n = d.qc.sum()
    print(f'{n} of {len(d)} breakers pass QC; figure written to {a.out}.pdf/.png')
    for typ in ('spilling', 'plunging'):
        b = d.B_max[(d.type == typ) & d.qc]
        print(f'  {typ:>9}: n={b.size:3d} median {b.median():.3f} IQR {b.quantile(.25):.3f}-{b.quantile(.75):.3f} '
              f'max {b.max():.3f}')


if __name__ == '__main__':
    main()
