"""Paper figures for the field test: example B(t), and breakers against non-breaking controls.

    python plot_field_paper.py runs/batch_breakers runs/batch_controls [--tight-breakers runs/tight_breakers/summary.csv
                               --tight-controls runs/tight_controls/summary.csv] [--outdir figs]

Writes fig_field_examples.pdf (B(t) of two breakers) and fig_field_separation.pdf:
(a) ROC curves of B_max (all waves passing the quality check, and controls on windows of the breakers' lengths),
(b) B against the time before onset for the breakers, with the controls' distribution for reference,
(c) B_max from the default and the tight (5 mm) fit for the same waves (if the tight summaries are given).
Also prints the statistics quoted in the paper.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figstyle


def qc_ok(r, kin=5.0, lap=10.0, cmin=3.0, cmax=9.0):
    fc = r.get('final_losses_crest') or [np.nan] * 5
    c = r.get('C_at_B_max')
    return bool(fc[2] < kin and fc[1] < lap and c is not None and cmin < c < cmax)


def load(folder):
    out = {}
    for f in glob.glob(os.path.join(folder, '*_pinn.json')):
        r = json.load(open(f))
        if r.get('B_max') is None:
            continue
        g = np.array(r['good'], bool)
        B = np.array(r['B'], float)
        t = np.array(r['t'][:B.size], float)
        r['_B'], r['_t'], r['_good'] = B, t, g
        r['_qc'] = qc_ok(r)
        out[int(r['id'])] = r
    return out


def auc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return (a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean()


def roc(pos, neg):
    th = np.unique(np.concatenate([pos, neg]))[::-1]
    tpr = np.array([0.0] + [(pos >= x).mean() for x in th] + [1.0])
    fpr = np.array([0.0] + [(neg >= x).mean() for x in th] + [1.0])
    return fpr, tpr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('breakers')
    ap.add_argument('controls')
    ap.add_argument('--tight-breakers', default=None)
    ap.add_argument('--tight-controls', default=None)
    ap.add_argument('--examples', type=int, nargs=2, default=[915, 132])
    ap.add_argument('--outdir', default='.')
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    figstyle.use()
    rng = np.random.default_rng(a.seed)
    br_all, co_all = load(a.breakers), load(a.controls)
    br = [r for r in br_all.values() if r['_qc']]
    co = [r for r in co_all.values() if r['_qc']]
    Bb = np.array([r['B_max'] for r in br])
    Bc = np.array([r['B_max'] for r in co])

    # ---- statistics -------------------------------------------------------------------------------
    A = auc(Bb, Bc)
    boot = [auc(rng.choice(Bb, Bb.size), rng.choice(Bc, Bc.size)) for _ in range(2000)]
    nb = np.array([r['_good'].sum() for r in br])
    matched, aucs_m = [], []
    for rep in range(200):
        m = []
        for r in co:
            Bg = r['_B'][r['_good']]
            n = min(int(rng.choice(nb)), Bg.size)
            s = int(rng.integers(0, Bg.size - n + 1))
            m.append(np.nanmax(Bg[s:s + n]))
        aucs_m.append(auc(Bb, m))
        if rep == 0:
            matched = np.array(m)
    print(f'QC: {len(br)} of {len(br_all)} breakers, {len(co)} of {len(co_all)} controls')
    print(f'B_max median: breakers {np.median(Bb):.3f}, controls {np.median(Bc):.3f}')
    print(f'AUC {A:.3f} (95% {np.percentile(boot, 2.5):.3f}-{np.percentile(boot, 97.5):.3f}); '
          f'matched windows {np.mean(aucs_m):.3f} ({np.percentile(aucs_m, 2.5):.3f}-{np.percentile(aucs_m, 97.5):.3f})')

    # ---- figure: examples -------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.0), sharey=True)
    for k, wid in enumerate(a.examples):
        r = br_all.get(wid)
        if r is None:
            continue
        typ = r['type']
        t = (r['_t'] - r['_t'][-1]) / r['T']           # time relative to onset, in periods
        g = r['_good']
        ax[k].plot(t, r['_B'], '-', color=figstyle.COLORS[typ], lw=1.4)
        ax[k].plot(t[g], r['_B'][g], 'o', color=figstyle.COLORS[typ], ms=6)
        ax[k].plot(t[~g], r['_B'][~g], 'o', mfc='white', color=figstyle.COLORS[typ], ms=6)
        ax[k].axhline(0.85, ls='--', c='k', lw=1.3)
        ax[k].axvline(0.0, ls=':', c='0.4', lw=1.3)
        ax[k].set_xlabel(r'$(t - t_{\mathrm{onset}})/T$')
        ax[k].set_title(f'wave {wid}, {typ}', fontsize=16)
    ax[0].set_ylabel(r'$B$ along the transect')
    ax[0].set_ylim(0, 1.0)
    figstyle.panel_labels(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(a.outdir, 'fig_field_examples.pdf'))
    fig.savefig(os.path.join(a.outdir, 'fig_field_examples.png'), dpi=150)

    # ---- figure: separation -----------------------------------------------------------------------
    tight = a.tight_breakers and a.tight_controls
    fig, ax = plt.subplots(1, 3 if tight else 2, figsize=(18 if tight else 12.5, 5.4))
    f, tp = roc(Bb, Bc)
    ax[0].plot(f, tp, color='k', lw=2.0, label=f'all records (AUC {A:.2f})')
    f, tp = roc(Bb, matched)
    ax[0].plot(f, tp, color='0.5', lw=2.0, ls='--', label=f'matched windows ({np.mean(aucs_m):.2f})')
    ax[0].plot([0, 1], [0, 1], ':', color='0.6', lw=1.2)
    ax[0].set_xlabel('false-positive rate')
    ax[0].set_ylabel('true-positive rate')
    ax[0].set_aspect('equal')
    ax[0].legend(loc='lower right')

    # B against the time before onset (breakers), controls for reference
    tb = np.concatenate([(r['_t'][-1] - r['_t'][r['_good']]) / r['T'] for r in br])
    bb = np.concatenate([r['_B'][r['_good']] for r in br])
    edges = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
    mid, med, q1, q3 = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (tb >= lo) & (tb < hi)
        if m.sum() > 10:
            mid.append(-(lo + hi) / 2)
            med.append(np.median(bb[m])); q1.append(np.percentile(bb[m], 25)); q3.append(np.percentile(bb[m], 75))
    bc_all = np.concatenate([r['_B'][r['_good']] for r in co])
    cmax = np.array([np.nanmax(r['_B'][r['_good']]) for r in co])
    ax[1].axhspan(np.percentile(bc_all, 25), np.percentile(bc_all, 75), color='0.85', lw=0,
                  label='non-breaking, all scans (IQR)')
    ax[1].axhline(np.median(cmax), color='0.35', ls='--', lw=1.6, label=r'non-breaking, median $B_{\max}$')
    ax[1].fill_between(mid, q1, q3, color=figstyle.COLORS['spilling'], alpha=0.25, lw=0)
    ax[1].plot(mid, med, 'o-', color=figstyle.COLORS['spilling'], label='breakers, median and IQR')
    ax[1].axhline(0.85, ls='--', c='k', lw=1.3)
    ax[1].set_xlabel(r'$(t - t_{\mathrm{onset}})/T$')
    ax[1].set_ylabel(r'$B$ along the transect')
    ax[1].set_ylim(0.2, 0.92)
    ax[1].legend(loc='upper left', bbox_to_anchor=(0.0, 0.88), fontsize=13)
    print('breakers, median B by time before onset:', [f'{m:.3f}' for m in med], ' at', [f'{x:.3f}' for x in mid])
    print(f'controls: median B over all scans {np.median(bc_all):.3f}, median B_max {np.median(cmax):.3f}')

    if tight:
        tb_ = pd.read_csv(a.tight_breakers)
        tc_ = pd.read_csv(a.tight_controls)
        for d, ref, typ in ((tb_, br_all, 'spilling'), (tc_, co_all, 'nonbreaking')):
            x = np.array([ref[i]['B_max'] for i in d.id if i in ref])
            y = np.array([b for i, b in zip(d.id, d.B_max) if i in ref])
            ok = (x < 1.0) & (y < 1.0)
            lab = 'breakers' if typ == 'spilling' else 'non-breaking'
            ax[2].scatter(x[ok], y[ok], s=36, color=figstyle.COLORS[typ], marker=figstyle.MARKERS[typ], label=lab)
            print(f'tight subset {lab}: n={ok.sum()} median default {np.median(x[ok]):.3f} tight {np.median(y[ok]):.3f}')
        ax[2].plot([0.2, 1.0], [0.2, 1.0], ':', color='0.5')
        ax[2].set_xlim(0.2, 1.0); ax[2].set_ylim(0.2, 1.0)
        ax[2].set_aspect('equal')
        ax[2].set_xlabel(r'$B_{\max}$, data tolerance 1 cm')
        ax[2].set_ylabel(r'$B_{\max}$, data tolerance 5 mm')
        ax[2].legend(loc='lower right')
    figstyle.panel_labels(ax)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(os.path.join(a.outdir, 'fig_field_separation.pdf'))
    fig.savefig(os.path.join(a.outdir, 'fig_field_separation.png'), dpi=150)
    print('figures written to', a.outdir)


if __name__ == '__main__':
    main()
