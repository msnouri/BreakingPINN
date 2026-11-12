"""Reconstruct the crest kinematics of one Duck 2016 wave with the potential-flow PINN.

    python run_field_wave.py carini_pinn_input.mat --id 915 --steps 8000
    python run_field_wave.py carini_pinn_input.mat --best plunging     # longest record before onset

Training data: the measured surface (gap fills dropped when carini_pinn_mask.mat is present) from
the first scan to the onset scan (non-breaking waves: the whole track). Bed: z = -hMSL(x), in the
datum of the surface. Output: B(t) = u / C at the reconstructed crest for every scan, written to
<id>_pinn.json, and a figure <id>_pinn.png if matplotlib is available.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pinn_potential as pp                                              # noqa: E402
from carini_pinn_load import load, training_points, scans_before_onset   # noqa: E402


def pick(waves, wave_id=None, best=None):
    if wave_id is not None:
        sel = [w for w in waves if int(w['id']) == wave_id]
        if not sel:
            raise SystemExit(f'wave {wave_id} is not in the export')
        return sel[0]
    names = {'spilling': 2, 'plunging': 3}
    cand = [w for w in waves if w['type'] == names.get(best, 3) and w['retained']]
    return max(cand, key=lambda w: w['tPre_T'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('export', nargs='?', default='carini_pinn_input.mat')
    ap.add_argument('--id', type=int, default=None)
    ap.add_argument('--best', default='plunging', choices=['spilling', 'plunging'])
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--include-onset', type=int, default=1, help='train on the onset scan too (1) or stop before it (0)')
    ap.add_argument('--keep-gap-fills', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--lbfgs', type=int, default=800, help='L-BFGS iterations per round')
    ap.add_argument('--lbfgs-rounds', type=int, default=4, help='L-BFGS rounds, each on a fresh batch')
    ap.add_argument('--lbfgs-batch', type=float, default=4.0)
    ap.add_argument('--crest-focus', type=float, default=0.5, help='share of physics points drawn near the crest')
    ap.add_argument('--pad-after', type=int, default=2,
                    help='scans after onset kept for training only, so onset is not at the time edge of the fit')
    ap.add_argument('--c-window', type=int, default=7, help='scans in the straight-line fit for C')
    ap.add_argument('--edge', type=float, default=4.0, help='crest closer than this to the lidar window edge is flagged (m)')
    ap.add_argument('--tol-data', type=float, default=0.01, help='acceptable misfit to the lidar surface (m)')
    ap.add_argument('--b-th', type=float, default=0.85)
    ap.add_argument('--outdir', default='.', help='folder for <id>_pinn.json and <id>_pinn.png')
    ap.add_argument('--sigma-eta-x', type=float, default=3.0, help='Fourier-feature spread in x for eta (higher = sharper crests)')
    ap.add_argument('--scan-range', type=int, nargs=2, default=None, metavar=('FIRST', 'LAST'),
                    help='use only scans FIRST..LAST (1-based, inclusive) for training and B; for non-breaking waves, '
                         'to match the record lengths of the breakers')
    a = ap.parse_args()

    waves, meta = load(a.export, retained_only=False)
    w = pick(waves, a.id, a.best)
    wid = int(w['id'])
    nt = w['t'].size
    if np.isfinite(w['onset']):
        last = int(w['onset']) if a.include_onset else int(w['onset']) - 1
    else:
        last = nt
    last_fit = min(nt, last + a.pad_after) if np.isfinite(w['onset']) else last
    first = 0
    if a.scan_range is not None:
        first = max(a.scan_range[0] - 1, 0)
        last = min(a.scan_range[1], last)
        last_fit = min(last_fit, a.scan_range[1])
    cols = np.zeros(nt, bool)
    cols[first:last_fit] = True
    eta = w['eta']
    keep = np.isfinite(eta) & cols[np.newaxis, :]
    if not a.keep_gap_fills:
        keep &= ~w['gapFill']
    X, T = np.meshgrid(w['x'], w['t'], indexing='ij')
    x_obs, t_obs, e_obs = X[keep], T[keep], eta[keep]
    print(f"wave {wid} ({w['typeName']}, {w['day']}): {last_fit} scans fitted of {nt}, B evaluated on {last}, onset at scan "
          f"{w['onset']:.0f}; {x_obs.size} surface points ({keep.sum() / max(np.isfinite(eta[:, :last_fit]).sum(), 1):.2f} "
          f"of the finite points kept); depth {np.nanmin(w['hMSL']):.2f}-{np.nanmax(w['hMSL']):.2f} m")

    ok = np.isfinite(w['hMSL'])
    cfg = pp.Config(steps_eta=a.steps // 5, steps_phi=2 * a.steps // 5, steps_joint=2 * a.steps // 5, seed=a.seed,
                    log_every=max(a.steps // 10, 1), steps_lbfgs=a.lbfgs, lbfgs_rounds=a.lbfgs_rounds, lbfgs_batch=a.lbfgs_batch,
                    crest_focus=a.crest_focus, tol_data=a.tol_data,
                    sigma_eta=(a.sigma_eta_x, 1.0))
    model = pp.fit(x_obs, t_obs, e_obs, w['x'][ok], w['hMSL'][ok], cfg, crest_track=(w['t'], w['xc']))

    t_fit = w['t'][first:last_fit]
    kf = pp.crest_kinematics(model, t_fit, w['xc'][first:last_fit], c_window=a.c_window)
    k = {q: v[:last - first] for q, v in kf.items()}
    t_eval = w['t'][first:last]
    # flag scans whose crest is near the lidar window edge or whose C fit is one-sided in time
    hw = a.c_window // 2
    near_edge = (np.abs(k['xc'] - w['x'].min()) < a.edge) | (np.abs(k['xc'] - w['x'].max()) < a.edge)
    one_sided = (np.arange(last - first) < hw) | (np.arange(last - first) > last_fit - first - 1 - hw)
    good = ~(near_edge | one_sided) & np.isfinite(k['B'])
    res = {
        'id': wid, 'type': w['typeName'], 'day': w['day'], 'onset_scan': w['onset'], 'scans_used': last,
        't': t_eval.tolist(), 'x_crest': k['xc'].tolist(), 'eta_crest': k['eta_c'].tolist(),
        'u_crest': k['u'].tolist(), 'C': k['C'].tolist(), 'B': k['B'].tolist(),
        'C_tracked': w['C'][first:last].tolist(), 'x_crest_tracked': w['xc'][first:last].tolist(),
        'scan_range': [first + 1, last],
        'h_onset': float(w['hOnset']), 'T': float(w['T']), 'steps': a.steps,
        'final_losses': model.history[-1, 2:].tolist(), 'final_losses_crest': model.final_crest,
        'good': good.tolist(), 'scans_fitted': last_fit - first, 'c_window': a.c_window,
    }
    Bg = np.where(good, k['B'], np.nan)
    if np.isfinite(Bg).any():
        im = int(np.nanargmax(Bg))
        res.update({'B_max': float(Bg[im]), 'scan_B_max': first + im + 1, 'u_at_B_max': float(k['u'][im]),
                    'C_at_B_max': float(k['C'][im]), 'B_onset_window': float(np.nanmean(k['B'][-3:])),
                    'lead_B_max_s': float(t_eval[-1] - t_eval[im]),
                    'lead_B_max_T': float((t_eval[-1] - t_eval[im]) / float(w['T']))})
    above = np.flatnonzero((np.asarray(k['B']) >= a.b_th) & good)
    if above.size and np.isfinite(w['onset']):
        i0 = int(above[0])
        lead = float(w['t'][int(w['onset']) - 1] - t_eval[i0])
        res.update({'first_scan_above': first + i0 + 1, 'lead_s': lead, 'lead_T': lead / float(w['T'])})
    res.update({'tEpoch0': float(w['tEpoch0']), 'H_onset': float(w.get('HOnset', np.nan)),
                'gamma_onset': float(w.get('gammaOnset', np.nan)), 'tPre_T': float(w['tPre_T']),
                'args': vars(a)})
    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, f'{wid}_pinn.json')
    json.dump(res, open(out, 'w'), indent=1)
    print(f"\n{'scan':>5} {'t (s)':>7} {'x_c (m)':>8} {'C (m/s)':>8} {'u (m/s)':>8} {'B':>6}  flag")
    for i in range(last - first):
        fl_ = '' if good[i] else ('edge' if near_edge[i] else 'time-edge')
        print(f"{first + i + 1:5d} {t_eval[i]:7.2f} {k['xc'][i]:8.2f} {k['C'][i]:8.2f} {k['u'][i]:8.2f} {k['B'][i]:6.3f}  {fl_}")
    if 'first_scan_above' in res:
        print(f"B first reaches {a.b_th} at scan {res['first_scan_above']}, {res['lead_s']:.2f} s "
              f"({res['lead_T']:.2f} T) before the observed onset")
    else:
        print(f"B stays below {a.b_th} over the unflagged scans (largest {np.nanmax(np.where(good, k['B'], np.nan)):.3f})")
    if 'B_max' in res:
        print(f"B_max {res['B_max']:.3f} at scan {res['scan_B_max']}, {res['lead_B_max_s']:.2f} s "
              f"({res['lead_B_max_T']:.2f} T) before onset; mean B over the last 3 scans {res['B_onset_window']:.3f}")
    fl = res['final_losses']
    print(f"final losses (each relative to its tolerance): data {fl[0]:.2f}, laplace {fl[1]:.2f}, kinematic {fl[2]:.2f}, "
          f"dynamic {fl[3]:.2f}, bed {fl[4]:.2f}")
    if model.final_crest is not None:
        fc = model.final_crest
        print(f"  near the crest:                               data {fc[0]:.2f}, laplace {fc[1]:.2f}, "
              f"kinematic {fc[2]:.2f}, dynamic {fc[3]:.2f}, bed {fc[4]:.2f}")
    if max(fl[1:]) > 4:
        print("  WARNING: physics residuals on fresh points are well above tolerance; u (and B) are not reliable yet")
    print(f'results written to {out}')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        tc = np.linspace(w['t'][first], w['t'][last_fit - 1], 6)
        xs = np.linspace(w['x'][0], w['x'][-1], 800)
        for j, tt in enumerate(tc):
            ax[0].plot(xs, model.eta(xs, np.full(xs.size, tt)) + 0.4 * j, 'k-', lw=1)
            jj = int(np.argmin(np.abs(w['t'] - tt)))
            m = keep[:, jj]
            ax[0].plot(w['x'][m], eta[m, jj] + 0.4 * j, '.', ms=1.5, color='tab:blue')
        ax[0].set_xlabel('x (FRF, m)')
        ax[0].set_ylabel('eta + 0.4 m per time (m)')
        ax[0].invert_xaxis()
        ax[1].plot(t_eval, k['B'], '-', color='tab:red', lw=1)
        ax[1].plot(t_eval[good], k['B'][good], 'o', color='tab:red', label='used')
        ax[1].plot(t_eval[~good], k['B'][~good], 'o', mfc='none', color='tab:red', label='flagged (edge)')
        ax[1].axvline(t_eval[-1], ls=':', color='grey')
        ax[1].legend(loc='upper left', fontsize=8)
        ax[1].axhline(0.85, ls='--', color='k')
        ax[1].set_xlabel('t (s)')
        ax[1].set_ylabel('B = u / C')
        fig.tight_layout()
        fig.savefig(os.path.join(a.outdir, f'{wid}_pinn.png'), dpi=150)
        plt.close(fig)
        print(f"figure written to {os.path.join(a.outdir, f'{wid}_pinn.png')}")
    except ImportError:
        pass


if __name__ == '__main__':
    main()
