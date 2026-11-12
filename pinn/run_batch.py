"""Run the field-wave PINN over many Duck 2016 waves, one after another, and collect a summary table.

    python run_batch.py carini_pinn_input.mat --set breakers                 # 165 retained breakers
    python run_batch.py carini_pinn_input.mat --set controls --n-controls 165
    python run_batch.py carini_pinn_input.mat --set breakers --limit 5      # quick trial
    python run_batch.py carini_pinn_input.mat --summary-only                # rebuild the table only

Each wave runs run_field_wave.py in its own process (a crash on one wave does not stop the batch).
Results go to <outdir>/<id>_pinn.json/.png and the console output to <outdir>/logs/<id>.log.
Waves that already have a JSON are skipped, so an interrupted batch resumes where it stopped.
Controls: retained non-breaking waves, the steepest first (largest H/h over the record), so they
cover the same H/h range as the breakers.
Extra arguments after -- go to run_field_wave.py, e.g.  -- --tol-data 0.005
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from carini_pinn_load import load   # noqa: E402


def max_gamma(w):
    """Largest H/h along the record of a wave (depth at the crest position)."""
    H = np.asarray(w['H'], float).ravel()
    xc = np.asarray(w['xc'], float).ravel()
    ok = np.isfinite(w['hMSL'])
    h = np.interp(xc, w['x'][ok], w['hMSL'][ok], left=np.nan, right=np.nan)
    g = H / h
    return float(np.nanmax(g)) if np.isfinite(g).any() else np.nan


def pick_waves(waves, which, n_controls):
    if which == 'breakers':
        sel = [w for w in waves if w['retained'] and w['type'] in (2, 3)]
        sel.sort(key=lambda w: -w['tPre_T'])
    else:
        sel = [w for w in waves if w['retained'] and w['type'] in (0,)]
        sel.sort(key=lambda w: -np.nan_to_num(max_gamma(w), nan=-1))
        sel = sel[:n_controls]
    return [int(w['id']) for w in sel]


def summarize(outdir):
    rows = []
    for f in sorted(glob.glob(os.path.join(outdir, '*_pinn.json'))):
        r = json.load(open(f))
        fl = r.get('final_losses') or [np.nan] * 5
        fc = r.get('final_losses_crest') or [np.nan] * 5
        rows.append({
            'id': r['id'], 'type': r['type'], 'day': r['day'],
            'tEpoch0': r.get('tEpoch0'), 'utc': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(r['tEpoch0']))
            if r.get('tEpoch0') else '',
            'T': r.get('T'), 'h_onset': r.get('h_onset'), 'H_onset': r.get('H_onset'),
            'gamma_onset': r.get('gamma_onset'), 'tPre_T': r.get('tPre_T'),
            'B_max': r.get('B_max'), 'scan_B_max': r.get('scan_B_max'),
            'lead_B_max_s': r.get('lead_B_max_s'), 'lead_B_max_T': r.get('lead_B_max_T'),
            'B_onset_window': r.get('B_onset_window'),
            'u_at_B_max': r.get('u_at_B_max'), 'C_at_B_max': r.get('C_at_B_max'),
            'first_scan_above': r.get('first_scan_above'), 'lead_s': r.get('lead_s'), 'lead_T': r.get('lead_T'),
            'res_data': fl[0], 'res_laplace': fl[1], 'res_kin': fl[2], 'res_dyn': fl[3], 'res_bed': fl[4],
            'crest_laplace': fc[1], 'crest_kin': fc[2], 'crest_dyn': fc[3],
            'onset_at_record_end': r.get('scans_fitted') == r.get('scans_used') and r.get('onset_scan') is not None,
        })
    if not rows:
        print('no results yet')
        return
    out = os.path.join(outdir, 'summary.csv')
    with open(out, 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f'{len(rows)} waves in {out}')
    for typ in ('plunging', 'spilling', 'nonbreaking'):
        b = np.array([r['B_max'] for r in rows if r['type'] == typ and r['B_max'] is not None], float)
        if b.size:
            print(f'  {typ:>12}: n={b.size:4d}  B_max median {np.median(b):.3f}  '
                  f'IQR {np.percentile(b, 25):.3f}-{np.percentile(b, 75):.3f}  share >= 0.85: {np.mean(b >= 0.85):.2f}')


def main():
    argv = sys.argv[1:]
    extra = []
    if '--' in argv:
        i = argv.index('--')
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument('export', nargs='?', default='carini_pinn_input.mat')
    ap.add_argument('--set', default='breakers', choices=['breakers', 'controls'])
    ap.add_argument('--n-controls', type=int, default=165)
    ap.add_argument('--outdir', default=None, help='default runs/batch_<set>')
    ap.add_argument('--limit', type=int, default=None, help='only the first N waves (trial)')
    ap.add_argument('--ids', type=int, nargs='*', default=None, help='explicit wave ids instead of --set')
    ap.add_argument('--summary-only', action='store_true')
    ap.add_argument('--match-lengths', default=None, metavar='BREAKER_DIR',
                    help='for controls: train each on its last N scans only, with N drawn (fixed seed) from the record '
                         'lengths up to onset of the breakers whose results are in BREAKER_DIR')
    a = ap.parse_args(argv)
    outdir = a.outdir or os.path.join('runs', f'batch_{a.set}')
    os.makedirs(os.path.join(outdir, 'logs'), exist_ok=True)
    if a.summary_only:
        summarize(outdir)
        return

    ranges = {}
    waves, _ = load(a.export, retained_only=False)
    ids = a.ids if a.ids else pick_waves(waves, a.set, a.n_controls)
    if a.match_lengths:
        lens = [int(json.load(open(f))['onset_scan']) for f in glob.glob(os.path.join(a.match_lengths, '*_pinn.json'))]
        if not lens:
            sys.exit(f'no breaker results in {a.match_lengths}')
        rng = np.random.default_rng(12345)
        nscan = {int(w['id']): int(w['t'].size) for w in waves}
        for wid in ids:
            n = min(int(rng.choice(lens)), nscan[wid])
            ranges[wid] = (nscan[wid] - n + 1, nscan[wid])
        print(f'record lengths matched to {len(lens)} breakers (median {int(np.median(lens))} scans)')
    del waves
    if a.limit:
        ids = ids[:a.limit]
    todo = [i for i in ids if not os.path.exists(os.path.join(outdir, f'{i}_pinn.json'))]
    print(f'{len(ids)} waves in the set, {len(ids) - len(todo)} already done, {len(todo)} to run -> {outdir}')
    tic = time.time()
    for n, wid in enumerate(todo, 1):
        cmd = [sys.executable, os.path.join(HERE, 'run_field_wave.py'), a.export, '--id', str(wid),
               '--outdir', outdir] + extra
        if wid in ranges:
            cmd += ['--scan-range', str(ranges[wid][0]), str(ranges[wid][1])]
        t0 = time.time()
        with open(os.path.join(outdir, 'logs', f'{wid}.log'), 'w') as log:
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        eta_h = (time.time() - tic) / n * (len(todo) - n) / 3600
        msg = 'ok' if rc == 0 else f'FAILED (exit {rc}, see logs/{wid}.log)'
        bm = ''
        jf = os.path.join(outdir, f'{wid}_pinn.json')
        if rc == 0 and os.path.exists(jf):
            r = json.load(open(jf))
            if r.get('B_max') is not None:
                bm = f"  B_max {r['B_max']:.3f}"
        print(f'[{n}/{len(todo)}] wave {wid}: {msg} in {dt:.0f} s{bm}   (about {eta_h:.1f} h left)', flush=True)
    summarize(outdir)


if __name__ == '__main__':
    main()
