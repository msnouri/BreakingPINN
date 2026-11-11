"""Collect the benchmark runs (run_*.json) into bench_results.md."""
import glob
import json

import numpy as np

rows = []
for f in sorted(glob.glob('run_*.json')):
    r = json.load(open(f))
    rows.append((f, r))
lines = ['# Benchmark of the potential-flow PINN on exact steady waves', '',
         'Depth 3 m, wavelength 40 m; lidar-like samples: 40-m window at 5 cm, 5.17 scans per second, '
         '5 mm noise; the window lasts 0.3 T (12 scans). Errors relative to the exact solution.', '',
         '| run | H (m) | B exact | coverage | depth error (m) | stages | C error | u_crest error | B median | B range over scans | B error | surface u RMS / C |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|']
for f, r in rows:
    B = np.array(r['B_pinn_all'])
    stages = f"Adam {r['steps']}" + (f" + L-BFGS {r['lbfgs']}" if r.get('lbfgs') else '')
    if r.get('tols', [0.01, 0.05])[1] > 0.02:
        stages += ', loose tolerances'
    if r.get('sigma_phi', [1, 0, 0.5])[0] > 1:
        stages += ', sharper phi features'
    if r.get('focus', 0) > 0:
        stages += ', crest focus'
    if r.get('train_on_fill'):
        stages += ', trained on filled gaps'
    lines.append(f"| {f[4:-5]} | {r['H']} | {r['B_exact']:.4f} | {(0.54 if r.get('train_on_fill') else r['measured_share']):.2f} | {r.get('depth_error', 0):.2f} | {stages} | "
                 f"{100 * (r['C_pinn_median'] / r['c_exact'] - 1):+.1f}% | {100 * (r['u_crest_pinn_median'] / r['u_crest_exact'] - 1):+.1f}% | "
                 f"{r['B_pinn_median']:.4f} | {B.min():.3f}-{B.max():.3f} | {r['B_pinn_median'] - r['B_exact']:+.4f} | "
                 f"{100 * r['surface_u_rms_over_c']:.1f}% |")
open('bench_results.md', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
