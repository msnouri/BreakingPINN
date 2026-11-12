"""Table of steady-wave benchmark runs:  python bench_table.py bench_sweep [bench_high ...]"""
import glob
import json
import os
import sys

rows = []
for folder in sys.argv[1:] or ['bench_sweep']:
    for f in sorted(glob.glob(os.path.join(folder, '*.json'))):
        r = json.load(open(f))
        fc = r.get('final_losses_crest') or [float('nan')] * 5
        rows.append((os.path.basename(f)[:-5], r['B_exact'], r['B_pinn_median'], r['B_pinn_median'] - r['B_exact'],
                     100 * (r['u_crest_pinn_median'] / r['u_crest_exact'] - 1), 100 * (r['C_pinn_median'] / r['c_exact'] - 1),
                     fc[2], fc[1], r.get('train_s', float('nan')) / 60))
print(f"{'case':24s} {'B exact':>7s} {'PINN':>6s} {'error':>7s} {'u %':>6s} {'C %':>6s} {'kin':>6s} {'lapl':>6s} {'min':>5s}")
for r in rows:
    print(f"{r[0]:24s} {r[1]:7.3f} {r[2]:6.3f} {r[3]:+7.3f} {r[4]:+6.1f} {r[5]:+6.1f} {r[6]:6.2f} {r[7]:6.2f} {r[8]:5.1f}")
