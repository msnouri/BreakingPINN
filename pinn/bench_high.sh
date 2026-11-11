#!/usr/bin/env bash
# Steady-wave benchmark near the breaking threshold: does the PINN recover B = 0.75-0.88?
# Same training as the field runs (bench_steady.py --field-config 1, the default), lidar-like gaps,
# a window of 0.5 T (the median record before onset in the field is 0.47 T).
# Exact B (Fenton, N = 48): H/d 0.50 -> 0.435, 0.70 -> 0.750, 0.717 -> 0.815, 0.723 -> 0.849, 0.727 -> 0.880
# About 3 min per case on one GPU; results in bench_high/*.json, console output in bench_high/*.log.
set -u
mkdir -p bench_high
run () {   # name, then arguments for bench_steady.py
    name=$1; shift
    if [ -f "bench_high/$name.json" ]; then echo "$name: done already"; return; fi
    echo "$name: $*"
    python bench_steady.py "$@" --window 0.5 --out "bench_high/$name.json" > "bench_high/$name.log" 2>&1 \
        && python -c "import json; r = json.load(open('bench_high/$name.json')); print(f\"  B exact {r['B_exact']:.3f}  PINN median {r['B_pinn_median']:.3f}  error {r['B_pinn_median'] - r['B_exact']:+.3f}  (u {100 * (r['u_crest_pinn_median'] / r['u_crest_exact'] - 1):+.1f}%, C {100 * (r['C_pinn_median'] / r['c_exact'] - 1):+.1f}%)\")" \
        || echo "  FAILED, see bench_high/$name.log"
}
run Hd0500_gaps   --Hd 0.500 --gaps 1
run Hd0700_gaps   --Hd 0.700 --gaps 1
run Hd0717_gaps   --Hd 0.717 --gaps 1
run Hd0723_gaps   --Hd 0.723 --gaps 1
run Hd0727_gaps   --Hd 0.727 --gaps 1
run Hd0723_full   --Hd 0.723 --gaps 0
run Hd0723_sharp  --Hd 0.723 --gaps 1 --sigma-eta-x 5
run Hd0723_tight  --Hd 0.723 --gaps 1 --tols 0.005 0.01 0.02 0.1 0.01
