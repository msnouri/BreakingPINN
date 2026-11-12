#!/usr/bin/env bash
# What limits the crest velocity near the breaking threshold? Steady waves with exact B = 0.849 (H/d 0.723)
# and 0.750 (H/d 0.700), field training and lidar-like gaps, window 0.5 T, changing one thing at a time:
# measurement (noise, spacing, scan rate, window), and reconstruction (network size, training length,
# data tolerance, crest focus), plus two more seeds of the reference case.
# About 3-5 min per case (longer for the dense cases); about 2.5 h in total. Resumable: finished cases are skipped.
# Results in bench_sweep/*.json; table with:  python bench_table.py bench_sweep
set -u
D=bench_sweep
mkdir -p $D
run () {
    name=$1; shift
    if [ -f "$D/$name.json" ]; then echo "$name: done already"; return; fi
    echo "$name: $*"
    python bench_steady.py --gaps 1 --window 0.5 "$@" --out "$D/$name.json" > "$D/$name.log" 2>&1 \
        && python -c "import json; r = json.load(open('$D/$name.json')); print(f\"  B exact {r['B_exact']:.3f}  PINN {r['B_pinn_median']:.3f}  error {r['B_pinn_median'] - r['B_exact']:+.3f}\")" \
        || echo "  FAILED, see $D/$name.log"
}
for Hd in 0.723 0.700; do
    h=${Hd/0./}
    run H${h}_ref          --Hd $Hd
    run H${h}_seed1        --Hd $Hd --seed 1
    run H${h}_noise0       --Hd $Hd --noise 0
    run H${h}_noise10mm    --Hd $Hd --noise 0.01
    run H${h}_noise20mm    --Hd $Hd --noise 0.02
    run H${h}_dx2cm        --Hd $Hd --dx 0.02
    run H${h}_dx10cm       --Hd $Hd --dx 0.10
    run H${h}_rate10       --Hd $Hd --rate 10
    run H${h}_rate2.5      --Hd $Hd --rate 2.585
    run H${h}_tol2.5mm     --Hd $Hd --tols 0.0025 0.01 0.02 0.1 0.01
    run H${h}_tolall       --Hd $Hd --tols 0.005 0.005 0.01 0.05 0.005
    run H${h}_width128     --Hd $Hd --width 128
    run H${h}_long         --Hd $Hd --steps 16000 --lbfgs-rounds 8
    run H${h}_focus0.8     --Hd $Hd --focus 0.8
done
run H723_window1T  --Hd 0.723 --window 1.0
run H723_seed2     --Hd 0.723 --seed 2
run H723_tol2.5mm_long --Hd 0.723 --tols 0.0025 0.01 0.02 0.1 0.01 --steps 16000 --lbfgs-rounds 8
