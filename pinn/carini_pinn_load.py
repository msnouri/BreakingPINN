"""Read the Duck 2016 waves exported by carini_export_pinn.m.

    from carini_pinn_load import load, training_points
    waves, meta = load('carini_pinn_input.mat')       # retained waves only
    w = waves[0]
    x, t, eta = training_points(w)                     # measured surface before onset

Each wave is a dict whose arrays keep their MATLAB shapes: eta and gapFill are (nX, nT),
one row per grid point x (FRF, m) and one column per scan t (s from the first scan).
eta is in the datum of the tracked crests and hMSL is the depth below mean sea level;
see meta for the definitions. If carini_pinn_mask.mat (written by carini_gap_study.m) is in
the same folder, its mask of points filled across gaps between lidar returns replaces gapFill,
and training_points drops those points. Run this file to print a summary of an export:

    python carini_pinn_load.py carini_pinn_input.mat
"""
import os
import sys

import numpy as np
from scipy.io import loadmat

TYPE_NAMES = {-1: 'unclassified', 0: 'nonbreaking', 1: 'steady state', 2: 'spilling', 3: 'plunging'}
VECTORS = ('t', 'x', 'hMSL', 'xc', 'zc', 'H', 'ztr', 'C')


def _to_dict(obj):
    return {name: getattr(obj, name) for name in obj._fieldnames}


def _load_mask(path):
    """{wave id: gapFill array} and the mask description from carini_pinn_mask.mat."""
    m = loadmat(path, squeeze_me=True, struct_as_record=False)
    masks = {int(item.id): np.asarray(item.gapFill, dtype=bool) for item in np.atleast_1d(m['mask'])}
    return masks, _to_dict(m['maskInfo'])


def load(path='carini_pinn_input.mat', retained_only=True, types=None, mask='auto'):
    """Return (waves, meta).

    retained_only: keep only the waves retained for the B test.
    types: wave types to keep, e.g. (2, 3) for spilling and plunging breakers, (0,) for controls.
    mask: 'auto' uses carini_pinn_mask.mat next to path if it exists; a path uses that file;
          None keeps the gapFill of the export.
    """
    m = loadmat(path, squeeze_me=True, struct_as_record=False)
    meta = _to_dict(m['meta'])
    if hasattr(meta.get('opts'), '_fieldnames'):
        meta['opts'] = _to_dict(meta['opts'])
    if mask == 'auto':
        mask = os.path.join(os.path.dirname(os.path.abspath(path)), 'carini_pinn_mask.mat')
        if not os.path.exists(mask):
            mask = None
    masks = {}
    if mask is not None:
        masks, meta['mask'] = _load_mask(mask)
    waves = []
    for item in np.atleast_1d(m['waves']):
        w = _to_dict(item)
        for key in VECTORS:
            w[key] = np.atleast_1d(np.asarray(w[key], dtype=float))
        nx, nt = w['x'].size, w['t'].size
        w['eta'] = np.asarray(w['eta'], dtype=float).reshape(nx, nt, order='F')
        w['gapFill'] = np.asarray(w['gapFill'], dtype=bool).reshape(nx, nt, order='F')
        if int(w['id']) in masks:
            w['gapFill'] = masks[int(w['id'])].reshape(nx, nt, order='F')
        w['type'] = int(w['type'])
        w['retained'] = bool(w['retained'])
        w['onset'] = float(w['onset'])
        if retained_only and not w['retained']:
            continue
        if types is not None and w['type'] not in types:
            continue
        waves.append(w)
    return waves, meta


def scans_before_onset(w):
    """0-based indices of the scans recorded before onset (all scans for non-breaking waves)."""
    nt = w['t'].size
    if np.isfinite(w['onset']):
        return np.arange(int(w['onset']) - 1)        # onset is a 1-based scan number
    return np.arange(nt)


def training_points(w, before_onset=True, drop_gap_fills=True):
    """Flattened (x, t, eta) of the measured free surface of wave w."""
    eta = w['eta']
    keep = np.isfinite(eta)
    if drop_gap_fills:
        keep &= ~w['gapFill']
    if before_onset:
        cols = np.zeros(eta.shape[1], dtype=bool)
        cols[scans_before_onset(w)] = True
        keep &= cols[np.newaxis, :]
    X, T = np.meshgrid(w['x'], w['t'], indexing='ij')
    return X[keep], T[keep], eta[keep]


def bed(w):
    """(x, z_bed): bed elevation relative to mean sea level along the grid."""
    return w['x'], -w['hMSL']


def summary(waves):
    """Counts of the exported waves by type, and how many are retained."""
    lines = []
    for code in sorted({w['type'] for w in waves}):
        sel = [w for w in waves if w['type'] == code]
        kept = sum(w['retained'] for w in sel)
        lines.append(f"{TYPE_NAMES.get(code, code):>12}: {len(sel):5d} exported, {kept:5d} retained")
    return '\n'.join(lines)


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'carini_pinn_input.mat'
    waves, meta = load(path, retained_only=False)
    print(f"{path}: {len(waves)} waves")
    print(summary(waves))
    brk = [w for w in waves if w['retained'] and w['type'] in (2, 3)]
    if brk:
        w = max(brk, key=lambda v: v['tPre_T'])
        x, t, eta = training_points(w)
        print(f"\nwave {int(w['id'])} ({w['typeName']}, {w['day']}): {w['eta'].shape[0]} grid points x "
              f"{w['eta'].shape[1]} scans, onset at scan {int(w['onset'])}, "
              f"{w['tPre_T']:.2f} T recorded before onset")
        print(f"  grid {w['x'][0]:.2f} to {w['x'][-1]:.2f} m (step {w['dx']:.3f} m); "
              f"{x.size} measured points before onset; "
              f"{w['gapFill'].mean():.3f} of the grid flagged as gap fills")
        print(f"  at onset: h = {w['hOnset']:.2f} m, H/h = {w['gammaOnset']:.2f}, C = {w['COnset']:.2f} m/s")
    if 'mask' in meta:
        print(f"\nmask: {meta['mask']['detector']} (F1 {meta['mask']['f1Crest']:.2f} near the crests in the example record)")
    print(f"eta: {meta['eta']}")
