"""Benchmark of the potential-flow PINN on a steady wave of finite height (exact solution from
fenton_steady.py), sampled like the Duck lidar: a 40-m window at 5 cm, 5.17 scans per second,
5 mm of noise, and optionally clustered gaps in the returns.

    python bench_steady.py --H 1.5 --window 0.3 --gaps 0 --steps 6000
"""
import argparse
import sys
import json
import time

import numpy as np

import fenton_steady as fs
import pinn_potential as pp


def clustered_mask(n, dx, rng, patch_mean=0.25, gap_mean=0.12, gap_min=0.02):
    """Measured points along a profile: patches of returns separated by gaps (lengths in m)."""
    keep = np.zeros(n, bool)
    x = 0.0
    L = n * dx
    while x < L:
        p = rng.exponential(patch_mean)
        keep[int(x / dx):min(n, int((x + p) / dx) + 1)] = True
        x += p + gap_min + rng.exponential(gap_mean)
    return keep


def cubic_fill(z, measured):
    """Fill the unmeasured points so that the fourth difference vanishes at each of them (a cubic
    across every gap, as in the del^4 option of inpaint_nans), between the first and last measured
    points; second difference near the ends of the array."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    z = np.array(z, float)
    idx = np.flatnonzero(measured)
    if idx.size < 2:
        return z
    gap = np.flatnonzero(~measured[idx[0]:idx[-1] + 1]) + idx[0]
    if gap.size == 0:
        return z
    n = z.size
    pos = -np.ones(n, int)
    pos[gap] = np.arange(gap.size)
    rows, cols, vals = [], [], []
    rhs = np.zeros(gap.size)
    for q, i in enumerate(gap):
        st, off = ([1, -4, 6, -4, 1], range(-2, 3)) if 2 <= i <= n - 3 else ([1, -2, 1], range(-1, 2))
        for c, o in zip(st, off):
            j = i + o
            if j < 0 or j >= n:
                continue
            if pos[j] >= 0:
                rows.append(q); cols.append(pos[j]); vals.append(c)
            elif measured[j]:
                rhs[q] -= c * z[j]
    A = coo_matrix((vals, (rows, cols)), shape=(gap.size, gap.size)).tocsr()
    z[gap] = spsolve(A, rhs)
    return z


def run(H=1.5, window=0.3, gaps=False, steps=6000, seed=0, d=3.0, L=40.0, noise=0.005, x_start=8.0, verbose=True,
        focus=0.0, x64=False, width=64, sigma_phi=(1.0, 0.0, 0.5), tols=(0.01, 0.01, 0.02, 0.1, 0.01), lbfgs=2000, depth_error=0.0, gap_params=(0.25, 0.12), train_on_fill=False,
        lbfgs_rounds=1, lbfgs_batch=2.0, sigma_eta_x=3.0, c_window=5, n_fourier=48):
    rng = np.random.default_rng(seed)
    wave = fs.solve(d, H, L, N=n_fourier, steps=60)
    T = L / wave.c
    x = np.arange(0.0, 40.0 + 1e-9, 0.05)
    n_scan = int(round(window * T * 5.17)) + 1
    t = np.arange(n_scan) / 5.17
    X, Tt = np.meshgrid(x, t, indexing='ij')
    eta_true = wave.eta(X - x_start, Tt)
    obs = eta_true + rng.normal(0.0, noise, eta_true.shape)
    meas = np.ones_like(obs, bool)
    if gaps:
        for j in range(n_scan):
            meas[:, j] = clustered_mask(x.size, 0.05, rng, patch_mean=gap_params[0], gap_mean=gap_params[1])
    cfg = pp.Config(steps_eta=steps // 5, steps_phi=2 * steps // 5, steps_joint=2 * steps // 5, seed=seed,
                    log_every=max(steps // 10, 1), crest_focus=focus, x64=x64, width_eta=width, width_phi=width,
                    sigma_phi=tuple(sigma_phi), tol_data=tols[0], tol_laplace=tols[1], tol_kin=tols[2],
                    tol_dyn=tols[3], tol_bed=tols[4], steps_lbfgs=lbfgs,
                    lbfgs_rounds=lbfgs_rounds, lbfgs_batch=lbfgs_batch, sigma_eta=(sigma_eta_x, 1.0))
    # crest track from the observations: highest observed point of each scan
    obs_m = np.where(meas, obs, -np.inf)
    track = (t, x[np.argmax(obs_m, axis=0)])
    tic = time.time()
    d_pinn = d + depth_error                       # depth given to the PINN (error in the bathymetry or datum)
    if train_on_fill:                              # gaps filled like the gridded field data, all points used
        filled = np.array([cubic_fill(obs[:, j], meas[:, j]) for j in range(n_scan)]).T
        span = np.zeros_like(meas)
        for j in range(n_scan):
            f = np.flatnonzero(meas[:, j])
            span[f[0]:f[-1] + 1, j] = True
        obs, meas = filled, span
    model = pp.fit(X[meas], Tt[meas], obs[meas], np.array([-5.0, 45.0]), np.array([d_pinn, d_pinn]), cfg, verbose=verbose,
                   crest_track=track)
    train_s = time.time() - tic
    x_guess = x_start + wave.c * t
    k = pp.crest_kinematics(model, t, x_guess, c_window=c_window)
    # surface velocity over the window, against the exact solution at the exact surface
    xe = np.arange(2.0, 38.0, 0.5)
    Xe, Te = np.meshgrid(xe, t, indexing='ij')
    ze = wave.eta(Xe - x_start, Te)
    u_ex, w_ex = wave.velocity(Xe - x_start, ze, Te)
    u_pi, w_pi = model.velocity(Xe, ze, Te)
    inner = (k['xc'] > 2) & (k['xc'] < 38)
    # velocity profile under the crest at mid-window, PINN and exact
    jm = n_scan // 2
    xcm = x_start + wave.c * t[jm]
    zp = np.linspace(-d, float(wave.eta(0, 0)), 25)
    u_ex_p, _ = wave.velocity(np.full(zp.size, xcm - x_start), zp, np.full(zp.size, t[jm]))
    u_pi_p, _ = model.velocity(np.full(zp.size, xcm), zp, np.full(zp.size, t[jm]))
    out = {
        'H': H, 'd': d, 'L': L, 'T': T, 'window_T': window, 'n_scan': n_scan, 'gaps': bool(gaps), 'focus': focus, 'x64': x64, 'width': width, 'sigma_phi': list(sigma_phi), 'tols': list(tols), 'lbfgs': lbfgs, 'depth_error': depth_error, 'gap_params': list(gap_params), 'train_on_fill': bool(train_on_fill),
        'measured_share': float(meas.mean()), 'steps': steps, 'train_s': train_s,
        'c_exact': wave.c, 'u_crest_exact': wave.u_crest, 'B_exact': wave.B,
        'crest_eta_exact': float(wave.eta(0, 0)),
        'C_pinn_median': float(np.nanmedian(k['C'][inner])), 'u_crest_pinn_median': float(np.nanmedian(k['u'][inner])),
        'B_pinn_median': float(np.nanmedian(k['B'][inner])), 'B_pinn_all': k['B'].tolist(),
        'eta_crest_pinn_median': float(np.nanmedian(k['eta_c'][inner])),
        'surface_u_rms_over_c': float(np.sqrt(np.mean((u_pi - u_ex) ** 2)) / wave.c),
        'surface_w_rms_over_c': float(np.sqrt(np.mean((w_pi - w_ex) ** 2)) / wave.c),
        'eta_rms': float(np.sqrt(np.mean((model.eta(Xe, Te) - ze) ** 2))),
        'profile_z': zp.tolist(), 'profile_u_exact': u_ex_p.tolist(), 'profile_u_pinn': u_pi_p.tolist(),
        'final_losses': model.history[-1, 2:].tolist(),
        'final_losses_crest': getattr(model, 'final_crest', None),
        'lbfgs_rounds': lbfgs_rounds, 'lbfgs_batch': lbfgs_batch, 'sigma_eta_x': sigma_eta_x, 'c_window': c_window,
        'B_pinn_inner': [float(b) for b in np.asarray(k['B'])[inner]],
    }
    return out, model, wave


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--H', type=float, default=1.5)
    ap.add_argument('--Hd', type=float, default=None, help='wave height as a fraction of the depth (overrides --H)')
    ap.add_argument('--field-config', type=int, default=1,
                    help='1: same training as the field runs (8000 Adam steps, crest focus 0.5, 4 L-BFGS rounds of 800 on x4 batches, '
                         '7-scan C fit); 0: the original benchmark settings')
    ap.add_argument('--window', type=float, default=0.3)
    ap.add_argument('--gaps', type=int, default=0)
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--focus', type=float, default=0.0)
    ap.add_argument('--x64', type=int, default=0)
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--sigma-phi', type=float, nargs=3, default=[1.0, 0.0, 0.5])
    ap.add_argument('--tols', type=float, nargs=5, default=[0.01, 0.01, 0.02, 0.1, 0.01],
                    help='data (m), laplace (1/s), kinematic (m/s), dynamic (m2/s2), bed (m/s)')
    ap.add_argument('--lbfgs', type=int, default=2000)
    ap.add_argument('--depth-error', type=float, default=0.0)
    ap.add_argument('--gap-params', type=float, nargs=2, default=[0.25, 0.12], help='mean patch and gap lengths (m)')
    ap.add_argument('--train-on-fill', type=int, default=0, help='fill the gaps with a cubic and train on all points')
    ap.add_argument('--sigma-eta-x', type=float, default=3.0)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    if a.Hd is not None:
        a.H = a.Hd * 3.0
    extra = {}
    if a.field_config:
        extra = dict(lbfgs_rounds=4, lbfgs_batch=4.0, c_window=7)
        if '--steps' not in sys.argv:
            a.steps = 8000
        if '--focus' not in sys.argv:
            a.focus = 0.5
        if '--lbfgs' not in sys.argv:
            a.lbfgs = 800
    res, model, wave = run(a.H, a.window, bool(a.gaps), a.steps, a.seed, focus=a.focus, x64=bool(a.x64), width=a.width,
                           sigma_phi=a.sigma_phi, tols=a.tols, lbfgs=a.lbfgs,
                           depth_error=a.depth_error, gap_params=tuple(a.gap_params),
                           train_on_fill=bool(a.train_on_fill), sigma_eta_x=a.sigma_eta_x, **extra)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()
                      if k not in ('B_pinn_all', 'profile_z', 'profile_u_exact', 'profile_u_pinn')}, indent=1))
    print('u under the crest (z, exact, PINN):')
    for zz, ue, up in zip(res['profile_z'][::4], res['profile_u_exact'][::4], res['profile_u_pinn'][::4]):
        print(f'  {zz:6.2f} {ue:7.3f} {up:7.3f}')
    print('B at each scan:', np.round(res['B_pinn_all'], 3).tolist())
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)
