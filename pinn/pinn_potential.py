"""Potential-flow PINN for the free surface and the crest kinematics of a single wave (PyTorch).

Unknown fields, in metres and seconds, with z upward from the datum of the observations:
    eta(x, t)      free-surface elevation
    phi(x, z, t)   velocity potential, u = phi_x, w = phi_z
Losses (each residual divided by an acceptable error, cfg.tol_*): misfit to the observed surface;
Laplace equation in the fluid, -h(x) < z < eta; kinematic and dynamic conditions at z = eta,
    eta_t + phi_x eta_x - phi_z = 0,    phi_t + (phi_x^2 + phi_z^2) / 2 + g eta = 0,
(the Bernoulli constant is absorbed into phi); no flow through the bed, phi_z + h_x phi_x = 0 at
z = -h(x). Separate networks for eta and phi, with random Fourier features of the scaled inputs
(spread set per input axis; none in z for phi). Training in four stages: (1) eta alone on the
observations; (2) phi alone on the physics, eta frozen; (3) both on all terms; (4) L-BFGS on all
terms with a fixed batch. Stage 4 is essential: without it the velocity field keeps a spurious
uniform current that biases B by 8-10 % (see README).

    model = fit(x_obs, t_obs, eta_obs, x_bed, h_bed, Config())
    k = crest_kinematics(model, t_eval, x_guess)     # crest position, speed C, fluid velocity, B = u/C

Works with Python >= 3.8 and PyTorch >= 1.10; uses a GPU if one is available.
"""
from dataclasses import dataclass
import time

import numpy as np
import torch

G = 9.81


@dataclass
class Config:
    width_eta: int = 64
    depth_eta: int = 3
    width_phi: int = 64
    depth_phi: int = 4
    n_ff: int = 32                       # Fourier features per network
    sigma_eta: tuple = (3.0, 1.0)        # spread of their frequencies for eta: (x, t), scaled inputs
    sigma_phi: tuple = (1.0, 0.0, 0.5)   # for phi: (x, z, t)
    steps_eta: int = 1600                # stage 1: eta on the observations
    steps_phi: int = 3200                # stage 2: phi on the physics, eta frozen
    steps_joint: int = 3200              # stage 3: both on all terms
    steps_lbfgs: int = 2000              # stage 4: L-BFGS iterations per round, each round on a fixed batch
    lbfgs_batch: float = 4.0             # its size, in multiples of the usual batch
    lbfgs_rounds: int = 1                # rounds, each on a freshly drawn batch (stops overfitting one batch)
    lr: float = 2e-3
    lr_joint: float = 5e-4
    lr_final: float = 2e-5
    n_data: int = 2000                   # points per batch (resampled every `resample` steps)
    n_interior: int = 2000
    n_surface: int = 1500
    n_bed: int = 500
    resample: int = 250
    tol_data: float = 0.01               # [m]       acceptable misfit to the observed surface
    tol_laplace: float = 0.01            # [1/s]     acceptable phi_xx + phi_zz
    tol_kin: float = 0.02                # [m/s]     acceptable residual of the kinematic condition
    tol_dyn: float = 0.1                 # [m^2/s^2] acceptable residual of the dynamic condition
    tol_bed: float = 0.01                # [m/s]     acceptable flow through the bed
    crest_focus: float = 0.0             # share of surface and interior points drawn near the crest track
    crest_halfwidth: float = 4.0         # [m]
    seed: int = 0
    log_every: int = 1000
    x64: bool = False
    device: str = ''                     # '' = cuda if available, else cpu


class FourierMLP(torch.nn.Module):
    """MLP (tanh) on [scaled inputs, sin and cos of random Fourier projections]."""

    def __init__(self, n_in, sigma, n_ff, width, depth, gen):
        super().__init__()
        B = torch.randn(n_ff, n_in, generator=gen) * torch.tensor(sigma, dtype=torch.float32)
        self.register_buffer('B', B)
        layers = []
        n = n_in + 2 * n_ff
        for _ in range(depth):
            lin = torch.nn.Linear(n, width)
            torch.nn.init.xavier_uniform_(lin.weight)
            torch.nn.init.zeros_(lin.bias)
            layers += [lin, torch.nn.Tanh()]
            n = width
        out = torch.nn.Linear(n, 1)
        torch.nn.init.xavier_uniform_(out.weight)
        torch.nn.init.zeros_(out.bias)
        layers.append(out)
        self.net = torch.nn.Sequential(*layers)

    def forward(self, v):                                # v: (n, n_in)
        a = 2 * np.pi * v @ self.B.T
        return self.net(torch.cat([v, torch.sin(a), torch.cos(a)], dim=1)).squeeze(1)


class Model(torch.nn.Module):
    """Networks and scales; eta(x, t), velocity(x, z, t) take and return numpy arrays."""

    def __init__(self, cfg, box, scales, xb, hb, device, dtype):
        super().__init__()
        self.cfg, self.box, self.scales = cfg, box, scales
        self.device, self.dtype = device, dtype
        gen = torch.Generator().manual_seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.eta_net = FourierMLP(2, cfg.sigma_eta, cfg.n_ff, cfg.width_eta, cfg.depth_eta, gen)
        self.phi_net = FourierMLP(3, cfg.sigma_phi, cfg.n_ff, cfg.width_phi, cfg.depth_phi, gen)
        self.register_buffer('xb', torch.as_tensor(xb))
        self.register_buffer('hb', torch.as_tensor(hb))
        self.to(device=device, dtype=dtype)

    # fields on tensors
    def eta_t(self, x, t):
        b = self.box
        v = torch.stack([(x - b['xm']) / b['sx'], (t - b['tm']) / b['st']], dim=1)
        return b['eta0'] + self.scales['Hs'] * self.eta_net(v)

    def phi_t(self, x, z, t):
        b = self.box
        v = torch.stack([(x - b['xm']) / b['sx'], (z - b['zm']) / b['sz'], (t - b['tm']) / b['st']], dim=1)
        return self.scales['Phi'] * self.phi_net(v)

    def h_and_slope(self, x):
        """Depth below the datum and its slope, linear between the bed points (constant outside)."""
        xb, hb = self.xb, self.hb
        i = torch.clamp(torch.searchsorted(xb, x.detach().contiguous()) - 1, 0, xb.numel() - 2)
        s = (hb[i + 1] - hb[i]) / (xb[i + 1] - xb[i])
        xc = torch.minimum(torch.maximum(x, xb[0]), xb[-1])
        h = hb[i] + s * (xc - xb[i])
        s = torch.where((x < xb[0]) | (x > xb[-1]), torch.zeros_like(s), s)
        return h, s

    # numpy interface
    def _tensor(self, a):
        return torch.as_tensor(np.asarray(a, float).ravel(), dtype=self.dtype, device=self.device)

    def eta(self, x, t):
        x, t = np.broadcast_arrays(np.asarray(x, float), np.asarray(t, float))
        with torch.no_grad():
            e = self.eta_t(self._tensor(x), self._tensor(t))
        return e.cpu().numpy().reshape(x.shape)

    def velocity(self, x, z, t):
        x, z, t = np.broadcast_arrays(np.asarray(x, float), np.asarray(z, float), np.asarray(t, float))
        xs = self._tensor(x).requires_grad_(True)
        zs = self._tensor(z).requires_grad_(True)
        p = self.phi_t(xs, zs, self._tensor(t))
        u, w = torch.autograd.grad(p.sum(), [xs, zs])
        return u.detach().cpu().numpy().reshape(x.shape), w.detach().cpu().numpy().reshape(x.shape)


def _grad(y, x):
    return torch.autograd.grad(y.sum(), x, create_graph=True)


def fit(x_obs, t_obs, eta_obs, x_bed, h_bed, cfg=Config(), verbose=True, crest_track=None):
    """Train the PINN on observations of the free surface over a bed of depth h_bed(x_bed).

    x_obs, t_obs, eta_obs: 1-D arrays of the observed surface (m, s, m), same datum as -h.
    x_bed, h_bed: depth below the datum along x (m), increasing x_bed.
    crest_track: optional (t, x) of the crest, used when cfg.crest_focus > 0.
    """
    dtype = torch.float64 if cfg.x64 else torch.float32
    device = torch.device(cfg.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    x_obs, t_obs, eta_obs = (np.asarray(a, float).ravel() for a in (x_obs, t_obs, eta_obs))
    xb, hb = np.asarray(x_bed, float), np.asarray(h_bed, float)
    x0, x1 = x_obs.min(), x_obs.max()
    t0, t1 = t_obs.min(), t_obs.max()
    d0 = float(np.median(np.interp(x_obs, xb, hb)))
    eta_lo, eta_hi = np.percentile(eta_obs, [1, 99])
    scales = {'L': d0, 'U': np.sqrt(G * d0), 'Hs': max(eta_hi - eta_lo, 0.1), 'Phi': np.sqrt(G * d0) * d0}
    box = {'xm': (x0 + x1) / 2, 'sx': (x1 - x0) / 2, 'tm': (t0 + t1) / 2, 'st': max((t1 - t0) / 2, 1e-3),
           'zm': (eta_hi - d0) / 2, 'sz': (eta_hi + d0) / 2, 'eta0': float(np.mean(eta_obs)),
           'x0': x0, 'x1': x1, 't0': t0, 't1': t1}
    box = {k: float(v) for k, v in box.items()}
    scales = {k: float(v) for k, v in scales.items()}
    model = Model(cfg, box, scales, xb, hb, device, dtype)
    if verbose:
        print(f"  device {device}, {'float64' if cfg.x64 else 'float32'}")
    rng = np.random.default_rng(cfg.seed)

    def T(a):
        return torch.as_tensor(a, dtype=dtype, device=device)

    def draw_x(t):
        x = rng.uniform(x0, x1, t.size)
        if crest_track is not None and cfg.crest_focus > 0:
            m = rng.random(t.size) < cfg.crest_focus
            tt, xx = np.asarray(crest_track[0], float), np.asarray(crest_track[1], float)
            f = np.isfinite(tt) & np.isfinite(xx)
            xc = np.interp(t[m], tt[f], xx[f])
            x[m] = np.clip(xc + rng.uniform(-cfg.crest_halfwidth, cfg.crest_halfwidth, m.sum()), x0, x1)
        return x

    def sample(scale=1.0):
        nd, nf, ns, nb = (int(scale * n) for n in (cfg.n_data, cfg.n_interior, cfg.n_surface, cfg.n_bed))
        i = rng.choice(x_obs.size, size=min(nd, x_obs.size), replace=False)
        tf = rng.uniform(t0, t1, nf)
        xf = draw_x(tf)
        top = model.eta(xf, tf)
        bot = -np.interp(xf, xb, hb)
        zf = bot + rng.uniform(0, 1, nf) * (top - bot)
        ts = rng.uniform(t0, t1, ns)
        xs = draw_x(ts)
        tbb = rng.uniform(t0, t1, nb)
        xbb = rng.uniform(x0, x1, nb)
        return {'xd': T(x_obs[i]), 'td': T(t_obs[i]), 'ed': T(eta_obs[i]), 'xf': T(xf), 'zf': T(zf), 'tf': T(tf),
                'xs': T(xs), 'ts': T(ts), 'xb': T(xbb), 'tb': T(tbb)}

    def residual_parts(batch, physics=True, data=True):
        zero = torch.zeros((), dtype=dtype, device=device)
        parts = [zero] * 5
        if data:
            r = (model.eta_t(batch['xd'], batch['td']) - batch['ed']) / cfg.tol_data
            parts[0] = torch.mean(r ** 2)
        if not physics:
            return parts
        # Laplace equation in the fluid
        x = batch['xf'].clone().requires_grad_(True)
        z = batch['zf'].clone().requires_grad_(True)
        p = model.phi_t(x, z, batch['tf'])
        px, pz = _grad(p, [x, z])
        pxx = _grad(px, x)[0]
        pzz = _grad(pz, z)[0]
        parts[1] = torch.mean(((pxx + pzz) / cfg.tol_laplace) ** 2)
        # kinematic and dynamic conditions at z = eta (partial derivatives of phi at fixed z)
        x1_ = batch['xs'].clone().requires_grad_(True)
        t1_ = batch['ts'].clone().requires_grad_(True)
        e = model.eta_t(x1_, t1_)
        ex, et = _grad(e, [x1_, t1_])
        x2_ = batch['xs'].clone().requires_grad_(True)
        t2_ = batch['ts'].clone().requires_grad_(True)
        ps = model.phi_t(x2_, e, t2_)
        sx, sz, st = _grad(ps, [x2_, e, t2_])
        parts[2] = torch.mean(((et + sx * ex - sz) / cfg.tol_kin) ** 2)
        parts[3] = torch.mean(((st + 0.5 * (sx ** 2 + sz ** 2) + G * e) / cfg.tol_dyn) ** 2)
        # no flow through the bed
        xb_ = batch['xb'].clone().requires_grad_(True)
        h, hx = model.h_and_slope(batch['xb'])
        zb_ = (-h).detach().requires_grad_(True)
        pb = model.phi_t(xb_, zb_, batch['tb'])
        bx, bz = _grad(pb, [xb_, zb_])
        parts[4] = torch.mean(((bz + hx * bx) / cfg.tol_bed) ** 2)
        return parts

    hist = []
    tic = time.time()
    it_total = 0

    def log(tag, it, val, parts):
        pr = [float(q.detach()) if torch.is_tensor(q) else float(q) for q in parts]
        print(f"  {tag} {it:6d}  loss {val:.3e}  data {pr[0]:.2e}  laplace {pr[1]:.2e}  kin {pr[2]:.2e}  "
              f"dyn {pr[3]:.2e}  bed {pr[4]:.2e}  ({time.time() - tic:.0f} s)")

    stages = [('eta on the observations', cfg.steps_eta, list(model.eta_net.parameters()), False, cfg.lr),
              ('phi on the physics, eta frozen', cfg.steps_phi, list(model.phi_net.parameters()), True, cfg.lr),
              ('both on all terms', cfg.steps_joint, list(model.parameters()), True, cfg.lr_joint)]
    for si, (name, n_steps, params, physics, lr) in enumerate(stages):
        if n_steps <= 0:
            continue
        if verbose:
            print(f"  stage: {name} ({n_steps} steps)")
        opt = torch.optim.Adam(params, lr=lr)
        gamma = (cfg.lr_final / lr) ** (1.0 / n_steps)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)
        use_data = si != 1
        for it in range(n_steps):
            if it % cfg.resample == 0:
                batch = sample()
            opt.zero_grad()
            parts = residual_parts(batch, physics=physics, data=use_data)
            val = sum(parts)
            val.backward()
            opt.step()
            sched.step()
            if verbose and (it % cfg.log_every == 0 or it == n_steps - 1):
                log('step', it_total, float(val.detach()), parts)
            if it % 100 == 0:
                hist.append([it_total, float(val.detach())] + [float(q.detach()) for q in parts])
            it_total += 1

    if cfg.steps_lbfgs > 0:
        for rnd in range(max(cfg.lbfgs_rounds, 1)):
            if verbose:
                print(f"  stage: L-BFGS on all terms, round {rnd + 1}/{max(cfg.lbfgs_rounds, 1)} "
                      f"({cfg.steps_lbfgs} iterations, fixed batch x{cfg.lbfgs_batch:g})")
            fixed = sample(cfg.lbfgs_batch)
            opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=cfg.steps_lbfgs,
                                    max_eval=int(1.5 * cfg.steps_lbfgs), history_size=50, tolerance_grad=1e-12,
                                    tolerance_change=1e-14, line_search_fn='strong_wolfe')
            count = [0]

            def closure():
                opt.zero_grad()
                parts = residual_parts(fixed)
                val = sum(parts)
                val.backward()
                if verbose and count[0] % max(cfg.steps_lbfgs // 4, 1) == 0:
                    log('lbfgs', count[0], float(val.detach()), parts)
                if count[0] % 100 == 0:
                    hist.append([it_total + count[0], float(val.detach())] + [float(q.detach()) for q in parts])
                count[0] += 1
                return val

            opt.step(closure)
            it_total += count[0]
            if verbose:
                chk = residual_parts(sample())
                log('check', it_total, float(sum(chk).detach()), chk)
    # held-out check: fresh points everywhere, and fresh points near the crest only
    final = [q.detach() for q in residual_parts(sample(2.0))]
    hist.append([it_total, float(sum(final))] + [float(q) for q in final])
    if verbose:
        log('final', it_total, float(sum(final)), final)
    model.final_crest = None
    if crest_track is not None:
        keep_focus = cfg.crest_focus
        cfg.crest_focus = 1.0
        fc = [float(q.detach()) for q in residual_parts(sample(2.0))]
        cfg.crest_focus = keep_focus
        model.final_crest = fc
        if verbose:
            log('crest', it_total, sum(fc), fc)
    model.history = np.array(hist)
    return model


def crest_kinematics(model, t_eval, x_guess=None, half_width=3.0, dx=0.01, c_window=5):
    """Crest position, elevation and fluid velocity of the reconstructed surface at times t_eval,
    and the crest speed C from a straight-line fit to c_window consecutive crest positions.
    x_guess: approximate crest positions (same length as t_eval) to choose the crest; if None,
    the highest point of the window is used."""
    t_eval = np.asarray(t_eval, float)
    box = model.box
    xs = np.arange(box['x0'], box['x1'] + dx / 2, dx)
    xc = np.full(t_eval.size, np.nan)
    ec = np.full(t_eval.size, np.nan)
    for i, t in enumerate(t_eval):
        e = model.eta(xs, np.full(xs.size, t))
        sel = np.ones(xs.size, bool) if x_guess is None else np.abs(xs - x_guess[i]) <= half_width
        if not sel.any():
            continue
        j = np.flatnonzero(sel)[np.argmax(e[sel])]
        if 0 < j < xs.size - 1:                      # parabola through the three highest points
            a, b, c = e[j - 1], e[j], e[j + 1]
            den = a - 2 * b + c
            off = 0.5 * (a - c) / den if den < 0 else 0.0
            xc[i] = xs[j] + off * dx
            ec[i] = b - 0.25 * (a - c) * off
        else:
            xc[i], ec[i] = xs[j], e[j]
    ok = np.isfinite(xc)
    u = np.full(t_eval.size, np.nan)
    w = np.full(t_eval.size, np.nan)
    if ok.any():
        u[ok], w[ok] = model.velocity(xc[ok], ec[ok], t_eval[ok])
    C = np.full(t_eval.size, np.nan)
    for i in range(t_eval.size):
        hw = c_window // 2
        k = np.arange(max(0, i - hw), min(t_eval.size, i + hw + 1))
        k = k[np.isfinite(xc[k])]
        if k.size >= min(c_window, 3) and k.size > hw:
            C[i] = abs(np.polyfit(t_eval[k], xc[k], 1)[0])
    direction = np.sign(np.nanmedian(np.gradient(xc))) if ok.sum() > 1 else 1.0
    return {'t': t_eval, 'xc': xc, 'eta_c': ec, 'u': u, 'w': w, 'C': C, 'B': u * direction / C}
