"""Steady water waves of finite height over a flat bed: Fourier approximation of the stream function
(Rienecker and Fenton, 1981; Fenton, 1988), with no mean Eulerian current.

    wave = solve(d=3.0, H=1.5, L=40.0)       # depth, height, wavelength (m)
    wave.c, wave.u_crest, wave.B             # wave speed, horizontal velocity at the crest, B = u/c
    eta = wave.eta(x, t)                     # surface elevation above the still-water level (m)
    u, w = wave.velocity(x, z, t)            # velocities in the fixed frame, z upward from the SWL

The wave travels in +x; its crest is at x = c t. In the frame moving with the wave, the stream
function is psi = -ubar Z + sum_j B_j sinh(j k Z) / cosh(j k d) cos(j k X), with Z measured up from
the bed and X = x - c t. With no Eulerian current, c = ubar.
"""
import numpy as np

G = 9.81


class SteadyWave:
    def __init__(self, d, H, L, N, k, ubar, Q, R, Bj, eta_nodes, X_nodes):
        self.d, self.H, self.L, self.N, self.k = d, H, L, N, k
        self.ubar, self.Q, self.R, self.Bj = ubar, Q, R, Bj
        self.eta_nodes, self.X_nodes = eta_nodes, X_nodes   # surface heights above the bed, X in [0, L/2]
        self.c = ubar                                         # no mean Eulerian current
        j = np.arange(1, N + 1)
        # cosine series of the surface above the bed: Z(X) = sum_j E_j cos(j k X), from the nodes
        m = np.arange(N + 1)
        w = np.ones(N + 1); w[0] = w[-1] = 0.5
        self.E = np.array([2.0 / N * np.sum(w * eta_nodes * np.cos(jj * m * np.pi / N)) for jj in range(N + 1)])
        self.E[0] /= 2.0
        self.E[-1] /= 2.0
        self._j = j
        U0, _ = self._moving_velocity(np.array([0.0]), np.array([eta_nodes[0]]))
        self.u_crest = float(U0[0] + self.c)                  # fixed frame
        self.B = self.u_crest / self.c

    def surface_above_bed(self, X):
        jj = np.arange(self.N + 1)
        return np.cos(np.multiply.outer(np.asarray(X, float), jj * self.k)) @ self.E

    def _moving_velocity(self, X, Z):
        """Velocities (U, W) in the frame moving with the wave; Z above the bed."""
        X = np.asarray(X, float)[..., None]
        Z = np.asarray(Z, float)[..., None]
        j, k, d = self._j, self.k, self.d
        ch = np.cosh(j * k * Z) / np.cosh(j * k * d)
        sh = np.sinh(j * k * Z) / np.cosh(j * k * d)
        U = -self.ubar + np.sum(j * k * self.Bj * ch * np.cos(j * k * X), axis=-1)
        W = np.sum(j * k * self.Bj * sh * np.sin(j * k * X), axis=-1)
        return U, W

    def eta(self, x, t):
        """Surface elevation above the still-water level at (x, t)."""
        X = np.asarray(x, float) - self.c * np.asarray(t, float)
        return self.surface_above_bed(X) - self.d

    def velocity(self, x, z, t):
        """Fixed-frame velocities (u, w) at (x, z, t), z upward from the still-water level."""
        X = np.asarray(x, float) - self.c * np.asarray(t, float)
        U, W = self._moving_velocity(X, np.asarray(z, float) + self.d)
        return U + self.c, W


def _residual(v, H, k, N):
    """Residuals in units of the depth d and sqrt(g d) (d = 1, g = 1)."""
    eta = v[:N + 1]
    Bj = v[N + 1:2 * N + 1]
    ubar, Q, R = v[2 * N + 1:]
    m = np.arange(N + 1)
    j = np.arange(1, N + 1)
    X = m * np.pi / (N * k)
    kx = np.outer(X, j * k)
    kz = np.outer(eta, j * k)
    # sinh(j k eta) / cosh(j k) and cosh(j k eta) / cosh(j k), written to avoid overflow for large j k
    e1 = np.exp(kz - j * k)
    e2 = np.exp(-kz - j * k)
    den = 1.0 + np.exp(-2.0 * j * k)
    sh = (e1 - e2) / den
    ch = (e1 + e2) / den
    cs, sn = np.cos(kx), np.sin(kx)
    psi = -ubar * eta + (sh * cs) @ Bj + Q
    U = -ubar + (ch * cs) @ (j * k * Bj)
    W = (sh * sn) @ (j * k * Bj)
    dyn = 0.5 * (U ** 2 + W ** 2) + eta - R
    w = np.ones(N + 1); w[0] = w[-1] = 0.5
    mean = np.sum(w * eta) / N - 1.0
    height = eta[0] - eta[-1] - H
    return np.concatenate([psi, dyn, [mean, height]])


def _initial(Hs, k, N, X):
    c0 = np.sqrt(np.tanh(k) / k)
    a = Hs / 2
    eta = 1.0 + a * np.cos(k * X)
    Bj = np.zeros(N)
    Bj[0] = a * c0 / np.tanh(k)
    return np.concatenate([eta, Bj, [c0, c0, 0.5 * c0 ** 2 + 1.0]])


def solve(d, H, L, N=40, steps=12, verbose=False, tol=1e-9, min_step=1e-4):
    """Steady wave of height H and wavelength L in depth d, by continuation in height.

    The height is raised from a small wave in `steps` equal increments; when a step fails to converge,
    it is halved and retried from the last converged wave (down to `min_step` in H/d). This reaches
    waves close to the limiting height, where B = u_crest / c approaches 1.
    """
    from scipy.optimize import least_squares
    N = int(N)
    k = 2 * np.pi / (L / d)                     # dimensionless wavenumber k d
    Hn = H / d
    m = np.arange(N + 1)
    X = m * np.pi / (N * k)
    dH = Hn / steps
    Hs_done, v = 0.0, None
    while Hs_done < Hn - 1e-12:
        Hs = min(Hs_done + dH, Hn)
        v0 = _initial(Hs, k, N, X) if v is None else v
        sol = least_squares(_residual, v0, args=(Hs, k, N), xtol=1e-15, ftol=1e-15, gtol=1e-15,
                            x_scale='jac', max_nfev=2000)
        res = np.max(np.abs(_residual(sol.x, Hs, k, N)))
        if res <= tol:
            v, Hs_done = sol.x, Hs
            if verbose:
                print(f"  H/d = {Hs:.4f}: residual {res:.1e}, c = {v[2 * N + 1]:.4f} sqrt(gd)")
            continue
        dH /= 2
        if verbose:
            print(f"  H/d = {Hs:.4f}: residual {res:.1e}, step halved to {dH:.2e}")
        if dH < min_step:
            raise RuntimeError(f"no convergence beyond H/d = {Hs_done:.4f} (residual {res:.1e} at {Hs:.4f})")
    sg = np.sqrt(G * d)
    eta = v[:N + 1] * d
    Bj = v[N + 1:2 * N + 1] * sg * d
    ubar, Q, R = v[2 * N + 1] * sg, v[2 * N + 2] * sg * d, v[2 * N + 3] * G * d
    return SteadyWave(d, H, L, N, k / d, ubar, Q, R, Bj, eta, X * d)


if __name__ == '__main__':
    # checks: small amplitude against linear theory and Stokes' second order; convergence in N
    d, L = 3.0, 40.0
    k = 2 * np.pi / L
    c0 = np.sqrt(G / k * np.tanh(k * d))
    w = solve(d, 0.01, L, N=20, steps=1)
    print(f"small wave: c = {w.c:.5f} m/s, linear theory {c0:.5f} m/s")
    for N in (32, 48, 64):
        w = solve(d, 1.5, L, N=N)
        print(f"d = {d} m, H = 1.5 m, L = {L} m, N = {N}: c = {w.c:.5f} m/s, u_crest = {w.u_crest:.5f} m/s, "
              f"B = {w.B:.5f}, crest {w.eta(0, 0):.4f} m, trough {w.eta(L / 2, 0):.4f} m")
    # steep waves near the limiting height (L/d = 13.3): B approaches 1; convergence in N
    for Hd in (0.70, 0.717, 0.723, 0.727):
        Bs = [solve(d, Hd * d, L, N=N, steps=60).B for N in (32, 40, 48)]
        print(f"H/d = {Hd:.3f}: B = " + ", ".join(f"{b:.4f} (N={N})" for b, N in zip(Bs, (32, 40, 48))))
    # Stokes second order: crest - trough asymmetry for a moderate wave in deep-ish water
    d2, H2, L2 = 20.0, 0.5, 30.0
    w = solve(d2, H2, L2, N=24)
    k2 = 2 * np.pi / L2
    a = H2 / 2
    s2 = a ** 2 * k2 / 4 * np.cosh(k2 * d2) * (2 + np.cosh(2 * k2 * d2)) / np.sinh(k2 * d2) ** 3
    print(f"Stokes check: crest {w.eta(0, 0):.4f} m vs {a + s2:.4f} m (2nd order), trough {w.eta(L2 / 2, 0):.4f} m vs {-a + s2:.4f} m")
