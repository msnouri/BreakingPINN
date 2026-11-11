"""Figure of the steady-wave benchmark: B at each scan for several sampling cases, and the velocity
profile under the crest with and without the L-BFGS stage."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 15,
                     'axes.linewidth': 1.6, 'xtick.major.width': 1.4, 'ytick.major.width': 1.4,
                     'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True})
R = {k: json.load(open(f'run_{k}.json')) for k in 'fgijkl'}
dt = 1 / 5.17
fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))

cases = [('g', 'full coverage', 'o', '#0072BD'), ('j', '54% coverage, measured points', 's', '#D95319'),
         ('l', '54% coverage, gaps filled', '^', '#7E2F8E'), ('k', 'depth 10 cm too large', 'D', '#77AC30'),
         ('f', 'without L-BFGS', 'v', '#7F7F7F'), ('i', 'steeper wave, 76% coverage', 'P', '#A2142F')]
for k, lab, mk, col in cases:
    B = np.array(R[k]['B_pinn_all'])
    ax[0].plot(np.arange(B.size) * dt, B, '-' + mk, color=col, ms=6, lw=1.4, label=lab)
for k in ('g', 'i'):
    ax[0].axhline(R[k]['B_exact'], color='k', ls='--', lw=1.4)
ax[0].text(0.02, R['g']['B_exact'] + 0.008, r'exact, $H/d = 0.50$', fontsize=13)
ax[0].text(0.02, R['i']['B_exact'] + 0.008, r'exact, $H/d = 0.58$', fontsize=13)
ax[0].set_xlabel(r'$t$ (s)')
ax[0].set_ylabel(r'$B = u_c / C$')
ax[0].set_xlim(-0.1, 2.25)
ax[0].set_ylim(0.36, 0.58)
ax[0].set_xticks(np.arange(0, 2.5, 0.5))
ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.30), ncol=2, frameon=False, fontsize=12)

z = np.array(R['g']['profile_z'])
ax[1].plot(R['g']['profile_u_exact'], z, 'k-', lw=2.2, label='exact')
ax[1].plot(R['g']['profile_u_pinn'], z, 'o', color='#0072BD', ms=6, label='PINN')
ax[1].plot(R['f']['profile_u_pinn'], z, 'v', color='#7F7F7F', ms=6, label='PINN without L-BFGS')
ax[1].set_xlabel(r'$u$ under the crest (m s$^{-1}$)')
ax[1].set_ylabel(r'$z$ (m)')
ax[1].set_xlim(1.0, 2.7)
ax[1].set_ylim(-3.1, 1.3)
ax[1].set_xticks(np.arange(1.0, 2.8, 0.4))
ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=3, frameon=False, fontsize=12)
for a, lab in zip(ax, ('(a)', '(b)')):
    a.text(0.97, 0.96, lab, transform=a.transAxes, ha='right', va='top', fontsize=18, fontweight='bold')
fig.tight_layout()
fig.savefig('bench_steady.png', dpi=200)
fig.savefig('bench_steady.pdf')
print('figure written')
