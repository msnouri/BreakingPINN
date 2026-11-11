"""Common style for the paper figures: LaTeX (Computer Modern) fonts, sizes for a full-width figure.

    import figstyle
    figstyle.use()                    # before creating figures
    fig.savefig('name.pdf')           # vector output for the paper

Uses the system LaTeX (text.usetex) when `latex` and cm-super are installed (Ubuntu:
`sudo apt install texlive-latex-extra cm-super dvipng`); otherwise falls back to matplotlib's
Computer Modern mathtext, which looks the same for most labels.
"""
import shutil

import matplotlib

COLORS = {'spilling': '#0072BD', 'plunging': '#D95319', 'nonbreaking': '#7F7F7F', 'exact': 'k'}
MARKERS = {'spilling': 'o', 'plunging': 's', 'nonbreaking': '^'}


def _latex_ok():
    """True when LaTeX and the Type 1 Computer Modern fonts (cm-super) that matplotlib needs are installed."""
    import subprocess
    if shutil.which('latex') is None or shutil.which('kpsewhich') is None:
        return False
    try:
        out = subprocess.run(['kpsewhich', 'type1ec.sty'], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return bool(out.strip())


def use(base=17, usetex=None):
    if usetex is None:
        usetex = _latex_ok()
    rc = {
        'font.family': 'serif', 'font.size': base,
        'axes.labelsize': base + 1, 'axes.titlesize': base + 1, 'legend.fontsize': base - 3,
        'xtick.labelsize': base - 1, 'ytick.labelsize': base - 1,
        'axes.linewidth': 1.4, 'lines.linewidth': 2.0, 'lines.markersize': 7,
        'xtick.major.width': 1.3, 'ytick.major.width': 1.3, 'xtick.minor.width': 1.0, 'ytick.minor.width': 1.0,
        'xtick.major.size': 6, 'ytick.major.size': 6, 'xtick.minor.size': 3.5, 'ytick.minor.size': 3.5,
        'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
        'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.dpi': 300, 'pdf.fonttype': 42,
    }
    if usetex:
        rc.update({'text.usetex': True, 'font.serif': ['Computer Modern Roman'],
                   'text.latex.preamble': r'\usepackage{amsmath}'})
    else:
        rc.update({'text.usetex': False, 'mathtext.fontset': 'cm', 'font.serif': ['cmr10', 'DejaVu Serif'],
                   'axes.formatter.use_mathtext': True})
    matplotlib.rcParams.update(rc)
    return usetex


def panel_labels(axes, x=0.03, y=0.96):
    for a, l in zip(axes, 'abcdefgh'):
        a.text(x, y, f'({l})', transform=a.transAxes, va='top', ha='left')
