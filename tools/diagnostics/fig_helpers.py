"""
Figure helpers for standardized visual diagnostics.
"""

import os
import matplotlib.pyplot as plt

DEFAULT_FIGSIZE = (6.4, 4.0)  # 640x400
DEFAULT_DPI = 150

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def new_ax(figsize=DEFAULT_FIGSIZE):
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, alpha=0.3)
    return fig, ax

def save_fig(fig, outdir: str, fname: str, dpi: int = DEFAULT_DPI):
    ensure_dir(outdir)
    outpath = os.path.join(outdir, fname)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    return outpath
