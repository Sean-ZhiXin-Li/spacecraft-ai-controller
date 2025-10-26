import matplotlib.pyplot as plt
import numpy as np

def plot_radius(time: np.ndarray, r: np.ndarray, r_target: float, out_png: str):
    plt.figure()
    plt.plot(time, r, label='r(t)')
    plt.axhline(r_target, linestyle='--', label='r*')
    plt.xlabel('t'); plt.ylabel('radius'); plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

def plot_error(time: np.ndarray, norm_e: np.ndarray, out_png: str):
    plt.figure()
    plt.plot(time, np.abs(norm_e), label='|ê(t)|')
    plt.xlabel('t'); plt.ylabel('normalized error'); plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()
