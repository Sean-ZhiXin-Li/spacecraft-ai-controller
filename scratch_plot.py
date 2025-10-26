import numpy as np
from tools.metrics.metrics_core import radial_distance, radial_error, norm_error
from tools.plots.plot_helpers import plot_radius, plot_error

NPZ = 'logs/day54/spiral_in/smoke/replay.npz'
R_TARGET = 9.375e12

d = np.load(NPZ)
xy = d['obs'][:, :2]
r  = radial_distance(xy)
ne = norm_error(radial_error(r, R_TARGET), R_TARGET)
t = np.arange(len(r))

plot_radius(t, r, R_TARGET, 'logs/day54/spiral_in/smoke/radius.png')
plot_error(t, ne, 'logs/day54/spiral_in/smoke/err_norm.png')
print('saved: radius.png, err_norm.png')
