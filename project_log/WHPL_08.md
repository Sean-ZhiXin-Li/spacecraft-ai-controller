# WHPL_08 — Structural Failure Localization

## Hypothesis
H1: Radius error does not converge because the controller fails to establish radial-error feedback (no stable radial convergence structure).
H2: The controller may still change orbital energy/angular momentum via tangential/radial thrust components, but without reducing |r - r_target|.

## Instrumentation (logging only)
Added periodic diagnostics (every 200 steps):
- cos(thrust, radial_unit)
- cos(thrust, tangential_unit)
- v_r = dot(v, r_unit)
- dr_dt ≈ (r_t - r_{t-1}) / dt

No changes to physics, reward, or controller.

## Evidence (representative logs)
step=200  cos_rad=-0.695590 cos_tan=+0.718439  v_r=+8.524970e+02  dr_dt=+2.097154e+05
step=400  cos_rad=-0.678932 cos_tan=+0.734201  v_r=-7.421101e+01  dr_dt=+1.048579e+05
step=600  cos_rad=-0.671581 cos_tan=+0.740931  v_r=-9.118462e+02  dr_dt=-1.048574e+05
step=800  cos_rad=-0.676450 cos_tan=+0.736488  v_r=-1.669324e+03  dr_dt=-2.097152e+05
step=1800 cos_rad=-0.847747 cos_tan=+0.530401  v_r=-4.548419e+03  dr_dt=-8.388604e+05

Radius error statistics:
min≈mean≈max≈1.875e+12 (no observable convergence).

Meanwhile, orbital quantities evolve:
eps changes substantially (even crosses sign), and |h| changes by orders of magnitude.

## Structural Conclusion
The controller injects a strong radial component (cos(thrust,rad) ~ -0.67 to -0.85, mostly inward) together with tangential thrust, and it clearly alters orbital energy and angular momentum. However, the radius error remains invariant across the episode. This indicates the control structure does not implement a stable radial-error feedback loop that drives r -> r_target. Instead, the policy primarily performs energy/phase shaping (and possibly persistent inward bias), producing radial motion (v_r, dr/dt oscillations) without radial convergence.

## Next Design Direction (WHPL_09)
Introduce an explicit radial-error feedback term (sign + magnitude) to inject *correct* radial control authority, e.g., a bounded controller component driven by (r - r_target) and/or v_r damping, then re-evaluate convergence.
