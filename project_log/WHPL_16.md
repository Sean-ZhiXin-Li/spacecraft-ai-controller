# WHPL_16 — Stability Domain → Mechanism Compression

## Date

Day16: 2026-02-22

---

# Objective

Transition from boundary visualization (Day15) to mechanism-level explanation.

Instead of drawing more maps, compress observed stability differences into dynamical structure.

Core question:

> Why does thrust amplification change the stability domain?

---

# Experimental Setup (Near-field)

Fixed condition:

* r0_over_target ≈ 1.005

Four configurations:

1. 800N + always_on (baseline)
2. 2000N + always_on (A)
3. 800N + gated (B)
4. 2000N + gated (C)

Metrics extracted per run:

* min_r_err
* max_r_err
* min_vr
* t_flip
* t_cross
* delta_r
* rel_delta_r

---

# Observations

## 800N + always_on (baseline)

* t_flip ≈ 840
* min_vr ≈ −890 m/s
* r_err almost constant
* No target crossing

Behavior:
Conservative correction, limited inward momentum.
Stable but slow convergence.

---

## 2000N + always_on (A)

* t_flip ≈ 327 (earlier)
* min_vr ≈ −1346 m/s (stronger inward plunge)
* Larger inward Δr
* No crossing within 2000 steps

Interpretation:
Thrust amplification increases effective closed-loop gain.
Earlier radial sign flip → stronger inward momentum accumulation.
System enters high-energy correction phase sooner.

This forms the structural precursor of overshoot risk.

---

## 800N + gated (B)

* t_flip = None
* min_vr ≈ +72 m/s (never truly inward)
* r_err barely decreases

Interpretation:
Gating suppresses correction under low thrust.
Highly conservative behavior.
Stable but under-responsive.

---

## 2000N + gated (C)

* t_flip ≈ 864 (delayed)
* min_vr ≈ −563 m/s (reduced inward magnitude)
* Smaller Δr magnitude than always_on

Interpretation:
Gating under high thrust reduces effective energy injection.
Suppresses inward momentum buildup.
Delays entry into high-gain regime.

Explains expanded stability domain under high thrust.

---

# Mechanism Compression

Thrust ≈ closed-loop gain.

When gain increases:

* Earlier velocity sign flip
* Larger inward velocity magnitude
* Faster error reduction
* Higher risk of overshoot beyond target radius

always_on allows full gain injection.

High thrust + always_on:
→ enters high-momentum regime
→ greater plunge tendency
→ potential overshoot/oscillation in extended horizon

Gated control acts as energy regulator:

* Reduces effective gain
* Delays inward transition
* Limits radial momentum amplitude

Therefore:

> Thrust is not monotonically stabilizing.
> It amplifies system gain.
> Stability depends on how energy injection is regulated.

---

# Structural Insight

In the near-field regime (r0 ≈ 1.005),
current 2000-step horizon captures the momentum build-up phase,
not full geometric convergence.

The system is observed in its kinetic shaping stage.

High thrust shifts the system into a higher-energy manifold.
Gating modifies the trajectory within that manifold.

---

# Conclusion

Day16 successfully transforms stability boundary observation
into dynamical mechanism explanation.

We now understand:

* Why thrust amplification changes stability region
* Why always_on risks high-momentum correction
* Why gated expands stability under high thrust

This establishes a two-dimensional structural interpretation:

Axis 1: Thrust magnitude (gain scaling)
Axis 2: Control structure (energy regulation)

Stability domain is an emergent property of their interaction.

---

Next step:
Extend horizon to test overshoot phase and validate full-cycle dynamics.
