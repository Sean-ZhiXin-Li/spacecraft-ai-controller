# WHPL_16 — Stability Domain → Mechanism Compression

## Date

Day16: 2026-02-22 -- 23

---

# Objective

Transition from boundary visualization (Day15) to mechanism-level explanation.

Instead of drawing additional stability maps, compress observed stability differences into dynamical structure.

Core question:

> Why does thrust amplification change the stability domain?

---

# Experimental Setup (Near-field Regime)

Fixed initial condition:

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

---

# Observations

## 800N + always_on (baseline)

* t_flip ≈ 840
* min_vr ≈ −890 m/s
* r_err remains nearly constant
* No target crossing within 2000 steps

Behavior:
Conservative inward correction with limited momentum buildup.
Stable, but convergence is slow.

---

## 2000N + always_on (A)

* t_flip ≈ 327 (earlier)
* min_vr ≈ −1346 m/s (stronger inward plunge)
* Larger inward Δr
* No crossing within 2000 steps

Interpretation:
Thrust amplification increases effective closed-loop gain.
Earlier velocity sign flip indicates faster inward transition.
Stronger inward momentum accumulation emerges.

This configuration enters a high-energy correction phase sooner.

This forms the structural precursor of overshoot risk in extended horizons.

---

## 800N + gated (B)

* t_flip = None
* min_vr ≈ +72 m/s (never significantly inward)
* r_err barely decreases

Interpretation:
Gating suppresses correction under low thrust.
Energy injection remains constrained.

Behavior is highly conservative.
Stable but under-responsive.

---

## 2000N + gated (C)

* t_flip ≈ 864 (delayed compared to always_on)
* min_vr ≈ −563 m/s (reduced inward magnitude)
* Smaller Δr magnitude than always_on

Interpretation:
Under high thrust, gating reduces effective energy injection.
Inward momentum buildup is moderated.

Entry into high-gain regime is delayed.

This explains expansion of the stability domain under high thrust when regulation is applied.

---

# Mechanism Compression

Thrust acts as a gain scaling parameter in the radial feedback loop.

Increasing thrust increases effective loop gain, which:

* Accelerates error correction
* Increases inward momentum magnitude
* Reduces time to velocity sign flip
* Raises overshoot probability in extended horizons

Therefore, thrust does not directly "stabilize" the system.
It reshapes the energy landscape of the closed-loop dynamics.

always_on allows full gain injection.

High thrust + always_on:

→ enters high-momentum regime
→ increases plunge tendency
→ elevates overshoot / oscillation risk in longer horizons

Gated control functions as an energy regulator:

* Reduces effective gain
* Delays inward transition
* Limits radial momentum amplitude

Therefore:

> Thrust is not monotonically stabilizing.
> It amplifies system gain.
> Stability depends on how energy injection is regulated.

---

# Structural Insight

We define a structural plane:

Dimension 1 — Gain scaling (Thrust magnitude)
Dimension 2 — Energy regulation structure (always_on vs gated)

Stability is not a function of thrust alone.
It is a surface defined over this 2D interaction space.

The observed stability domain is a projection of this surface
under a finite-horizon kinetic observation window.

In the near-field regime (r0 ≈ 1.005),
the current 2000-step horizon captures the momentum build-up phase,
not full geometric convergence.

The system is observed in its kinetic shaping stage.

High thrust shifts the system into a higher-energy manifold.
Gating modifies the trajectory within that manifold.

---

# Conclusion

Day16 transforms stability boundary observation
into dynamical mechanism explanation.

We now understand:

* Why thrust amplification changes stability region
* Why always_on risks high-momentum correction
* Why gated expands stability under high thrust

Stability domain is an emergent property of the interaction between:

1. Gain scaling (thrust)
2. Energy regulation structure (control logic)

---

# Structural Continuation (Day17)

Day17 does not extend experiments.
It formalizes trajectory-level metrics into a batch summary table.

The summary CSV becomes the structural interface between:

raw kinetic trajectories → 2D structural interpretation

This freezes the mechanism layer before further expansion.

---

Next phase (post-freeze):

Extend horizon to test overshoot phase
and validate full-cycle dynamics in extended time regimes.
