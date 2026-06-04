# Trajectory Family Audit

## Overall Assessment

Phase36A is a useful first step, but it has not yet discovered a superior family. It has made trajectory-family differences visible. That is enough for a visualization-first phase.

Phase36A supports the hypothesis that crossing-generation is influenced by trajectory geometry, but only cautiously. The evidence is visual and subset-based, not full-benchmark proof.

## baseline_phase34

Meaningful role: reference trajectory.

Result on Phase36A subset:

- `1 / 3` crossings
- `1 / 3` recoverable crossings
- `0` overspeed

Keep as the reference for all Phase36B testing.

## spiral_approach

Meaningful role: yes.

The family corresponds to gradual radius shaping and produced:

- `1 / 3` crossings
- `1 / 3` recoverable crossings
- higher mean crossing potential than baseline in the subset
- no overspeed

This deserves Phase36B testing. It may reveal whether gentle geometric shaping can preserve Phase34 handoff quality.

## grazing_corridor

Meaningful role: yes.

The family directly targets the Phase35 diagnosis: near-crossing and over-conservative-transfer cases. It produced:

- `1 / 3` crossings
- `1 / 3` recoverable crossings
- highest mean crossing potential in the subset
- no overspeed

It also had a higher mean crossing sync among crossing cases, so it may create crossings that are geometrically successful but dynamically rough. This makes it important to test, not automatically good.

## delayed_crossing

Meaningful role: plausible but underdeveloped.

It produced:

- `0 / 3` crossings
- `0 / 3` recoverable crossings
- `0` overspeed
- high mean crossing potential

This is worth redesigning rather than dropping. It may be too conservative, but it is safer than the overspeed-prone families.

## energy_bleed_then_cross

Meaningful role: physically plausible label, poor current implementation.

It produced:

- `0 / 3` crossings
- `3 / 3` overspeed

The idea may be valid, but this implementation is not. It should not move into Phase36B without strong constraints on speed and energy evolution.

## overshoot_return

Meaningful role: plausible orbital concept, poor current implementation.

It produced:

- `0 / 3` crossings
- `3 / 3` overspeed

This family is currently too unstable. Drop or redesign it before full benchmarking.

## two_stage_transfer

Meaningful role: architecturally plausible, poor current implementation.

It produced:

- `0 / 3` crossings
- `3 / 3` overspeed

The concept matches the project architecture, but the current behavior is not useful. Redesign it with explicit speed and corridor constraints before treating it as a candidate.

## Phase36B Candidates

Test in Phase36B:

- `baseline_phase34`
- `spiral_approach`
- `grazing_corridor`
- redesigned `delayed_crossing`

Do not test without redesign:

- `energy_bleed_then_cross`
- `overshoot_return`
- `two_stage_transfer`

## Verdict

Phase36A is appropriately scoped. It clarified geometry but did not improve crossing count. Its main value is narrowing which transfer-family ideas deserve disciplined full-benchmark testing.

