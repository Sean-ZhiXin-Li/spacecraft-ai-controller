# Phase 33 Optimal Structure Extraction Summary

## Findings

1. What structural behavior made optimal better? Smooth low-thrust synchronization of radius, vr, and vt over the full horizon, rather than discrete Burn A/B corrections.
2. Was recoverability mainly timing, geometry, or control smoothness? Primarily geometry-time synchronization; smoothness is the mechanism that prevents corrections from destroying another state component.
3. What architecture limit blocked prior phases? Prior phases treated crossing and insertion as staged events, while the optimal trajectory keeps steering after the first crossing until the full recoverability state aligns.
4. Which missing structure matters most? A continuous low-authority post-cross steering arc that can trade temporary crossing quality for late basin entry.
5. Can this be approximated heuristically? Partially, with an MPC-lite or imitation controller trained on optimal trajectories; a hand heuristic would be fragile.
6. Should Phase 34 be imitation controller, MPC-lite, CasADi full collocation, or hybrid heuristic-optimal? `hybrid heuristic-optimal`: first imitate Phase 32 motifs, then validate with MPC-lite or CasADi collocation.

## Best Case

- Objective mode: `recoverability_target`
- Case label: `baseline_crossing_high_angle`
- Crossing step: `81`
- Best sync: `0.000464`
- Best distance: `0.000470`

## Honesty Note

- Phase 32 used SciPy direct shooting because CasADi was unavailable in the checked runtime.
- The optimal advantage is structural for the representative recoverable case, but it is not yet a production-controller result.
- The best Phase 32 row is crossing-labeled because it both crosses and later reaches recoverability; the first crossing state remains outside the recoverability basin.
- Phase31 baseline thrust profile was not logged; the overlay uses the Phase22/31 baseline stage activity as a control proxy.

## Phase 34 Blueprint

- Extract state-control pairs from Phase 32 trajectories.
- Train or hand-fit a low-thrust smooth steering policy around recoverability-targeted state errors.
- Add an MPC-lite layer that optimizes short receding windows for sync error, not crossing alone.
- Validate against Phase31 baseline on the reduced grid before attempting broader generalization.