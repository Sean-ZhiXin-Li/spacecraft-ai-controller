# Phase39A Missing Measurements

Scope: observability gaps that would improve interpretability before any new controller implementation. These are proposed measurements only. They do not change physics, thresholds, or historical artifacts.

| Missing measurement | Why it matters | Phase38 unknown it could answer | Risk to add | Priority |
|---|---|---|---|---|
| Specific orbital energy | Helps distinguish energy-deficient, energy-excess, and geometry-only failures using standard orbital mechanics quantities. | Are near-crossing and over-conservative-transfer failures energy-limited or timing-limited? | Low if computed passively from existing position/velocity and gravitational parameter. | P0 |
| Energy error evolution | Tracks how energy changes over the rollout relative to a target/reference energy. | Does a failed trajectory approach the correct energy but miss geometry, or never reach the right energy regime? | Medium because target/reference definition must be explicit. | P0 |
| Angular momentum | Captures tangential state and orbital geometry in a standard scalar/vector quantity. | Is over-conservative-transfer failure angular-momentum limited rather than radial-timing limited? | Low if computed passively from state. | P0 |
| Angular momentum error evolution | Shows whether angular momentum approaches a target/reference value over time. | Can angular-momentum mismatch explain failures where closest approach improves without crossing? | Medium because reference value must be defined carefully. | P0 |
| Eccentricity estimate | Provides a compact orbit-shape descriptor from state. | Are failures associated with overly eccentric or insufficiently eccentric transfer geometry? | Medium because simplified simulator assumptions and units must be documented. | P1 |
| Cumulative delta-v proxy | Summarizes integrated control effort. | Are failures caused by insufficient effort, excessive effort, or ineffective effort direction? | Medium because current actions may be normalized thrust commands rather than physical delta-v. | P0 |
| Radial/tangential work proxies | Separates effort applied along radial versus tangential directions. | Are tested radial/tangential interventions actually doing useful orbital work? | Medium because work calculation must be tied to available force/action semantics. | P1 |
| Controller phase transition log | Ordered sequence of controller phases and transition steps. | Did failures occur because the controller entered a phase too early, too late, or not at all? | Low to medium; requires logging without changing phase logic. | P0 |
| State history summaries | Min/max/mean/final summaries for radius error, radial velocity, tangential velocity error, speed, energy, and angular momentum. | Which state dimensions separate failure modes over the trajectory, not just at closest approach? | Low if generated from existing rollout arrays. | P0 |
| Time-in-phase counters | Counts or durations spent in each controller phase. | Are non-crossing failures dominated by one controller stage or by missing handoff? | Low if phase labels already exist internally. | P1 |
| Closest-approach local state snapshot | State values at closest approach: radius error, radial velocity, tangential velocity error, speed, energy, angular momentum, phase. | Why did closest approach fail to become crossing? | Low if computed from saved trajectory state. | P0 |
| Pre-cross trajectory descriptors | Summaries before first crossing or before closest approach. | What distinguishes crossing-producing approach geometry before the event? | Medium because descriptor set must remain simple and non-tuned. | P1 |
| Handoff state snapshot | State at first crossing, including energy/angular momentum if crossing occurs. | Why do some crossings become recoverable while others need post-cross correction? | Low; crossing-state metrics already exist but can be expanded. | P1 |
| Action-direction summaries | Mean/max radial and tangential action components by phase. | Did the controller actually apply the intended radial/tangential behavior? | Medium; must avoid treating action components as physical work without conversion. | P1 |
| Regression-case observability parity | Ensures new logs are collected for both selected failure cases and known crossing-producing guard cases. | Does a proposed variable improve failures while damaging known crossing cases? | Low. | P0 |

## P0 Recommendation

Before new controller implementation, prioritize passive logging of:

- specific orbital energy;
- angular momentum;
- energy/angular-momentum error summaries with explicit reference definitions;
- cumulative delta-v or effort proxy;
- phase transition log;
- state history summaries;
- closest-approach local state snapshot;
- regression-case observability parity.

These measurements would make future failures interpretable without changing control behavior.
