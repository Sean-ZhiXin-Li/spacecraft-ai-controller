# Failure Label Taxonomy v0

Status: Week 2 draft.

Date: July 8-14, 2026.

Scope: future 2D spacecraft recoverability experiments after Recoverability Benchmark v1.

This document defines a standard failure-label taxonomy for future recoverability outputs. It is a documentation and schema-design artifact only. It does not modify controllers, historical CSVs, Phase34/36/37 scripts, or old analysis artifacts.

The taxonomy principle is:

```text
Failure labels describe mechanism, not just outcome.
```

## Purpose

Failure labels exist to make future recoverability experiments auditable.

They should answer:

- Did the rollout fail before a meaningful state could be evaluated?
- Did it violate a safety or validity condition?
- Did it never cross the target radius?
- Did it cross but remain unrecoverable?
- Did it become recoverable after crossing but fail later?
- Is the available artifact insufficient to support a more precise label?

Labels must support negative results without turning diagnostic signals into benchmark progress.

## Relationship To Recoverability Benchmark v1

Recoverability Benchmark v1 separates target-radius crossing, recoverable crossing, final simulator success, safety flags, and diagnostic subset evidence.

This taxonomy is the label layer for that benchmark. It standardizes how future outputs should encode the mechanism behind each terminal outcome while preserving the Week 1 benchmark rule:

```text
Intermediate success is not recoverable task completion.
```

The taxonomy does not change the protected Phase34/36/37 evidence. It should be used for new outputs and new result schemas only.

## Label Types

### Terminal Label

A terminal label is the single primary label assigned to a rollout after applying the priority rules in this document.

Future CSV/JSON outputs should contain exactly one terminal label per rollout. The terminal label should represent the dominant mechanism that determined the rollout's final status under the declared experiment criteria.

Recommended future field name:

- `terminal_label`

Historical fields such as `termination_reason`, `dominant_failure_label`, `failure_label`, and `simulator_success_label` are evidence sources, not already-normalized terminal labels.

### Precursor Label

A precursor label records an event that happened before termination and may be needed to interpret the terminal label.

Examples:

- `crossed_target_radius`
- `entered_recoverable_basin`
- `phase34_compatible_crossing`
- `near_crossing`
- `over_conservative_transfer`

Precursor labels should not override a higher-priority terminal label. For example, a rollout can cross the target radius and still terminate as `overspeed`.

Recommended future field name:

- `precursor_labels`

For CSV output, use a semicolon-separated list. For JSON output, use an array of strings.

### Diagnostic Label

A diagnostic label describes a hypothesis or mechanism useful for analysis but not strong enough to be the terminal label.

Current examples include Phase36C labels such as `near_crossing` and `over_conservative_transfer`. These explain non-crossing geometry but should not be treated as primary benchmark outcomes.

Recommended future field name:

- `diagnostic_labels`

For CSV output, use a semicolon-separated list. For JSON output, use an array of strings.

### Manual-Audit Note

A manual-audit note is free text for caveats, ambiguity, or reviewer observations. It is not a controlled label.

Manual-audit notes should be used when:

- the artifact has conflicting fields,
- the simulator status is valid but the recoverability interpretation is unclear,
- a subset result damages a regression case,
- a label assignment required human review,
- the available data supports only `unknown`.

Recommended future field name:

- `manual_audit_note`

## Standard Labels

The controlled terminal labels are:

- `success`
- `no_crossing`
- `crossing_unrecoverable`
- `recoverable_crossing_failed_late`
- `overspeed`
- `instability`
- `timeout`
- `resource_depletion`
- `unsafe_state`
- `invalid_simulation`
- `unknown`

Do not add phase-specific terminal labels unless a future benchmark version extends this taxonomy.

## Priority Rules

Future experiments should assign the first matching terminal label in this order:

1. `invalid_simulation`
2. `overspeed`
3. `instability`
4. `resource_depletion`
5. `unsafe_state`
6. `success`
7. `recoverable_crossing_failed_late`
8. `crossing_unrecoverable`
9. `no_crossing`
10. `timeout`
11. `unknown`

Rationale:

- Validity failures outrank all scientific interpretation because the rollout cannot be trusted.
- Explicit safety failures outrank performance labels because unsafe progress is not accepted recoverability progress.
- `success` appears before non-success performance labels only after validity, safety, and resource checks have cleared.
- Recoverability-specific crossing failures are more informative than generic timeout.
- `no_crossing` outranks `timeout` when the artifact clearly shows the target-radius event never happened.
- `unknown` is last because it should be used only when the artifact cannot support a more precise mechanism.

If a future experiment has a declared resource budget that can be exhausted before timeout, `resource_depletion` should remain above `timeout` because it identifies a more specific mechanism.

## Exact Label Definitions

### `invalid_simulation`

Use when the simulator output became numerically or structurally invalid.

Examples include NaN or infinite state values, missing required output fields, non-monotonic time, corrupted artifact rows, impossible state dimensions, or a simulator exception that prevents reliable interpretation.

This label means the rollout is not scientifically interpretable without rerun or repair.

### `overspeed`

Use when the rollout violates the experiment's declared speed threshold.

This label should be assigned even if the rollout also crossed the target radius or entered a recoverable-looking region. A crossing that depends on overspeed is not clean benchmark progress.

Current evidence example: Phase34, Phase36B, Phase37A, and Phase37B protected summaries report `0` overspeed for the guarded results. The label is still required because future experiments must continue reporting it.

### `instability`

Use when the closed-loop behavior violates the experiment's declared instability criterion.

Examples include divergent trajectories, unacceptable oscillations, controller blow-up, repeated mode thrashing, or other instability flags defined before running the experiment.

Current evidence example: Phase36B, Phase37A, and Phase37B protected results report `0` instability. This supports keeping instability separate from crossing and recoverability rather than folding it into generic failure.

### `resource_depletion`

Use when a rollout fails because a declared consumable or control budget is exhausted before success.

Examples include fuel budget depletion, impulse budget depletion, maximum allowed control effort, battery or compute budget in future experiments, or another predeclared resource limit.

Current Phase34/36/37 artifacts do not provide a protected resource-depletion count. Until a future schema records a stable budget field, use `unknown` or another supported label rather than inventing this label from control effort alone.

### `unsafe_state`

Use when the rollout enters a declared unsafe state that is not already covered by `overspeed`, `instability`, or `invalid_simulation`.

Examples include collision-region entry, forbidden radius bands, keep-out zone violation, unrecoverable attitude or orbital-state constraint violation, or a benchmark-defined safety envelope breach.

Current evidence example: Phase37B's summary classifies the weak tangential diagnostic as `unsafe globally` because it did not preserve the regression crossing set. That phrase is a benchmark decision about a controller variant, not direct evidence that each row should be relabeled `unsafe_state`.

### `success`

Use when the rollout satisfies the declared success criteria for the current benchmark version without higher-priority validity, safety, or resource failures.

For recoverability benchmarks, success must not rely only on legacy simulator `CAPTURE`, `LOCK`, or `success` labels unless the experiment has explicitly defined those labels as sufficient. A future recoverability success should normally require the relevant crossing and recoverability fields to agree with the declared benchmark contract.

Current evidence example: Phase34 `radius_priority` preserves `8 / 24` crossings and produces `8 / 24` recoverable crossings with `0` overspeed. That is the current clean recoverability baseline for crossing-producing cases, not proof that upstream crossing generation is solved.

### `recoverable_crossing_failed_late`

Use when the rollout crossed the target radius, reached the declared recoverable-crossing condition, and then failed before final benchmark success.

This label distinguishes "the crossing was usable, but the later policy or terminal phase failed" from "the crossing was never recoverable."

Current evidence example: The protected Phase34/36/37 aggregate summaries do not establish a clear count of this mechanism. Future experiments should use this label when the artifact records both `recoverable_crossing=True` and later non-success without a higher-priority failure.

### `crossing_unrecoverable`

Use when the rollout crossed the target radius but did not reach the declared recoverable-crossing condition before termination, and no higher-priority label applies.

This label captures the Week 1 distinction that crossing is not insertion.

Current evidence example: The reduced Phase31-style baseline in Phase34 produced `8 / 24` target-radius crossings and `0 / 24` recoverable crossings. That evidence supports the mechanism, even though historical rows should not be rewritten under this taxonomy.

### `no_crossing`

Use when the rollout never crossed the target radius and no higher-priority label applies.

This label should be used even if closest approach or crossing potential improved. Closest approach is diagnostic; it is not crossing.

Current evidence examples:

- Phase36B `baseline_phase34` had `16 / 24` non-crossing cases.
- Phase36C diagnoses those baseline non-crossing cases as `8` `near_crossing` and `8` `over_conservative_transfer`.
- Phase37A created `0` new crossings on baseline non-crossing cases.
- Phase37B selected non-crossing cases produced `0 / 4` selected-case crossings under weak tangential shaping.

### `timeout`

Use when a rollout reaches the configured step, wall-clock, or horizon limit and the artifact does not support a more specific non-success mechanism.

If the artifact clearly shows no target-radius crossing, prefer `no_crossing` over `timeout`. If the artifact clearly shows an invalid state, overspeed, instability, resource depletion, unsafe state, or crossing/recoverability mechanism, use that more specific label.

Current evidence example: Historical rows often record `termination_reason=max_steps`. Under this taxonomy, that field alone is not enough to force `timeout` if the artifact also supports `no_crossing` or another more specific label.

### `unknown`

Use when available artifacts cannot support a more precise label.

Use `unknown` rather than guessing when:

- required fields are missing,
- current and historical fields conflict,
- a label would depend on unstated thresholds,
- a subset diagnostic cannot be mapped to a full-benchmark mechanism,
- a row records only a proxy metric such as closest approach.

This label is not a failure of the taxonomy. It prevents fake precision.

## Evidence Examples From Current Phase34/36/37 Artifacts

The examples below are interpretive guidance for future labeling. They do not modify historical evidence.

| Evidence | Supported future label use | Notes |
| --- | --- | --- |
| Phase34 `radius_priority`: `8 / 24` crossings, `8 / 24` recoverable crossings, `0` overspeed | `success` for clean recoverability cases if future success criteria match | Does not expand upstream crossing generation. |
| Phase34 reduced Phase31-style baseline: `8 / 24` crossings, `0 / 24` recoverable crossings | `crossing_unrecoverable` mechanism | Legacy simulator success fields must not be treated as recoverability success without a schema bridge. |
| Phase36B all transfer families: `8 / 24` crossings and `8 / 24` recoverable crossings, `0` overspeed, `0` instability | `success` for known crossing-producing cases; `no_crossing` for unresolved non-crossing cases | No family beat Phase34 on full benchmark crossing or recoverability counts. |
| Phase36C baseline non-crossing set: `16` cases split into `near_crossing=8` and `over_conservative_transfer=8` | `no_crossing` terminal label with diagnostic labels | `near_crossing` and `over_conservative_transfer` are diagnostic, not terminal taxonomy labels. |
| Phase37A: `0` new crossings on baseline non-crossing cases, `0` overspeed, `0` instability | `no_crossing` for baseline non-crossing rows without new crossing | Delayed radial commitment preserved some known behavior but did not solve the bottleneck. |
| Phase37B weak selected cases: `0 / 4` selected crossings; weak regression cases: `4 / 8` crossings and `4 / 8` recoverable crossings | selected rows support `no_crossing`; damaged regression preservation requires manual audit | The subset result is diagnostic and not accepted benchmark progress. |

## Ambiguous-Case Handling

Use the most specific supported label, not the most optimistic label.

Rules:

- If safety or validity fields are missing, do not assume they passed.
- If a row has `crossing_occurs=True` but lacks recoverability fields, do not infer recoverability.
- If a row has `termination_reason=max_steps` and `crossing_occurs=False`, prefer `no_crossing` over `timeout`.
- If a row has `simulator_success_label=True` but `recoverable_crossing=False`, do not automatically label it `success` for recoverability benchmarking.
- If a row appears in a diagnostic subset, label the subset status separately and avoid full-benchmark conclusions.
- If closest approach improves without crossing, use `no_crossing` plus a diagnostic label, not `success`.
- If the artifact cannot distinguish `crossing_unrecoverable` from `recoverable_crossing_failed_late`, use `unknown` and explain the gap in `manual_audit_note`.

## Future CSV And JSON Output Guidance

This section is guidance for Week 3 result-schema work. It does not implement that schema.

Future CSV rows should include at least:

- `terminal_label`
- `precursor_labels`
- `diagnostic_labels`
- `manual_audit_note`
- `label_taxonomy_version`

Recommended values:

- `label_taxonomy_version`: `failure_label_taxonomy_v0`
- `terminal_label`: one controlled label from this document
- `precursor_labels`: semicolon-separated list, empty when none apply
- `diagnostic_labels`: semicolon-separated list, empty when none apply
- `manual_audit_note`: short free-text note, empty when no audit note is needed

Example CSV fragment:

```csv
terminal_label,precursor_labels,diagnostic_labels,manual_audit_note,label_taxonomy_version
no_crossing,,near_crossing,"closest approach improved but target radius was not crossed",failure_label_taxonomy_v0
crossing_unrecoverable,crossed_target_radius,,crossed target radius but did not enter recoverable basin,failure_label_taxonomy_v0
overspeed,crossed_target_radius,,overspeed outranks crossing event,failure_label_taxonomy_v0
```

Future JSON output should prefer arrays:

```json
{
  "terminal_label": "no_crossing",
  "precursor_labels": [],
  "diagnostic_labels": ["near_crossing"],
  "manual_audit_note": "Closest approach improved but target radius was not crossed.",
  "label_taxonomy_version": "failure_label_taxonomy_v0"
}
```

Future outputs may keep compatibility fields such as `dominant_failure_label`, but they should not replace `terminal_label`.

## What Labels Should Not Imply

These labels do not imply:

- real spacecraft validation,
- hardware readiness,
- sim-to-real transfer,
- docking readiness,
- general orbital insertion success,
- that Phase34 solves upstream crossing generation,
- that a subset result is a full-benchmark win,
- that closest approach is success,
- that target-radius crossing is recoverability,
- that simulator `CAPTURE`, `LOCK`, or legacy `success` is mission success,
- that absence of an overspeed flag proves all safety constraints were evaluated,
- that `unknown` means the rollout was unimportant.

## Week 3 Handoff Questions For `result_schema_v1.md`

Week 3 should answer:

- Should the future normalized field be named `terminal_label`, `dominant_failure_label`, or both?
- What exact compatibility rule maps historical `dominant_failure_label` and `failure_label` into the new taxonomy, if any?
- Which fields are required before assigning `success` under recoverability benchmarking?
- Should `crossing_unrecoverable` and `recoverable_crossing_failed_late` require explicit `first_crossing_step` and `recoverable_crossing` fields?
- How should JSON arrays be represented in CSV without breaking simple readers?
- Should `unknown` require a non-empty `manual_audit_note`?
- Should `near_crossing` and `over_conservative_transfer` remain diagnostic labels only, or should a future taxonomy add a separate controlled diagnostic-label list?
- How should subset membership and regression-set membership be represented next to terminal labels?
- What minimal resource-budget fields are needed before `resource_depletion` can be assigned reliably?
- What small validation script should reject labels outside this controlled set in future outputs?

## Week 2 Completion Rule

Week 2 is complete when this document exists, the protected regression guard still passes, and no historical evidence has been modified.
