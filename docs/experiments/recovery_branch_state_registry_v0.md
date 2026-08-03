# Recovery Branch-State Registry v0 Generation

Status: Case-specific frozen generation protocol defined; registry generation not yet executed.

Completed: 2026-08-03

## Source inventory

The source inventory is the 13-case Final Veto manifest. Eligibility requires a complete frozen initialization, supported existing Phase34 or Phase35 nominal controller, simulator configuration, source configuration hash, seed, transition implementation, and controller-source hash. A case ID alone is insufficient.

## Boundary contract

No common transition count is used. The canonical member preserves its published transition-27 boundary. Every other case uses the last valid pre-transition state before its frozen monitor-off terminal transition. Counts and reasons are fixed in `configs/recovery_branch_boundary_registry_v0.json` from checked-in evidence before generation.

`PreTransitionActionContext(step=N)` contains the state after `N-1` realized transitions. The transition-22 overspeed case therefore branches from the step-22 pre-transition state after 21 realized transitions; transition 22 is executed only to reproduce the frozen terminal evidence.

## Generation command

```powershell
python scripts/generate_recovery_branch_state_registry_v0.py --execute-frozen-registry-generation
```

The command permits no case, seed, boundary, state, threshold, output, retry, or resume override.

## Discovery procedure

The command freshly reproduces the legacy canonical state, executes each eligible noncanonical case to its own frozen boundary and terminal-validation point, computes branch-state and selection metrics, then selects three distinct generated members.

## Selection

Member A is the legacy canonical state. Member B is the maximum predicted speed ratio at or below `1.90`. Member C is the minimum predicted speed ratio above `1.90`. Member D is the largest absolute tangential velocity error ratio among remaining eligible cases. Frozen lexical and hash tie-breaks apply.

Metrics are measured at different source transition counts. The registry is a multi-boundary calibration registry, not a synchronized-time experiment.

## Determinism

Every selected generated case is initialized again from frozen inputs. Discovery and reproduction must match in boundary type and count, terminal metadata, Cartesian state, derived state, prediction, action-trace hash, state-trace hash, transition count, branch step, and canonical payload hash. The legacy complete canonical document must reproduce exactly.

## Execution count

When all 13 cases are eligible, the command performs one canonical reproduction, 12 noncanonical discoveries, and three selected reproductions: 16 nominal-prefix executions. It performs no recovery branch transition and retries nothing.

## Publication boundary

Only the three selected generated state artifacts are published. Unselected discovery executions contribute hashes and metrics to reports but do not publish complete Cartesian state files. The existing canonical artifact remains external and byte-identical.

## Non-claims

Registry generation is deterministic input preparation, not a recovery experiment, branch comparison, staged-controller run, policy-improvement result, formal safety result, or deployment result.
