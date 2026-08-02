# Recovery Branch-State Registry v0

Status: Registry infrastructure implemented; final member artifacts not yet generated.

Completed: 2026-08-02

## Purpose

The registry replaces a single hard-coded executable source state with a member-ID based, provenance-bound collection while preserving the legacy canonical path.

## Registry architecture

The published manifest indexes one external legacy artifact and registry-local generated artifacts. Each member binds its case, source configuration, simulator configuration, constants, transition implementation, nominal controller, prefix count, branch step, canonical state hash, and raw artifact hash.

## Member identity

Member IDs and case IDs are unique and deterministically ordered. A member identifies one state artifact only; artifact substitution and duplicate identities are invalid.

## External legacy member

`legacy_canonical` resolves only to `analysis/recovery_action_branching_nonformal_v0/branch_state.json`. The registry does not rewrite, normalize, or mirror that artifact.

## Generated members

Generated members use `recovery_branch_state_registry_member_v0`. Their physical origin is deterministic execution of an existing frozen nominal controller for exactly 27 realized transitions, followed by observation before the step-28 action.

## Immutable loading

The loader returns an immutable wrapper whose serialized document can be copied for internal execution. It validates registry identity, deterministic ordering, member identity, path scope, raw bytes, canonical payload, source case, and source configuration before returning.

## Hash verification

Scientific payload hashes exclude the fixed generation date and self-hash fields. Raw file hashes remain distinct from canonical scientific hashes. Mutation of either a state or registry manifest invalidates loading.

## Path safety

Callers supply a registry member ID, never an arbitrary artifact path. Legacy paths are exact; generated paths must resolve below the registry `branch_states` directory. Absolute paths, traversal, symbolic artifacts, and path substitution are rejected.

## Executor integration

`execute_recovery_branch` remains the legacy-canonical mapping API. `execute_registered_recovery_branch` is a separate member-ID API that loads and validates the registry before entering the existing one-step action, Final Veto, and transition core.

## Backward compatibility

No registry argument is required by existing callers. The canonical schema, case restriction, default path, action laws, threshold, monitor behavior, transition function, and terminal semantics remain unchanged.

## Provenance

Case definitions originate from the frozen Final Veto manifest. Initialization and nominal actions originate from the existing Phase34 or Phase35 rollout selected by that manifest. Generated states are not reconstructed from result logs.

## Publication

Generation validates all in-memory artifacts, stages the complete directory in a sibling path, validates staged bytes, and publishes with one same-filesystem directory rename. Existing and protected targets are rejected.

## Scientific boundary

This registry freezes executable experiment inputs. It does not execute a recovery branch, implement a new controller, demonstrate recovery performance, validate staged phase logic, or support formal safety claims.
