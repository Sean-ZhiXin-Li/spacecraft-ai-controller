# WHPL_05 — Idempotent Logging & De-dup Guardrails

**Date**: 2026-02-XX
**Type**: Single-issue engineering fix
**Scope**: CSV logging only (no controller / reward / experiment changes)

---

## Problem

During WHPL_03–04, the ablation CSV accumulated duplicated and structurally corrupted rows due to non-idempotent append behavior. Re-running the same command could silently add duplicate entries, making it impossible to guarantee that each row corresponds to a unique engineering run.

This breaks a core assumption required for later 2D analysis:
**one row = one unique, reproducible experiment.**

---

## Fix: Idempotent Logging via De-dup Key

A minimal, explicit de-duplication guardrail was introduced at the CSV write boundary.

Each row is assigned a stable `dedup_key`, computed from invariant identifiers of a logical run:

* controller name
* effective scenario
* thrust_newton
* r0_over_target
* target_radius
* git commit short SHA

The CSV is now append-only but idempotent:

* If the same `dedup_key` already exists, the write is skipped with an explicit status.
* If the key is new, the row is appended.
* No controller logic, reward function, or experiment parameters were modified.

---

## Verification

The same command was executed twice consecutively:

```
python src/quick_compare_v3_v4.py
```

Observed behavior:

* First run: exactly one row appended.
* Second run: duplicate detected and blocked (`csv_status=skip`).
* CSV remained unchanged after the second run.

---

## Data Cleanup

Historical rows produced before the de-dup guardrail were audited.
Only rows with:

* non-empty `dedup_key`
* finite numerical metrics

were considered valid.

All legacy polluted rows were removed. The cleaned dataset now contains exactly one verified row, corresponding to a single trusted engineering run.

---

## Conclusion

CSV logging is now idempotent.
Repeated executions no longer corrupt the dataset, and each row corresponds to a unique, verifiable engineering event.

This establishes a reliable data foundation for subsequent 2D analysis.
