# WHPL_11 — Controller Variant Tracking + Dedup Integrity

Date: 2026-02-16
Environment: spacecraft (conda)
File modified: src/quick_compare_v3_v4.py

---

# Objective

Ensure different controller structure variants are preserved in the 2D experiment table instead of being merged by dedup collision.

Core goal:

> Same (thrust, difficulty, r0) coordinate must allow multiple controller structure branches to coexist in CSV.

No changes to:

* controller logic
* physics
* reward
* parameters
* scenario design

Pure experiment-structure upgrade only.

---

# Problem Identified

Observed behavior before WHPL_11:

```
csv_status=skip
```

Even after modifying controller structure (e.g., r_full scaling changes),
CSV refused to append new rows.

Root cause:

`dedup_key` did not include any controller structure lineage.

Therefore:

Different controller variants at the same coordinate
(thrust × difficulty × r0)
collapsed into the same dedup_key.

This invalidates 2D analysis because structural forks become invisible.

---

# Implementation

## A) Add Explicit Controller Variant Tag

Added global constant:

```python
CONTROLLER_VARIANT = "whpl11_variant_tracking"
```

Injected into CSV row:

```python
"controller_variant": CONTROLLER_VARIANT
```

Added `controller_variant` to FIELDNAMES so it becomes a real column.

No dynamic detection. No automation.
Explicit lineage only.

---

## B) Include Variant in dedup_key

Updated `_compute_dedup_key()` parts:

Before:

* controller
* scenario
* thrust_newton
* r0_over_target
* target_r
* git_sha

After:

```python
row.get("controller_variant", "legacy")
```

is included in key computation.

Effect:

Same coordinate + different controller_variant
→ different dedup_key
→ separate CSV row

Backward compatibility preserved:
Old rows have empty controller_variant column.

---

# Execution Evidence

Console output:

```
csv_status=upgraded_append
```

CSV header now contains:

```
...,dedup_key,controller_variant
```

Old row example:

```
...,9d8ec62d,
```

New WHPL_11 row:

```
...,0fb55740,whpl11_variant_tracking
```

This confirms:

* Schema upgraded safely
* No legacy corruption
* Variant now affects dedup
* Lineage preserved

---

# Structural Impact

Experiment space upgraded from:

2D:
(thrust × difficulty × r0)

To:

3D:
(thrust × difficulty × r0) × controller_variant

This enables:

* Same-coordinate structural comparisons
* Variant-controlled heatmaps
* Proper branch evolution tracking

Without WHPL_11, all future structural improvements would silently merge.

---

# Conclusion

WHPL_11 is a structural experiment-layer upgrade.

It does not improve performance.
It does not change physics.
It does not alter reward.

It makes future results scientifically valid.

Experiment lineage is now preserved.

2D conclusions before April are now structurally trustworthy.

---

End of WHPL_11.
