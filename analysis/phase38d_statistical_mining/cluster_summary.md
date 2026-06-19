# Phase38D Cluster Summary

Scope: qualitative clustering from recorded feature patterns. No clustering model was fit.

## Cluster A - Preserved Crossing-Producing Cases

Signature:

- actual target-radius crossing exists;
- crossing-state metrics are present;
- recoverable crossing is available under fixed Phase34 downstream behavior;
- associated with the middle radius-ratio regime in the reduced benchmark.

Interpretation:

These cases are not the current bottleneck. They are regression guards for future work.

## Cluster B - Near-Crossing Failures

Signature:

- non-crossing rows with `near_crossing` label;
- lower radius-ratio side in Phase36C baseline labels;
- relatively higher crossing potential than over-conservative-transfer rows;
- often late or max-step closest-approach behavior.

Interpretation:

These cases appear closer to crossing in diagnostic metrics but still do not cross. They require care because proxy improvement is not success.

## Cluster C - Over-Conservative-Transfer Failures

Signature:

- non-crossing rows with `over_conservative_transfer` label;
- higher radius-ratio side in Phase36C baseline labels;
- early closest approach in baseline rows;
- lower crossing potential than near-crossing rows.

Interpretation:

This class motivated Phase37B selected cases, but Phase37B did not create crossings and damaged regression preservation. It remains a useful analysis cluster, not an approved implementation target.

## Cluster D - Proxy-Improvement Without Crossing

Signature:

- closest-approach or crossing-potential movement;
- no target-radius crossing;
- possible regression damage.

Interpretation:

This is the most important anti-pattern from Phase36C and Phase37B. Phase39 must not be approved if its only expected evidence is proxy movement.
