# Phase 33 Architecture Gap Map

## Phase 7.6

- had: phase-structured pre-window shaping, window seeking, CAPTURE, and LOCK.
- lacked: continuous trajectory-wide optimization of vt/vr/r synchronization.
- lacked: explicit permission to let vt error temporarily grow if it improves late recoverability.

## Phase 20

- had: predictive local candidate scoring.
- lacked: global state-control trajectory optimization.
- lacked: a mechanism for smooth long-horizon control motifs.

## Phase 31

- had: named global transfer families with burn/coast schedules.
- lacked: continuous adjustment of every control interval.
- lacked: the low-thrust smooth arc discovered by Phase 32.

## Phase 32

- introduced: finite-horizon direct control optimization against recoverability.
- introduced: smooth low-authority steering as an upper-bound behavior.
- introduced: evidence that recoverability can be reached as a state and, in one representative case, as a trajectory that also crosses target radius.
- did not prove that the first radius-crossing state itself is recoverable.