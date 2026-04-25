# Phase 7 Pre-Window Trajectory Shaping Summary

## Setup

- Scope: 2D Python-only pre-window shaping before WS activation.
- Grid: same 270 regimes from Phase 6.5-6.7.
- WS band: `8e-5` relative radius error.
- Pre-window band: `8e-5` to `5e-4` relative radius error.
- Outside the pre-window band, DESCENT remains baseline retrograde.
- Inside the WS band, the same Phase 6.7 WS candidate structure and CAPTURE/LOCK logic are used.

## Ranking Result

- Original WS-1: success `157`, CAPTURE `157`, near-miss `42`.
- Adaptive soft reference: success `172`, CAPTURE `172`, near-miss `44`.
- Best pre-window variant: `prewindow_radial_medium` with success `209`, CAPTURE `209`, near-miss `56`.
- Best overall by ranking: `prewindow_radial_medium`.

## Answers

1. Does pre-window shaping improve over adaptive_soft? `yes`. Best pre-window delta vs adaptive_soft: success `+37`, CAPTURE `+37`, near-miss `+12`, mean minimum radius error `-1.092e+08`.
2. Does it increase success count or only reduce near-misses? It increases strict success by `+37`.
3. Which shaping strategy works best? `prewindow_radial_medium` by success count, CAPTURE count, near-miss count, and mean minimum radius error.
4. Does it widen the capture window or shift it? It appears to `widen` the success set relative to adaptive_soft: gained `44`, lost `7`, retained `165`, changed memberships `51`.
5. Best next 2D Python-only step: keep the same evaluator and run a small parameter refinement around the best pre-window family, using gained/lost membership vs adaptive_soft as the primary diagnostic before changing any CAPTURE/LOCK logic.

## Caution

This phase changes only DESCENT action selection in the pre-window band. It does not alter physics, PPO, learned experiments, CAPTURE/LOCK equations, or Phase 3-6.7 output directories.