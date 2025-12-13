# Expert Controller (Improved Version)

## Overview
This controller is a hand-crafted expert designed to stabilize a spacecraft
near a target circular orbit in a 2D Newtonian gravity environment.
It serves as a deterministic baseline and as a teacher for future imitation learning.

## Control Objective
- Reach and maintain a circular orbit with radius `target_radius`
- Minimize radial error, velocity mismatch, and oscillations
- Avoid aggressive thrusting and numerical instability

## Control Structure

### Radial Control
The controller monitors the radial distance error relative to the target orbit.
Radial thrust is applied to correct inward or outward drift.

### Tangential Control
Tangential thrust adjusts orbital speed toward the circular-orbit reference
velocity computed from the gravitational parameter.

### Damping and Smoothing
Compared to earlier versions, the improved controller introduces:
- Reduced thrust jitter
- Smoother transitions near the target orbit
- Conservative thrust cutoffs inside the tolerance window

## Optional Modules
Some control components are implemented but currently disabled:
- Scheduling logic for phase-dependent gains
- Alignment logic for future trajectory shaping

These modules are reserved for future extensions.

## Thrust Termination Logic
When the spacecraft remains inside the tolerance window for a sufficient
number of steps, thrust output is reduced or fully stopped to maintain stability.

## Evaluation Protocol
The controller is evaluated using fixed scenario comparisons
(e.g., default and spiral starts) and metrics including:
- Total reward
- Average radius error
- Jitter magnitude
- Success consistency

This expert controller provides a stable and interpretable reference
for subsequent imitation learning and reinforcement learning experiments.
