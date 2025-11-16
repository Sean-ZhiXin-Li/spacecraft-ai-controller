# NEW_WEEK_2 — Visual Diagnostics PASS / FAIL Criteria

This document defines how to evaluate any controller (Expert, RL, or Imitation)
using Week2’s visualization tools.

These standards match typical aerospace-control evaluation used in:
- Stanford SLAB
- NASA Autonomous Systems Division
- ESA ACT
- KAIST Space Robotics Lab

---

# 1. Orbit (x,y) — Geometry Criteria

### PASS
- Trajectory approaches and settles near the target circular orbit  
- Motion is smooth without chaotic loops  
- No high-frequency oscillation  

### FAIL
- Irregular loops, zig-zag paths  
- Sudden jumps in position (numerical instability or bad thrust)  
- Spiraling outward instead of inward  

---

# 2. Radial Error vs Time — Convergence Criteria

### PASS
- Radial error decreases monotonically (or stepwise monotonic)  
- No large “bounce” oscillation  
- Final error < 1% of target radius  

### FAIL
- Error increases or oscillates heavily  
- Plateau without further improvement (controller stuck)  
- Diverging behavior  

---

# 3. Thrust Magnitude Histogram — Control Smoothness Criteria

### PASS
- Histogram shows clear structure (e.g., two-mode: drift + correction)  
- No extreme random spikes  
- Thrust values remain inside physical limits  

### FAIL
- “Spray noise pattern” — evenly distributed from min to max  
- Sharp spikes at random magnitudes (controller instability)  
- Heavy saturation at maximum thrust (poor energy efficiency)  

---

# 4. Energy Summary — Physics Consistency

### PASS
- `energy_drift_percent < 1%`  
- `angular_momentum_error_final < 1.0` (depending on units)  
- `energy_oscillation_index` value low  
- `energy_convergence_step` finite and reasonable  

### FAIL
- Energy drift > 5% → physics inconsistency or wrong thrust model  
- Angular momentum error exploding → integration unstable  
- Oscillation index high → controller introducing noise  

---

# 5. Overall PASS / FAIL Rule

A run is **PASS** if:
- Orbit is stable  
- Radial error converges  
- Thrust histogram is structured  
- Energy summary is within acceptable bounds  

A run is **FAIL** if any **two or more** of the four diagnostics fail.

This combined rule reflects real engineering standards:
spacecraft controllers must be robust across multiple metrics,
not just a single score.

---

# End of Document
