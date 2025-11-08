\# NEW\_WEEK\_1 — Spiral-In Energy Analysis (Energy View)



\## Motivation



In \*\*Week0\*\*, we built quantitative metrics to \*measure\* trajectory behavior.  

In \*\*NEW\_WEEK\_1\*\*, the goal is to \*explain why\* some spiral-in trajectories converge slowly —  

by analyzing \*\*energy and angular momentum dynamics\*\* instead of geometry alone.



> Why can a trajectory that looks “geometrically correct” still fail to converge in energy?



---



\## Theoretical Foundation



For a spacecraft under central gravity:



\\\[

E\_t = \\frac{1}{2} m v\_t^2 - \\frac{GMm}{r\_t}

\\]

\\\[

L\_t = m (x\_t v\_{y,t} - y\_t v\_{x,t})

\\]



Target circular orbit:

\\\[

E^\* = -\\frac{GMm}{2r^\*}, \\quad

L^\* = m r^\* \\sqrt{\\frac{GM}{r^\*}}

\\]



Energy convergence efficiency:

\\\[

η\_t = \\frac{|\\Delta E\_t|}{|\\Delta E\_{\\text{required}}|}

\\]

where \\(\\Delta E\_t = E\_{t+1} - E\_t\\) and \\(\\Delta E\_{\\text{required}} = E^\* - E\_t\\).



---



\## Implementation Overview



\- \*\*Module:\*\* `tools/metrics/energy\_view.py`

\- \*\*Input:\*\* `logs/new\_week\_1/spiral\_in/high\_thrust/replay.npz`

\- \*\*Output:\*\* `logs/new\_week\_1/spiral\_in/high\_thrust/metrics\_energy.json`



\*\*Metrics computed:\*\*



\- `energy\_convergence\_step` — first step where relative |E − E\*| < ε  

\- `energy\_drift\_percent` — steady-state energy bias relative to |E\*|  

\- `angular\_momentum\_error\_final` — final L deviation normalized by |L\*|  

\- `energy\_oscillation\_index` — frequency of ΔE sign flips  

\- `thrust\_energy\_ratio` — experimental thrust-to-energy efficiency indicator  

\- `energy\_convergence\_ratio\_median` — median of η\_t (energy convergence efficiency)



---



\## Geometry–Energy Mapping



Reference: `analysis/NEW\_WEEK\_1\_energy\_mapping\_table.md`



| Geometry Phenomenon | Energy Metric | Interpretation |

|----------------------|----------------|----------------|

| Slow convergence | `energy\_convergence\_step`, `η\_median` | Thrust adds too little energy per step |

| Weak thrust | `thrust\_energy\_ratio` | Low energy gain per thrust unit |

| Oscillation | `energy\_oscillation\_index` | ΔE sign flips frequently (unstable thrust direction) |

| Cut-in error | `angular\_momentum\_error\_final` | Thrust misalignment during orbit entry |

| Long-term drift | `energy\_drift\_percent` | Persistent offset from target energy shell |



---



\## Results — NEW\_WEEK\_1 High Thrust Baseline



From `logs/new\_week\_1/spiral\_in/high\_thrust/metrics\_energy.json`:



| Metric | Value | Interpretation |

|---------|--------|----------------|

| \*\*energy\_convergence\_step\*\* | 7 | Reaches ±5% energy band within 7 steps |

| \*\*energy\_drift\_percent\*\* | 0.389 | Final energy remains 38.9% above target |

| \*\*angular\_momentum\_error\_final\*\* | 0.41 | ~40% L deviation, misaligned entry |

| \*\*energy\_oscillation\_index\*\* | 0.005 | Almost monotonic energy change |

| \*\*thrust\_energy\_ratio\*\* | 2.33 | Strong energy response (incl. gravity) |

| \*\*energy\_convergence\_ratio\_median\*\* | 0.028 | ~2.8% of remaining gap closed per step |



\*\*Summary:\*\*



The controller quickly touches the target energy shell (fast initial response),

but fails to remain on it — the trajectory stabilizes on a higher-energy, misaligned orbit.



> The spacecraft \*spirals correctly, but not efficiently\*.



---



\## Acceptance



\- `tools/metrics/energy\_view.py` runs without errors  

\- `logs/new\_week\_1/spiral\_in/high\_thrust/metrics\_energy.json` generated  

\- `analysis/NEW\_WEEK\_1\_energy\_mapping\_table.md` completed  

\- This file (`NEW\_WEEK-PROJECT\_LOG\_01.md`) marks the completion of NEW\_WEEK\_1 analysis stage



