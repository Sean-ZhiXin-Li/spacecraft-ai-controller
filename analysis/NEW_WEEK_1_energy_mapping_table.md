\# NEW\_WEEK\_1 — Geometry ↔ Energy Mapping (Spiral-In)



| Geometry Phenomenon | Energy / L Metric | Physical Interpretation |

|----------------------|------------------|--------------------------|

| \*\*Slow convergence\*\* | large `energy\_convergence\_step`, low `energy\_convergence\_ratio\_median` | Thrust contributes insufficient energy per step, leading to slow convergence in total orbital energy. |

| \*\*Weak thrust\*\* | low `thrust\_energy\_ratio` | Indicates low energy gain per thrust impulse; propulsion underpowered relative to energy deficit. |

| \*\*Oscillation\*\* | high `energy\_oscillation\_index` | Frequent ΔE sign reversals imply unstable or alternating thrust direction. |

| \*\*Cut-in error\*\* | nonzero `angular\_momentum\_error\_final` | Thrust vector misaligned with desired orbital entry direction, causing angular momentum mismatch. |

| \*\*Long-term drift\*\* | nonzero `energy\_drift\_percent` | Persistent deviation from the target energy shell; indicates insufficient energy control precision. |

| \*\*Fast stabilization\*\* | small `energy\_convergence\_step`, high `η` | Energy rapidly approaches target value with accurate and efficient thrust alignment. |



