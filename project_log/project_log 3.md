# Day 3 - Project Log

**Date**: 2025-07-18  
**Today's Goals**:
- [x] Integrated continuous + impulse thruster controller (`smart_combined_controller`)
- [x] Added thrust decay mechanism (supports linear and exponential modes)
- [x] Performed orbit stability testing and parameter tuning (adjusted α, β, decay_rate)
- [x] Analyzed orbit deviation and target orbit fitting accuracy
- [ ] Attempted animated output (abandoned midway)

---

**Key Progress**:
1. **Controller Fusion**:  
   Combined radial and tangential thrust controllers with support for both continuous and impulse modes.

2. **Thrust Decay Mechanism**:  
   - Command parameters: `thrust_decay_type=('none'|'linear'|'exponential')`, `decay_rate`  
   - Implemented formulas:
     - Linear: `decay_factor = max(1 - decay_rate * t, 0)`
     - Exponential: `decay_factor = exp(-decay_rate * t)`

3. **Main Trajectory Simulation Result**:  
   - Successfully reached the expected target orbit range.
   - Thrust decay helps mitigate over-acceleration.

4. **Debugging Notes**:
   - Animation saving error: `FuncAnimation RuntimeError` and `ffmpeg unavailable`
   - Fix attempts failed; reverted to static trajectory plots.

---

**Parameter Settings**
```python
alpha = 0.1
beta = 0.05
impulse = True
impulse_period = 5.0
impulse_duration = 1.0
thrust_decay_type = 'exponential'
decay_rate = 1e-6
```

---

**Visualization**
- `plot_trajectory()` successfully shows controlled trajectory vs baseline
- Main trajectory approaches the target orbit with appropriate control direction
- Combining radial and tangential control significantly improves correction ability

---

**Issues**
- `FuncAnimation` throws an error when generating `.gif`: `x must be a sequence` → caused by incorrect data structure passed to `update()`
- `ffmpeg` not recognized by matplotlib despite installation → likely due to PATH misconfiguration or missing matplotlib dependencies
