# Project Log - Spacecraft AI Controller

## Day 1 - 2025-07-16

###  What I did
- Initialized the project structure under `spacecraft_ai_project`
- Wrote a basic 2D orbit simulator (`main.py`) using Newtonian gravity
- Visualized the spacecraft trajectory using `matplotlib`
- Refactored the simulation logic into a reusable function `simulate_orbit()` under `simulate_orbit.py`
- Created `.gitignore` and `LICENSE` files
- Initialized a local Git repository and pushed the project to GitHub
- Resolved the first merge conflict caused by a GitHub pull (via terminal and Vim)

###  Problems I encountered
- First `git push` was rejected due to non-fast-forward error; resolved by pulling and merging
- Vim editor opened during merge commit; didn’t know how to exit (later learned `Esc → :wq`)
- Confused about how to organize simulation logic into modular files (later separated into a simulator package)
- **Arrow (quiver) visualization too small** on the trajectory plot; tried adjusting `scale` and `width`, parameters to make it clearly visible

###  Plans for tomorrow (Day 2)
- Implement a switchable thruster model: impulse vs continuous thrust
- Try simulating with real NASA trajectory parameters (e.g., Voyager)
- Explore AI-based thrust direction control (e.g., mimic expert strategies)