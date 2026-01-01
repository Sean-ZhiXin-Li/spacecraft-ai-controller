# 2025 Annual Summary — Spacecraft AI Controller

*Written for future advisors, and for my future self*

**Date:** Beijing Time, January 1, 2026  
**Scope:** A complete reflection on my 2025 work around spacecraft AI control, engineering systems, and research judgment

---

## 1. What Was I Actually Doing in 2025?

If I had to summarize 2025 in a single sentence:

> **2025 was the first year in which I tried to build independent engineering judgment under strong real-world constraints and heavy AI assistance.**

I was not simply “building an AI controller.” I was learning how to answer a more fundamental question:

> **When AI stops giving me immediate answers, do I still know how to move forward?**

---

## 2. First Time Anchor — July 16, 2025: The First Script

The true starting point of this project is precise.

On **July 16, 2025**, I wrote the very first script of what later became the *Spacecraft AI Controller* project: a 2D Newtonian gravity orbital simulator.

At that moment, the goal was intentionally minimal:

- A single point mass  
- A central gravitational body  
- A stable orbit

This was the physical origin of the entire project.

At the time, I believed that once the simulation was correct and AI was connected, the problem would naturally be “solved.”

In hindsight, that belief was naive — but necessary. It pushed me into the real problem space quickly, instead of keeping me at the level of abstractions.

---

## 3. Second Time Anchor — Late July 2025: The First Real Failure

The first true turning point was not a success, but a failure.

By **late July 2025**, I had:

- Extended simulations to long horizons  
- Trained imitation learning models  
- Introduced PPO  
- Expanded scenarios to circular, elliptic, transfer, and spiral-in orbits  

What I observed repeatedly was the same pattern:

- Training appeared stable  
- Rewards did not explode  
- Short-term trajectories looked reasonable  
- **Long-term behavior inevitably diverged or collapsed**

This was my **first real frustration** of 2025.

The failure did not come from syntax errors or broken code. It came from something more unsettling:

> **I no longer knew whether “pushing forward” would actually generate new information.**

This phase forced a critical realization:

> **A model running smoothly is not evidence that a control problem has been solved.**

This was the first time I clearly distinguished **training stability** from **control correctness**.

---

## 4. Third Time Anchor — Late 2025 (Day 40+): From Pushing Forward to Maintaining Judgment

Around **late 2025 (approximately Day 40+)**, I made a deliberate and difficult decision:

> **Further feature expansion was no longer increasing information density.**

At that point, the project changed direction:

- From daily feature pushing → weekly consolidation  
- From adding new modules → freezing baselines  
- From “making models stronger” → explaining why they fail  
- From momentum-driven progress → system maintenance and diagnostics  

This was not a retreat. It was the first time I consciously treated engineering and research as judgment-driven processes rather than forward motion.

---

## 5. A Year of Heavy AI Usage — Without Surrendering Control

Looking back honestly:

> **2025 was the most AI-assisted year of my life so far.**

While developing this project, I was also:

- Managing school coursework  
- Maintaining and starting multiple repositories  
- Working under fragmented time and energy constraints  

Under these conditions, AI functioned as:

- An acceleration tool  
- A reasoning aid  
- A potential source of over-dependence  

However, even during the most AI-assisted phases, I retained final control over:

- Physical modeling assumptions  
- Metric definitions  
- Experimental judgments  

Recognizing dependence did not invalidate the year. It became the starting point for a methodological correction.

---

## 6. What Was Actually Completed in 2025

From an engineering standpoint, I completed and validated:

- A physically consistent 2D orbital simulation system  
- A clear baseline family with lower and upper bounds  
- A reproducible and auditable evaluation pipeline  
- A structured failure taxonomy with geometry–energy diagnostics  

More importantly:

> **I learned how to distinguish engineering issues, modeling issues, and method-level misalignment.**

This judgment capability is the most valuable outcome of 2025.

---

## 7. What Was Intentionally Not Completed

Based on a conscious trade-off between system complexity and judgment maturity, the following directions were deliberately deferred to 2026:

- Full 3D / 6-DOF dynamics  
- Forcing end-to-end RL success cases  
- Over-engineered reward shaping and network architectures  

Before judgment stabilizes, scaling only amplifies error.

---

## 8. Looking Toward 2026 — A Methodological Shift

My primary objective for 2026 is **not stronger AI**.

It can be summarized as:

> **Gradually reducing structural dependence on AI, allowing AI to serve as an assistant rather than a prerequisite.**

Concrete goals include:

1. Advancing core ideas even without AI assistance  
2. Achieving a clean, closed-loop success at the 2D model level  
3. Selectively integrating components from other repositories into this project  
4. If time permits late in the year:  
   - Introducing C++ components  
   - Beginning preliminary 3D modeling  

All extensions remain subordinate to understanding, not scale.

---

## 9. A Note to My Future Self

If you are reading this years later, remember:

> **2025 was not the year you were strongest — it was the year you first learned how to judge.**

If you ever doubt whether you truly understand a system, return to the 2025 failure catalog and energy diagnostics. That was the first stable ground.

---

## Closing

I did not solve autonomous spacecraft control in 2025.

But I achieved something more foundational:

> **I built a research-grade engineering framework capable of evolving for the next decade without relying on illusion.**

This was not the finish line.

It was the first moment of standing firmly on my own reasoning.

—  
**2026: move forward, but with clarity.**
