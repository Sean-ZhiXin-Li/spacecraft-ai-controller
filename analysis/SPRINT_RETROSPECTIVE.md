# Sprint Retrospective (Session 8 — Jan 4)

This document marks the formal close of the current engineering sprint.
The goal is not to extend the project, but to **freeze what is now solid**, clarify what changed, and define a clean entry point for the next phase.

---

## What Changed?

The most important shift in this sprint was not numerical performance, but **epistemic clarity** — I now know *what kind of system this project actually is*.

Earlier, the project was evaluated primarily by whether it could run and whether reward improved. During this sprint, evaluation itself became a first-class object. By introducing saturation-aware metrics, ablation-style comparisons, and explicit scenario handling, I moved from *"the controller runs"* to *"the controller can be judged."*

Another key change was the separation between **physical limitations** and **controller logic**. By explicitly logging saturation rate and decoupling thrust intent from action clipping, it became clear when poor outcomes were caused by actuation limits rather than algorithmic failure. This reframed earlier results that looked like controller collapse as, in fact, regime mismatch.

Finally, the project transitioned from an implicit AI narrative to an explicit one. I no longer assume AI is useful everywhere; instead, I can point to *where decision-making under constraints becomes the bottleneck*, and where learning-based methods may become justified.

---

## What Is Now Solid?

Several components of the project are now considered **stable assets** rather than exploratory prototypes.

At the system level, the end-to-end pipeline — orbital simulation, controller interface, action normalization, metric logging, and reproducibility scripts — is coherent and internally consistent. This pipeline can support controlled comparisons without structural changes.

At the evaluation level, metrics are no longer singular or opaque. Reward is supplemented by saturation rate, radius error statistics, and saturation-adjusted scores. These metrics are monotonic, interpretable, and sufficient to support contest or research-style arguments.

At the methodological level, rule-based controllers are now understood as *necessary baselines*, not failed AI attempts. They define the boundary where physics-informed heuristics end and learning-based strategies may begin. This framing is critical for future AI integration.

These elements are now frozen: future work should build *on top of them*, not rework them.

---

## What Is Next (Week 8+)?

The next phase is **not** about adding features or increasing complexity. Its purpose is to enter a regime where controller choices produce visibly different outcomes.

The immediate objective is to construct at least one controlled scenario in which:

* Physical parameters remain unchanged at the global level
* Actuation strength varies within a narrow, realistic band
* Controller behavior produces divergent error trajectories

This will establish that the control logic is not only well-structured, but *causally relevant*.

Large-scale reinforcement learning and end-to-end training are intentionally postponed. Until the environment reliably exposes decision-sensitive regimes, learning would add opacity without insight.

---

## Phase Status

* Sprint status: **Closed**
* Core pipeline: **Frozen**
* Evaluation framework: **Operational**
* Next phase: **Controlled sensitivity exploration**

This document serves as a phase boundary. Future commits should assume this state as baseline truth.
