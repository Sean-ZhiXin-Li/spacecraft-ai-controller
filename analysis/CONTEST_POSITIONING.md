# Contest Positioning — Orbit Decision & Diagnosability System

## 1. Innovation Point

This project is not a one-off orbit simulation or a standard reinforcement learning demo. Instead, it builds a control evaluation system where different controllers can be compared under the same orbital dynamics, and where failures and misleading improvements are diagnosable rather than hidden.

A key innovation is the introduction of saturation-aware evaluation metrics (Session 6). In thrust-limited orbital control, different controllers can appear artificially equivalent when evaluated only by episode reward, especially under scaling or saturation effects. To address this, the system explicitly records saturation_rate while keeping reward informative instead of trivially collapsing it to zero.

These metrics are designed to be continuous, monotonic, and interpretable. As a result, evaluation reflects real control trade-offs—such as accuracy versus actuation limits—rather than surface-level performance, enabling reliable and defensible comparisons between control strategies.

---

## 2. Technical Difficulty

The main technical difficulty of this project is not making an orbit simulation run, but determining which controller is genuinely better. Orbital dynamics operate on long time scales with weak and delayed feedback, making naive short-horizon signals unreliable.

At the same time, control inputs are constrained by thrust saturation and scaling. Under these constraints, multiple controllers may achieve similar rewards while exhibiting fundamentally different control behaviors. This creates a false sense of equivalence that cannot be resolved by code complexity alone.

To overcome this, evaluation itself is treated as a first-class engineering problem. By introducing saturation-aware metrics, the system exposes hidden behavioral differences and transforms controller comparison from an anecdotal judgment into a reproducible, system-level assessment.

---

## 3. Why AI Is Necessary

The project intentionally begins with a rule-based expert controller as a baseline, providing a clear and interpretable reference point. However, the control objective in this setting lacks a clean analytic solution: it is nonlinear, constrained by actuation limits, and requires long-horizon planning.

AI becomes necessary because the decision policy must learn trade-offs that are difficult to hand-design across scenarios. These include when to apply thrust, how much thrust to apply, and how to balance orbit error against limited actuation under saturation and scaling constraints.

The saturation-aware metrics introduced in Session 6 provide the technical foundation for this learning-based approach. They prevent reward-only false equivalence and allow learning progress to be evaluated in terms of interpretable control behavior rather than a single opaque score, making AI a necessary component rather than a decorative addition.

---

## 4. Why This Is Not a Toy Simulation

This project is not evaluated by visual appeal or single-episode success, but by controlled metrics, reproducibility, and diagnosability. Ablation studies (e.g., raw versus prescale control) are used to isolate the effects of thrust scaling and saturation instead of attributing improvements to uncontrolled changes.

Failures are systematically recorded through a failure taxonomy, turning unsuccessful runs into structured data rather than ignored noise. In addition, a shadow step mechanism is used to verify that control actions genuinely take effect in the environment, preventing silent no-op behavior from being misinterpreted as valid control.

Finally, the entire experimental workflow is designed for reproducibility. Configuration files, JSON-based logging, and fixed execution pipelines ensure that results can be rerun and audited, reinforcing that this system is an engineering and research artifact rather than a toy simulation.
