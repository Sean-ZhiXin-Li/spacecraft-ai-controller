# One-Page Research Summary (v0.1)

## 1) Problem

Modern spacecraft control evaluations are frequently affected by implementation-level choices at the action interface, such as action scaling and clipping. When performance differences are observed, it is often ambiguous whether these differences originate from controller logic or from interface-induced artifacts. This attribution ambiguity undermines the reliability of controller comparisons, particularly in long-horizon orbital simulations where small interface effects can accumulate over time.

## 2) System Design

Experiments are conducted using a physics-based orbital simulator with two rule-based expert controllers (ExpertV3 and ExpertImproved). Controlled execution is enforced through `ACTION_IF_MODE`, which switches the action interface between `raw` and `prescale` while holding all other variables fixed. All runs share an identical configuration (`default.yaml`), including physics parameters, reward definition, and numerical integration settings. This design explicitly isolates the action interface as the sole independent variable, enabling unambiguous attribution of observed outcomes.

## 3) Key Finding

Under the default scenario with a 2000-step horizon, changing the action interface from `raw` to `prescale` yields no measurable difference in saturation rate, average radius error, or total reward for either controller. This result indicates that, within the tested configuration, action scaling does not act as a confounding factor in observed control performance.

## 4) Evidence

All evidence is derived from completed and logged experimental runs. Week7 ablation experiments re-execute the same scenario while varying `ACTION_IF_MODE` between `raw` and `prescale`. The mean action saturation rate remains identical (0.50) across both interface modes for ExpertV3 and ExpertImproved (Fig. 1). Complementary energy-level diagnostics recorded in `metrics_energy.json` show negligible energy drift (−0.0095%), a low energy oscillation index (~5×10⁻⁵), and near-zero final angular momentum error (−0.0048). Collectively, these measurements demonstrate that action interface scaling introduces no observable artifacts at either the action-clipping level or the orbital dynamics level under the evaluated conditions.

## 5) Next Steps

Future work will extend the same interface ablation framework to stress scenarios, such as weak-thrust and misaligned-entry conditions, to assess whether interface effects emerge outside the nominal regime. Only after validating interface invariance across such regimes will differences in controller logic be evaluated as the primary explanatory factor.
