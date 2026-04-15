# PPO Transfer Results

## Goal

Transfer the successful phase-structured explicit controller into a learned policy using:

1. behavior cloning
2. PPO fine-tuning from the cloned policy

This stage kept the project baseline fixed:

- `dt = 50`
- `max_steps = 100000`
- `r0_over_target = 1.00005`
- `thrust_scale = 20000`
- strict success thresholds:
  - `tol_r = 0.001`
  - `tol_v = 0.001`
  - `tol_ang = 0.02`
  - `success_threshold = 200`

## Training Setup

### Behavior cloning

Dataset:

- `analysis/phase_controller_dataset/phase_controller_dataset_balanced.npz`

Balanced dataset stats:

- total samples: `3063`
- per phase target count: `1021`
- phase balance:
  - `DESCENT`: `264405 -> 1021`
  - `CAPTURE`: `101 -> 1021`
  - `LOCK`: `1021 -> 1021`

Model:

- MLP with PPO-compatible actor structure
- input: normalized observation
- output: action
- loss: MSE

Training result:

- best validation loss: `0.0681`
- epochs run: `133`
- saved model: `models/bc_policy.pth`

### PPO fine-tuning

Initialization:

- actor/shared weights loaded from `models/bc_policy.pth`

Fine-tune setup:

- short run
- low learning rate
- no reward or physics changes

Saved model:

- `models/ppo_bc_finetuned.pth`

## Fixed-Baseline Evaluation

Artifacts:

- `analysis/figs/bc_policy/explicit_policy_metrics.json`
- `analysis/figs/bc_policy/bc_policy_metrics.json`
- `analysis/figs/bc_policy/ppo_policy_metrics.json`
- `analysis/figs/bc_policy/probe_policy_metrics.json`

### Controller comparison

| Controller | Learned? | radius_crossings_total | first_crossing_step | success | final_radius_error |
|---|---:|---:|---:|---:|---:|
| Explicit phase controller | no | 1 | 96173 | true | 31433.54 |
| Behavior cloning | yes | 0 | - | false | 375039922.79 |
| PPO fine-tuned from BC | yes | 0 | - | false | 374964010.17 |
| Probe max retrograde | no | 1 | 96173 | false | 8143288.72 |

## What Each Stage Actually Learned

### Explicit controller

The explicit controller still provides the full successful structure:

- `DESCENT`: physically effective retrograde energy removal
- `CAPTURE`: post-crossing regulation
- `LOCK`: final strict stabilization

This is the only controller in this comparison that reaches both:

- real crossing
- strict success

### Behavior cloning

Behavior cloning learned a policy that matches the explicit controller locally well enough to achieve a low validation MSE, but it did not reproduce the long-horizon phase transition behavior on rollout.

Observed result:

- no crossing
- no success
- large residual final radius error

Interpretation:

- the cloned policy learned part of the action manifold
- but it did not recover the full sequential control structure required for:
  - first crossing
  - phase transition
  - post-crossing stabilization

### PPO fine-tuning from BC

PPO fine-tuning did not recover the missing structure.

Observed result:

- still no crossing
- still no success
- final radius error improved only slightly relative to BC

Improvement over BC:

- `375039922.79 -> 374964010.17`

That difference is negligible at the project scale and does not change the qualitative behavior.

## Why PPO Still Fails Here

The main problem is not raw function approximation.

The problem is that successful insertion depends on a sequence of distinct control regimes:

1. remove enough orbital energy to guarantee crossing
2. detect that crossing has happened
3. switch to post-crossing damping and support
4. settle into a narrow lock region

The explicit controller encodes that structure directly.

The learned policies in this stage do not yet reproduce those transitions.

Most likely reasons:

- the balanced imitation dataset greatly reduced raw `DESCENT` coverage to enforce phase balance
- one-step supervised action matching is not enough to guarantee multi-step phase consistency
- short PPO fine-tuning from that BC initialization is still too weak to discover the missing transition logic on its own

## Where PPO Improves

In this transfer stage, PPO does **not** improve in the way that matters for insertion.

What improved:

- a very small reduction in final radius error relative to pure BC

What did **not** improve:

- no first crossing
- no success
- no demonstrated recovery of `CAPTURE` or `LOCK`

So the current PPO fine-tuning stage should be treated as:

- a valid initialization experiment
- not yet a successful transfer of the explicit controller structure

## Conclusion

The project now has a clear transfer result:

- the explicit phase controller contains the correct control structure
- behavior cloning alone does not yet recover that structure
- short PPO fine-tuning from BC also does not yet recover it

At this stage, the only stage that is fully successful is still the explicit controller.

## Recommended Next Step

Do not go back to PPO from scratch.

The next learning step should be:

1. keep behavior cloning as the initialization path
2. improve imitation data coverage so `DESCENT` remains physically effective while `CAPTURE` and `LOCK` stay represented
3. then fine-tune PPO on the fixed successful baseline after the cloned policy can already produce the first crossing

In short:

- explicit structure is solved
- learning transfer is not solved yet
- the next target is **recover crossing before optimizing lock**
