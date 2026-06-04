# Phase Progression and Architecture Analysis

## Overall Progression

The phase progression is coherent at the research level:

1. Try learned or reactive policies.
2. Discover that crossing and insertion are not the same.
3. Use trajectory-family and optimal-control probes to identify structure.
4. Implement post-cross synchronization as a terminal controller.
5. Test whether local pre-cross biases expand the crossing basin.
6. Move toward transfer-family search.

This is a real scientific arc.

## Phase20

Phase20 is useful as historical context for predictive local planning. It should not be the center of the public narrative. Its value is that it helped motivate the later claim that short-horizon local action selection is not enough.

Public role: internal exploratory history.

## Phase31

Phase31 is important. It tested named global transfer families and showed that crossings were possible without recoverable crossings.

Evidence:

- Phase31 `phase31_phase22_baseline`: `12 / 48` crossings, `0 / 48` recoverable crossings.
- Other Phase31 families also failed to produce recoverable crossings.

Public role: core narrative, because it establishes the crossing/recoverability gap.

## Phase32

Phase32 is important but must remain carefully scoped. It used SciPy direct shooting, not full CasADi/IPOPT direct collocation. Its value is upper-bound probing: it showed recoverable states were physically reachable under the simplified dynamics.

Public role: core narrative with caveat.

## Phase33

Phase33 is one of the strongest scientific phases. It extracted structure rather than just reporting a score.

Evidence:

- first crossing step: `81`
- first crossing sync error: `1.676881`
- best recoverable state step: `512`
- best sync error: `0.000464`
- best state occurred after first crossing

Public role: core narrative.

## Phase34

Phase34 is the central architecture result. It converted crossing-producing cases into recoverable crossings without expanding the crossing set.

Evidence:

- Phase31-style reference: `8 / 24` crossings, `0 / 24` recoverable crossings.
- Phase34 `radius_priority`: `8 / 24` crossings, `8 / 24` recoverable crossings.

Public role: core narrative.

## Phase35

Phase35 is a valuable negative result. It showed that local pre-cross biases did not expand crossing-producing cases.

Public role: core narrative, because it prevents the next step from being another local-gain tuning phase.

## Phase36 Research Context

The Phase36 research context is conceptually appropriate. It correctly shifts from local steering to transfer-family discovery.

Public role: forward-looking research plan.

## Phase36A

Phase36A is appropriately scoped as visualization-first. Its subset results do not improve crossing count, but they make family geometry visible.

Evidence:

- baseline, spiral, and grazing each produced `1 / 3` crossings and recoverable crossings.
- delayed produced `0 / 3` crossings without overspeed.
- energy_bleed, overshoot_return, and two_stage each produced `0 / 3` crossings with `3 / 3` overspeed.

Public role: early exploratory analysis, not benchmark proof.

## Noise Versus Insight

Phases that most advanced understanding:

- Phase31
- Phase32
- Phase33
- Phase34
- Phase35

Phases most at risk of being perceived as noise:

- any phase that only adds a named heuristic without changing the scientific question
- Phase36A families if not followed by full benchmark structure

## Recommendation

The public narrative should emphasize Phase31 through Phase36A. Older PPO and local phases should be summarized as context, not presented as equally central.

