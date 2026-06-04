# Research Maturity Audit

## Overall Assessment

As an exploratory controls/autonomy project, this is substantially more mature than a typical script collection. It shows a real habit of hypothesis, experiment, negative result, and architecture revision.

It is not yet a formal lab-grade research codebase.

## Genuinely Impressive

- The project moved away from reward chasing when PPO-style progress was insufficient.
- It identified the difference between geometric crossing and recoverable insertion.
- It used direct shooting as an upper-bound probe rather than pretending heuristics were enough.
- It extracted structure from an optimal trajectory before implementing Phase34.
- It reported Phase35 as a negative structural result.
- It kept Phase36A scoped as visualization-first.

## Genuinely Weak

- The codebase is too phase-heavy.
- Rollout logic is duplicated.
- Benchmark definitions are spread across scripts.
- Some metrics are hand-weighted and heuristic.
- Representative subsets are useful but could be misread by readers.
- Public narrative still depends on careful wording to avoid overclaiming.

## Lab Mentor Reaction

A lab mentor would likely take the project seriously because it shows:

- honest negative results
- structured control reasoning
- meaningful simulator metrics
- clear evolution from PPO to trajectory geometry
- a sensible next research question

A lab mentor would also immediately flag:

- lack of reusable experiment infrastructure
- no formal benchmark manifest
- simplified physics
- no uncertainty or robustness testing
- too many one-off phase scripts

## Public Versus Internal

Show publicly:

- README
- research direction document
- Phase31 through Phase36A summaries
- PL34 and PL35 logs
- key plots showing geometry and post-cross sync
- accuracy and goal-mode audits

Keep mostly internal:

- very old logs
- failed local-tuning phases unless summarized
- raw exploratory scripts that do not support the main scientific arc

## Verdict

The project is strong as a serious independent research effort and plausible as early undergraduate lab-readiness work. It is not yet a polished research artifact, but the intellectual direction is real.

