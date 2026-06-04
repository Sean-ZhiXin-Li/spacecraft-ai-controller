# Engineering Audit

## Overall Assessment

The repository is productive but increasingly phase-heavy. It has enough structure for an independent research project, but not yet enough modularity for lab-grade reproducibility.

## Strengths

- Outputs are organized by phase under `analysis/`.
- CSVs are present for key benchmark results.
- Markdown summaries accompany most phases.
- Plots are useful for trajectory interpretation.
- Phase scripts generally preserve prior behavior rather than destructively modifying older phases.

## Weaknesses

The main engineering weakness is duplication. Phase34, Phase35, and Phase36A each contain similar rollout logic, state-machine handling, metrics, and plotting code.

Approximate script sizes inspected:

- Phase31 global transfer solver: about `1699` lines
- Phase32 direct optimal control: about `700` lines
- Phase33 structure extraction: about `485` lines
- Phase34 post-cross sync: about `941` lines
- Phase35 crossing basin expansion: about `1037` lines
- Phase36A transfer-family visualization: about `1034` lines

This is too much duplicated experiment infrastructure.

## Reproducibility Risks

Risks if another person tries to rerun the repo:

- environment assumptions are not fully centralized
- older generator templates can overwrite manually cleaned markdown wording
- constants are imported from previous phase scripts rather than from a stable benchmark module
- plotting depends on local Matplotlib behavior and output directories
- representative-case selection is encoded in scripts, not in a standalone benchmark manifest

## Naming and Output Hygiene

The phase naming is mostly clear from Phase31 onward, but the repo contains many older project logs. That history is useful internally but noisy for public readers.

The best public structure is:

- README
- `docs/research_direction.md`
- Phase31 through Phase36A analysis outputs
- PL34 and PL35 logs
- the current audit files

Older logs should remain available but not be treated as the main path.

## Plot Quality

The plots are useful and generally readable. Phase36A's geometry plots are a good direction because they show trajectory shape rather than only score bars.

Plot risks:

- too many plots can become dashboards rather than evidence
- representative-case plots can visually overstate subset behavior
- legends and event markers must remain clear as families increase

## Recommended Engineering Step

Before Phase37 or advanced planners, extract a shared 2D benchmark core:

- dynamics step
- initial condition builder
- common termination logic
- Phase34 terminal controller
- metric computation
- CSV writing schema

This would reduce phase bloat and make future comparisons more credible.

## Verdict

Engineering maturity is adequate for rapid research iteration but below lab-grade infrastructure. The project should pause feature growth long enough to standardize the benchmark and reduce duplicated rollout logic.

