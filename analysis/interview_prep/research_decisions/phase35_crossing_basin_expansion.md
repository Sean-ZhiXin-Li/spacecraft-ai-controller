## Research Question

Can local upstream shaping expand the number of target-radius crossings while preserving the Phase34 post-cross recoverability behavior?

---

## Why was this question important?

Phase34 solved the tested downstream recoverability gap for cases that already crossed, but it did not create crossings for the remaining non-crossing cases. Phase35 asked whether local upstream modifications could expand the crossing basin.

---

## Previous evidence

The audit cites `analysis/phase34_post_cross_sync/summary.md` and `analysis/phase35_crossing_basin_expansion/summary.md`. Phase34 `radius_priority` produced `8 / 24` crossings and `8 / 24` recoverable crossings, leaving `16 / 24` non-crossing cases.

---

## Competing hypotheses

- Radial energy push would help non-crossing cases reach target radius.
- Tangential corridor entry would improve approach geometry.
- Predictive crossing bias would expand crossings while preserving Phase34 success cases.
- Local upstream shaping would be insufficient because the missing behavior is more global.
- Some variants would create crossings but introduce overspeed or regression failures.

---

## Why was this experiment designed this way?

Phase35 kept the Phase34 terminal/post-cross controller fixed and tested upstream variants on the same 24-case reduced benchmark. The variants included baseline Phase34, radial energy push, tangential corridor entry, and predictive crossing bias. Metrics included crossings, recoverability, non-crossing behavior, crossing potential, and overspeed.

---

## What result was expected?

The reasonable hope was that at least one upstream shaping variant would create new crossings while preserving the `8 / 24` Phase34 recoverable crossings. A cautious expectation was that local shaping might not be enough and could damage known good behavior.

---

## What actually happened?

The audit records that baseline and predictive bias each produced `8 / 24` crossings. Radial energy push and tangential corridor each produced `0 / 24` crossings. Radial energy push produced `5` overspeed cases.

---

## Interpretation

Phase35 supports a negative conclusion: the tested local upstream variants did not expand the crossing basin. Predictive bias preserved baseline count but did not improve it. Radial energy push and tangential corridor were worse, and radial energy push introduced overspeed risk.

---

## Alternative explanations

- The variant gains may have been poorly chosen.
- The local variables may have been the wrong abstraction for upstream transfer.
- The 24-case benchmark may contain cases requiring a more global trajectory plan.
- The variants may have interacted badly with the fixed Phase34 terminal controller.

---

## Reviewer criticism

Reviewer #2 would say the negative result is useful but limited: failure of these variants does not prove local shaping cannot work. They would ask whether the design space was broad enough and whether overspeed was caused by a simple gain error.

---

## Sean's response

The honest response is that Phase35 falsified several specific upstream modifications, not the entire idea of upstream shaping. The important outcome was that the project did not count proxy intent as success: crossing count did not improve, and one variant introduced overspeed.

---

## If you repeated this experiment

I would add a small parameter sweep for each variant, record per-case approach geometry, and require regression preservation of the known `8 / 24` crossing cases before treating any variant as promising.

---

## Future direction

The next logical experiment was Phase36B: move beyond local shaping variants and test named transfer families with Phase34 terminal behavior fixed.

---

## Scientific maturity score

Question quality: 8/10. It targeted the correct next bottleneck after Phase34.

Experimental design: 7/10. Keeping Phase34 fixed was strong, but variant coverage was limited.

Evidence quality: 8/10. The counts and overspeed result are clear.

Interpretation: 8/10. The negative conclusion is appropriately narrow.

Claim safety: 8/10. Safe if stated as "tested variants failed," not "upstream shaping cannot work."

Overall: 7.8/10. A useful negative phase that protected the project from overclaiming Phase34.
