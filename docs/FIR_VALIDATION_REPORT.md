# Independent Validation of the FIR/SPIRE pop-cosmos Extension

**Scope.** I re-derived your headline result from the raw catalogues and code — flux
prediction physics, unit conversions, the count evaluator, and every robustness lever —
rather than trusting the stored scorecards. Bottom line: **the core result holds and is
well-supported.** Baseline FSPS overpredicts SPIRE differential counts; a warmer/broader
FIR dust SED at fixed `L_IR` fixes it. I also found four things to tighten before
submission — none of them overturn the result, but two of them change the *numbers* you
quote and one changes the *wording*.

Every number below was recomputed by me from `master.dat.gz`, the compiled count table,
`fsps_map_median_full.h5`, and the prediction pickle. Where my number differs from a
stored file I say so.

---

## 1. What is solid

### 1.1 The forward-model physics is correct
- **Flux prediction** (`predict_*_flux_mjy`): the chain `L_IR → shape/∫shape dν →
  Lν → Fν = (1+z)Lν/(4π dL²) → mJy` is dimensionally correct, the `(1+z)`
  k-correction for a νLν-normalised template is right, and `WMAP9` luminosity distance is
  used consistently. The optically-thin MBB (`ν^β B_ν`, β=1.8) and the Casey join
  (power-law matched to the MBB where its log-slope equals α) are standard.
- **Rest-frame vs flux-space refactor is exact.** `popcosmos_restframe_hybrid_method_check`
  shows the rest-frame SED mixing reproduces the old flux-space hybrid to a **max
  fractional difference of 3×10⁻⁵** across all 430k galaxies and 3 bands. The method
  cleanup introduced no error.
- **Count unit conversions are internally consistent.** Pearson (`×1e7`, mJy^1.5→Jy^1.5),
  Oliver (raw dN/dS ×S^2.5), Glenn/Varnish (log10→linear), and the sr→deg² factor all
  check out. **Independent cross-check:** where two *different* published sources overlap
  in flux and band, they agree to a **median 0.12 dex (p90 0.26 dex)**. A unit error in
  any one source would show up as a ~1-dex offset against the others; the worst pair is
  0.40 dex, which is ordinary field/method variance. The conversions are clean.

### 1.2 The core result is robust — this is your thesis
- **FSPS is too bright in every independent source, every band.** Median
  `log10(FSPS/obs)` is **positive in 7/7 sources at all three bands**:
  +0.25 / +0.41 / +0.55 dex (median over sources) at 250 / 350 / 500 µm, worsening
  monotonically with wavelength. This unanimity across independent fields (H-ATLAS,
  HerMES, SPIRE Dark Field, COSMOS), surveys and extraction methods is the strongest
  single fact in the project — far more convincing than any one χ² value.
- **The excess exceeds the systematic floor.** The FSPS offset is **2.6–3.3× larger**
  than the measured inter-survey scatter (0.25 dex vs 0.09 at 250; 0.55 vs 0.21 at 500).
  So it cannot be blamed on catalogue-to-catalogue disagreement.
- **Warmer/broader dust fixes it, and the ranking is stable.** On a single common
  174-point set spanning all 9 count tables, χ²/N improves from **9.7 (FSPS) → 3.4
  (Casey T30K α=2.5)**, and the median offset goes from +0.28 dex to −0.06 dex. That
  ranking is unchanged: with vs without the model Poisson term (Spearman ρ=0.99), across
  error floors 0.08–0.20 dex, under χ²/N vs χ²/(N−k) (ρ=0.999), on the clean-independent
  subset, and when restricted to the well-populated 10–100 mJy regime.
- **Direct SED evidence closes the loop.** Over a 4,000-galaxy random sample of the
  baseline FSPS SEDs, the FIR dust bump peaks at a **median of ~136 µm rest-frame**, with
  **64% of galaxies peaking beyond 100 µm** (cold, long-wavelength-heavy) — see
  `fsps_fir_sed_diagnostic.png`. The distribution is bimodal: a real **~20% warm/hot tail
  peaks below 40 µm** (these are the higher-`L_IR`, more AGN-affected objects), so this is
  a population-level statement, not a single "typical" SED. Cold-dominated FIR emission is
  *mechanistically* what a 350/500 µm count excess requires. You are not just fitting a
  warmer template because it scores better; the baseline SED shape independently shows the
  population skews too cold.

---

## 2. What to fix before submission

### 2.1 (numbers) The stored Casey/MBB scorecards predate Valiante — rerun on one point set
The saved `casey_like...scorecard.csv` and `mbb...scorecard.csv` were computed on **6
sources**; the hybrid evaluator ran on **9** (it includes the three Valiante tables). So
the headline "FSPS reduced χ² 8.40 → best 3.36" in your supervisor note mixes two
different point sets and is not internally comparable.

When I recompute **all** models on **one** common point set the story is identical (FSPS
worst, Casey T30K best), so nothing scientific changes — but the thesis must quote every
model's score from a single consistent run. **Action:** rerun the Casey and MBB grids now
that the Valiante rows exist, or state explicitly that the baseline and template numbers
come from the same evaluator pass. I've saved a corrected `validation_unified_scorecard.csv`
with every model on the common set for both the all-9 and clean-independent definitions.

### 2.2 (numbers) χ²/N ≈ 3–10 for "good" models means the error floor is too tight
Your 0.08 dex floor is *smaller* than the inter-survey scatter you can actually measure
(median 0.12, up to 0.4 dex). That's why even the best template sits at χ²/N ≈ 2–3 rather
than ≈1 — the model isn't the problem, the assumed errors are. Adopting a **0.12–0.15 dex
floor, justified by the measured inter-survey scatter**, brings the best models to
χ²/N ≈ 1.3–1.6 ("consistent within systematics") while leaving the ranking untouched (see
`validation_error_floor_sensitivity.csv`). This is a cleaner and more defensible story
than reporting χ²/N ≈ 3 and calling it a good fit. Keep calling it a *rough* evaluator.

### 2.3 (caveat) Poisson-starved bright end — lead with 10–100 mJy
Above ~150 mJy the model curve has ≤4 objects/bin (Poisson error 0.2–0.4 dex) and above
300 mJy essentially none (the lone 500 µm/838 mJy model object is noise). The largest FSPS
residuals you quote (+0.69 to +1.08 dex at 100–300 mJy) sit exactly where **both** the
model and the observed counts are sparse. The result does **not** depend on them:
restricting to 10–100 mJy still puts FSPS clearly worst and Casey T30K best. **Action:**
make 10–100 mJy the quantitative anchor and present 100–300 mJy as corroborating/qualitative,
noting the small-N caveat. This pre-empts the obvious examiner question.

### 2.4 (wording) The energy-balance "check" is tautological
`fsps_lir_ratio` = 1.000 for all 430k galaxies because the SED file was already
`L_IR`-normalised. This confirms your renormalisation code is applied correctly — state it
that way — but it is **not** an independent validation of the FSPS `L_IR` values, so don't
present it as one.

### 2.5 (minor, verified OK) Wang-match sample restriction does not bias the counts
Model counts are built on the 26.5% of pop-cosmos objects matched to Wang positive IDs,
normalised over the Wang/Farmer 1.278 deg². The match drops **~1.6–1.9% of model sources
in aggregate (N-weighted) over 10–300 mJy per band** — negligible where the counts carry
statistical weight (the well-populated 10–100 mJy regime, which drives the result). The
loss is *not* uniform bin-to-bin: individual sparse bright bins lose more (e.g. ~9% at
250 µm/100 mJy, ~12% at 250 µm/143 mJy, ~20% at 350 µm/204 mJy), because those bins hold
only a handful of objects so removing one or two matters proportionally. This is the same
Poisson-starved bright end flagged in §2.3 and reinforces that conclusion. **Action:** add
one sentence to Methods stating the match selection removes <2% of sources in aggregate
above 10 mJy and is negligible in the 10–100 mJy anchor regime, while noting the sparse
bright bins are governed by small-N rather than by the match.

---

## 3. Deliverables from this validation
- `fsps_fir_sed_diagnostic.png` — baseline FSPS FIR SEDs peak at ~136 µm (the physical
  root cause). Candidate thesis figure.
- `validation_unified_scorecard.csv` — every model scored on one common point set (all-9
  and clean-independent), with χ²/N, χ²/(N−k) and median offset. Use this to replace the
  mixed-point-set numbers.
- `validation_error_floor_sensitivity.csv` — headline models vs error floor; supports §2.2.
- `validation_fsps_persource_offsets.csv` — the 7/7 sign test, per source per band.

## 4. Your supervisor decision gates, answered by the data
1. **P(D) in the formal score?** Keep it separate — defensible. The ranking is identical
   with/without P(D), so nothing is lost, and the correlated-knot argument is sound.
2. **χ²/N vs χ²/(N−k)?** Immaterial — ρ=0.999 either way. But fix the *floor* (§2.2); that
   matters far more than the k correction.
3. **Is Clements enough at the bright end?** You already added Valiante (wide-area H-ATLAS
   DR1, ~162 deg²) — that *is* the stronger bright anchor. Use Valiante as the bright-end
   benchmark and Clements as the historical cross-check.
4. **Casey-like enough, or need Draine-Li/CIGALE?** Casey-like is sufficient to *demonstrate
   the direction*. The honest framing: you are not claiming a unique dust template, you are
   showing that any warmer/broader FIR SED at fixed `L_IR` moves counts the right way.
   Draine-Li/CIGALE is genuine future work, not a submission blocker.
